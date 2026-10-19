"""The ingestion orchestrator.

It depends only on ports (source, extractor, cleaners, guards, chunker,
embedder, unit of work), never on concrete engines.

Per-document flow
-----------------
1. extract -> Document
2. clean (ordered) and run guards
3. cheap idempotency check on the document content hash
4. chunk (section-aware + recursive)
5. embed **outside** any transaction (GPU work must not hold a DB txn open)
6. write document + chunks + vectors + job status in **one** transaction

When ``transactions are atomic per document``, a crash mid-run leaves the store
consistent: a document is either fully indexed or untouched; a re-run skips
everything whose content hash is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from rag_ingest.domain.models import (
    Document,
    IngestStats,
    IngestStatus,
    RunManifest,
)
from rag_ingest.logging_setup import get_logger
from rag_ingest.ports.chunker import Chunker
from rag_ingest.ports.cleaner import Cleaner, Guard
from rag_ingest.ports.embedder import Embedder
from rag_ingest.ports.extractor import Extractor
from rag_ingest.ports.source import SourceReader
from rag_ingest.ports.storage import UnitOfWork

log = get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        *,
        reader: SourceReader,
        extractor: Extractor,
        cleaners: list[Cleaner],
        guards: list[Guard],
        chunker: Chunker,
        embedder: Embedder,
        uow: UnitOfWork,
        split: str,
        run_id: str | None = None,
        log_every: int = 25,
    ) -> None:
        self._reader = reader
        self._extractor = extractor
        self._cleaners = cleaners
        self._guards = guards
        self._chunker = chunker
        self._embedder = embedder
        self._uow = uow
        self._split = split
        self._run_id = run_id or uuid4().hex
        self._log_every = max(1, log_every)

    @property
    def run_id(self) -> str:
        return self._run_id

    def run(
        self,
        limit: int | None = None,
        on_progress: Callable[[IngestStats], None] | None = None,
    ) -> IngestStats:
        """Ingest the reader's records.

        ``on_progress`` is called after every document with the running stats,
        so callers (CLI, scripts) can render their own progress UI without the
        pipeline knowing anything about it.
        """
        stats = IngestStats()
        self._start_run()
        try:
            for raw in self._reader.read():
                if limit is not None and stats.documents_seen >= limit:
                    break
                stats.documents_seen += 1
                self._process_one(raw.filename, raw, stats)
                if on_progress is not None:
                    on_progress(stats)
                if stats.documents_seen % self._log_every == 0:
                    log.info(
                        "progress seen=%d indexed=%d skipped=%d failed=%d chunks=%d",
                        stats.documents_seen,
                        stats.documents_indexed,
                        stats.documents_skipped,
                        stats.documents_failed,
                        stats.chunks_written,
                    )
        finally:
            self._finish_run(stats)
        log.info(
            "run %s finished: %s",
            self._run_id,
            stats.model_dump(),
        )
        return stats

    # -- internals ---------------------------------------------------------
    def _process_one(self, doc_id: str, raw: object, stats: IngestStats) -> None:
        document: Document | None = None
        try:
            document = self._extractor.extract(raw)  # type: ignore[arg-type]
            for cleaner in self._cleaners:
                document = cleaner.clean(document)

            for guard in self._guards:
                decision = guard.check(document)
                if not decision.allowed:
                    reason = f"blocked by {guard.name}: {decision.reason}"
                    self._record(doc_id, IngestStatus.SKIPPED, reason)
                    stats.documents_skipped += 1
                    return

            if document.is_empty:
                self._record(doc_id, IngestStatus.SKIPPED, "empty document")
                stats.documents_skipped += 1
                return

            existing = self._get_hash(doc_id)
            if existing == document.content_hash:
                self._record(doc_id, IngestStatus.SKIPPED, "unchanged content hash")
                stats.documents_skipped += 1
                return

            chunks = self._chunker.chunk(document)
            if not chunks:
                self._record(doc_id, IngestStatus.SKIPPED, "produced no chunks")
                stats.documents_skipped += 1
                return

            vectors = self._embedder.embed([c.text for c in chunks])

            with self._uow.transaction() as tx:
                tx.documents.upsert_document(document, IngestStatus.INDEXED)
                written = tx.index.replace_chunks(document, chunks, vectors)
                tx.jobs.record(self._run_id, doc_id, IngestStatus.INDEXED)

            stats.documents_indexed += 1
            stats.chunks_written += written
        except Exception as exc:  # noqa: BLE001 - per-document isolation is the point
            log.exception("document %s failed: %s", doc_id, exc)
            self._mark_failed(document, doc_id, exc)
            stats.documents_failed += 1

    def _get_hash(self, doc_id: str) -> str | None:
        with self._uow.transaction() as tx:
            return tx.documents.get_content_hash(doc_id)

    def _record(self, doc_id: str, status: IngestStatus, error: str = "") -> None:
        with self._uow.transaction() as tx:
            tx.jobs.record(self._run_id, doc_id, status, error)

    def _mark_failed(self, document: Document | None, doc_id: str, exc: Exception) -> None:
        try:
            with self._uow.transaction() as tx:
                if document is not None:
                    tx.documents.upsert_document(document, IngestStatus.FAILED)
                tx.jobs.record(self._run_id, doc_id, IngestStatus.FAILED, str(exc)[:2000])
        except Exception:  # noqa: BLE001 - never let error handling crash the run
            log.exception("could not record failure for %s", doc_id)

    def _start_run(self) -> None:
        manifest = RunManifest(
            run_id=self._run_id,
            source=self._reader.name,
            split=self._split,
            started_at=datetime.now(UTC).isoformat(),
        )
        with self._uow.transaction() as tx:
            tx.jobs.start_run(manifest)

    def _finish_run(self, stats: IngestStats) -> None:
        status = "completed" if stats.documents_failed == 0 else "completed_with_errors"
        with self._uow.transaction() as tx:
            tx.jobs.finish_run(self._run_id, status, stats)
