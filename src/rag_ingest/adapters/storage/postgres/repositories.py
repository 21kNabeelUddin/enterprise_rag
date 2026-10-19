"""Store implementations bound to one SQLAlchemy session (one transaction)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from rag_ingest.adapters.storage.postgres.tables import (
    VECTOR_DIM,
    ChunkRow,
    DocumentRow,
    EmbeddingRow,
    IngestJobRow,
    IngestRunRow,
)
from rag_ingest.domain.models import (
    Chunk,
    Document,
    IngestStats,
    IngestStatus,
    RunManifest,
)


class PostgresDocumentStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_content_hash(self, doc_id: str) -> str | None:
        return self._session.scalar(
            select(DocumentRow.content_hash).where(DocumentRow.doc_id == doc_id)
        )

    def upsert_document(self, document: Document, status: IngestStatus) -> None:
        row = self._session.get(DocumentRow, document.doc_id)
        if row is None:
            row = DocumentRow(doc_id=document.doc_id)
            self._session.add(row)
        row.cik = document.cik
        row.year = document.year
        row.source = document.source
        row.content_hash = document.content_hash
        row.status = status.value
        row.section_count = len(document.sections)
        self._session.flush()


class PostgresIndexWriter:
    """Writes chunks + embeddings + the generated keyword index."""

    def __init__(
        self,
        session: Session,
        *,
        model_id: str = "BAAI/bge-large-en-v1.5",
        revision: str | None = None,
        dim: int = VECTOR_DIM,
    ) -> None:
        self._session = session
        self._model_id = model_id
        self._revision = revision
        self._dim = dim

    def replace_chunks(
        self, document: Document, chunks: Sequence[Chunk], embeddings: np.ndarray
    ) -> int:
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"embeddings/chunks mismatch: {embeddings.shape[0]} != {len(chunks)}"
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != self._dim:
            raise ValueError(
                f"embedding dim {embeddings.shape} does not match expected {self._dim}"
            )

        # cascade deletes embeddings for the removed chunks
        self._session.execute(delete(ChunkRow).where(ChunkRow.doc_id == document.doc_id))

        for chunk, vector in zip(chunks, embeddings, strict=True):
            meta = chunk.metadata
            self._session.add(
                ChunkRow(
                    chunk_id=chunk.chunk_id,
                    doc_id=document.doc_id,
                    cik=meta.cik,
                    year=meta.year,
                    section_key=meta.section_key,
                    chunk_index=meta.chunk_index,
                    text=chunk.text,
                    token_count=meta.token_count,
                    char_start=meta.char_start,
                    char_end=meta.char_end,
                    content_hash=chunk.content_hash,
                    acl=list(meta.acl),
                )
            )
            self._session.add(
                EmbeddingRow(
                    chunk_id=chunk.chunk_id,
                    model_id=self._model_id,
                    revision=self._revision,
                    dim=self._dim,
                    embedding=np.asarray(vector, dtype=np.float32).tolist(),
                )
            )
        self._session.flush()
        return len(chunks)


class PostgresJobStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def start_run(self, manifest: RunManifest) -> None:
        self._session.add(
            IngestRunRow(
                run_id=manifest.run_id,
                source=manifest.source,
                split=manifest.split,
                status="running",
                stats=manifest.stats.model_dump(mode="json"),
            )
        )
        self._session.flush()

    def finish_run(self, run_id: str, status: str, stats: IngestStats) -> None:
        row = self._session.get(IngestRunRow, run_id)
        if row is None:
            return
        row.status = status
        row.stats = stats.model_dump(mode="json")
        row.finished_at = datetime.now(UTC)
        self._session.flush()

    def record(self, run_id: str, doc_id: str, status: IngestStatus, error: str = "") -> None:
        row = self._session.get(IngestJobRow, (run_id, doc_id))
        if row is None:
            row = IngestJobRow(run_id=run_id, doc_id=doc_id, attempts=0)
            self._session.add(row)
        row.status = status.value
        row.attempts = (row.attempts or 0) + 1
        row.error = error
        self._session.flush()
