"""Storage ports.

``UnitOfWork.transaction()`` is the ACID boundary: everything written inside a
``with uow.transaction() as tx:`` block commits together or rolls back
together. That is how a document's chunks, vectors and status stay consistent
even if the process dies mid-run.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

import numpy as np

from rag_ingest.domain.models import Chunk, Document, IngestStats, IngestStatus, RunManifest


@runtime_checkable
class DocumentStore(Protocol):
    """Document-level state: identity, content hash and status."""

    def get_content_hash(self, doc_id: str) -> str | None: ...

    def upsert_document(self, document: Document, status: IngestStatus) -> None: ...


@runtime_checkable
class IndexWriter(Protocol):
    """Writes the retrievable payload: chunks, embeddings and keyword data."""

    def replace_chunks(
        self, document: Document, chunks: Sequence[Chunk], embeddings: np.ndarray
    ) -> int:
        """Delete the document's previous chunks, then insert the new ones.

        Returns the number of chunks written.
        """


@runtime_checkable
class JobStore(Protocol):
    """Per-run and per-document progress, so runs resume and stay idempotent."""

    def start_run(self, manifest: RunManifest) -> None: ...

    def finish_run(self, run_id: str, status: str, stats: IngestStats) -> None: ...

    def record(self, run_id: str, doc_id: str, status: IngestStatus, error: str = "") -> None: ...


@runtime_checkable
class TransactionScope(Protocol):
    """Stores bound to a single open transaction."""

    documents: DocumentStore
    index: IndexWriter
    jobs: JobStore


@runtime_checkable
class UnitOfWork(Protocol):
    def transaction(self) -> AbstractContextManager[TransactionScope]: ...
