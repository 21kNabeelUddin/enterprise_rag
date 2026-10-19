"""Test doubles and helpers.

These let the pipeline be exercised with zero external dependencies (no DB, no
GPU), which is the whole point of depending on ports rather than engines.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np

from rag_ingest.domain.models import (
    Chunk,
    Document,
    IngestStats,
    IngestStatus,
    ModelFingerprint,
    RawRecord,
    RunManifest,
    Section,
)


# --- text / token fakes ---------------------------------------------------
class WordCounter:
    """Counts whitespace-separated tokens."""

    name = "word_counter"

    def count(self, text: str) -> int:
        return len(text.split())


class CharCounter:
    """Counts characters; makes token-budget tests deterministic."""

    name = "char_counter"

    def count(self, text: str) -> int:
        return len(text)


# --- embedder fake --------------------------------------------------------
class FakeEmbedder:
    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._fp = ModelFingerprint(model_id="fake", dim=dim, normalize=True)

    @property
    def fingerprint(self) -> ModelFingerprint:
        return self._fp

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self._dim).astype(np.float32)
            norm = float(np.linalg.norm(vector)) or 1.0
            out[i] = vector / norm
        return out


# --- source / extractor fakes ---------------------------------------------
class ListReader:
    def __init__(self, records: list[RawRecord], name: str = "list") -> None:
        self._records = records
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def read(self) -> Iterator[RawRecord]:
        yield from self._records


class BoomExtractor:
    """Raises for a specific filename to exercise failure isolation."""

    name = "boom"

    def __init__(self, bad: str) -> None:
        self._bad = bad

    def extract(self, record: RawRecord) -> Document:
        if record.filename == self._bad:
            raise RuntimeError("kaboom")
        from rag_ingest.adapters.extractors.passthrough import PassthroughExtractor

        return PassthroughExtractor().extract(record)


# --- storage fakes --------------------------------------------------------
class InMemoryStore:
    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.runs: dict[str, dict] = {}
        self.jobs: dict[tuple[str, str], dict] = {}


class _InMemoryScope:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self.documents = _FakeDocumentStore(store)
        self.index = _FakeIndexWriter(store)
        self.jobs = _FakeJobStore(store)


class _FakeDocumentStore:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def get_content_hash(self, doc_id: str) -> str | None:
        row = self._store.documents.get(doc_id)
        return row["content_hash"] if row else None

    def upsert_document(self, document: Document, status: IngestStatus) -> None:
        self._store.documents[document.doc_id] = {
            "content_hash": document.content_hash,
            "status": status.value,
            "section_count": len(document.sections),
        }


class _FakeIndexWriter:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def replace_chunks(
        self, document: Document, chunks: Sequence[Chunk], embeddings: np.ndarray
    ) -> int:
        stale = [
            cid
            for cid, row in self._store.chunks.items()
            if row["doc_id"] == document.doc_id
        ]
        for cid in stale:
            del self._store.chunks[cid]
            self._store.embeddings.pop(cid, None)
        for i, chunk in enumerate(chunks):
            self._store.chunks[chunk.chunk_id] = {
                "doc_id": document.doc_id,
                "section_key": chunk.metadata.section_key,
                "text": chunk.text,
            }
            self._store.embeddings[chunk.chunk_id] = embeddings[i]
        return len(chunks)


class _FakeJobStore:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def start_run(self, manifest: RunManifest) -> None:
        self._store.runs[manifest.run_id] = {
            "split": manifest.split,
            "status": "running",
            "stats": manifest.stats.model_dump(),
        }

    def finish_run(self, run_id: str, status: str, stats: IngestStats) -> None:
        if run_id in self._store.runs:
            self._store.runs[run_id]["status"] = status
            self._store.runs[run_id]["stats"] = stats.model_dump()

    def record(self, run_id: str, doc_id: str, status: IngestStatus, error: str = "") -> None:
        row = self._store.jobs.setdefault(
            (run_id, doc_id), {"status": status.value, "attempts": 0, "error": ""}
        )
        row["status"] = status.value
        row["attempts"] += 1
        row["error"] = error


class InMemoryUnitOfWork:
    def __init__(self, store: InMemoryStore | None = None) -> None:
        self.store = store or InMemoryStore()

    @contextmanager
    def transaction(self) -> Iterator[_InMemoryScope]:
        yield _InMemoryScope(self.store)


# --- builders -------------------------------------------------------------
def make_document(doc_id: str = "0001_2020.htm", **sections: str) -> Document:
    if not sections:
        sections = {"section_1": "Business. " * 5, "section_1A": "Risk. " * 5}
    built = [
        Section(key=f[len("section_") :], field=f, ordinal=i, text=t)
        for i, (f, t) in enumerate(sections.items())
    ]
    from rag_ingest.domain.models import content_hash

    digest = content_hash(doc_id + "".join(s.text for s in built))
    return Document(
        doc_id=doc_id,
        cik="123",
        year="2020",
        source="test",
        sections=built,
        content_hash=digest,
    )


def make_record(doc_id: str = "0001_2020.htm", **sections: str) -> RawRecord:
    if not sections:
        sections = {"section_1": "Business. " * 5, "section_1A": "Risk. " * 5}
    return RawRecord(filename=doc_id, cik="123", year="2020", sections=sections)
