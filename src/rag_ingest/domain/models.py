"""Domain models: pure data, no I/O, no framework imports.

These types are the vocabulary shared by every port. Keeping them free of
storage or ML details is what lets the same pipeline run against Postgres or
Qdrant, and lets stages be unit-tested without a database or a GPU.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def content_hash(text: str) -> str:
    """Stable SHA-256 of a string. Used for document/chunk idempotency."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IngestStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    SKIPPED = "skipped"
    FAILED = "failed"


class RawRecord(BaseModel):
    """One line of an EDGAR-CORPUS jsonl file: a single 10-K filing.

    ``sections`` maps the original field name (``section_1``, ``section_1A``,
    ...) to its raw text, preserving the source schema.
    """

    filename: str
    cik: str
    year: str
    sections: dict[str, str]


class Section(BaseModel):
    """A single SEC Item (e.g. 1A Risk Factors) after cleaning."""

    model_config = ConfigDict(frozen=True)

    key: str  # normalised label, e.g. "1A"
    field: str  # original field name, e.g. "section_1A"
    ordinal: int  # order within the filing
    text: str

    @property
    def char_len(self) -> int:
        return len(self.text)


class Document(BaseModel):
    """A cleaned 10-K: identity, provenance and ordered sections."""

    doc_id: str  # canonical id == filename
    cik: str
    year: str
    source: str  # adapter name, e.g. "edgar_jsonl"
    sections: list[Section]
    content_hash: str

    @property
    def is_empty(self) -> bool:
        return not any(s.text.strip() for s in self.sections)


class ChunkMetadata(BaseModel):
    """Everything stored alongside a chunk for filtering and citations."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    cik: str
    year: str
    section_key: str
    chunk_index: int  # 0-based position within the document
    token_count: int
    char_start: int  # offset into the section text
    char_end: int
    acl: tuple[str, ...] = ("public",)  # forward-compatible; EDGAR is public


class Chunk(BaseModel):
    """A retrievable unit of text plus its metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str  # deterministic: "doc_id:section_key:chunk_index"
    text: str
    metadata: ChunkMetadata

    @property
    def content_hash(self) -> str:
        return content_hash(self.text)


class ModelFingerprint(BaseModel):
    """The full embedding contract.

    Records more than the model name: the same weights with a different
    normalisation, max length or query prefix produce incomparable vectors.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str | None = None
    dim: int
    normalize: bool = True
    max_seq_length: int = 512
    pooling: str = "cls"
    query_prefix: str = ""


class IngestStats(BaseModel):
    """Counters for one ingestion run."""

    documents_seen: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_written: int = 0


class RunManifest(BaseModel):
    """Persisted record of what a run did, for observability and debugging."""

    run_id: str
    source: str
    split: str
    started_at: str
    finished_at: str | None = None
    stats: IngestStats = Field(default_factory=IngestStats)
