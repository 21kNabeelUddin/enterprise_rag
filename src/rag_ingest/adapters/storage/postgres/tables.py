"""SQLAlchemy table definitions (Postgres + pgvector).

Design notes
------------
* ``documents``  : one row per filing; ``content_hash`` drives idempotency.
* ``chunks``     : retrievable text + metadata. A generated ``tsvector`` column
  gives keyword search for free (GIN indexed), and ``acl`` is present but
  defaults to ``{public}`` so the schema is forward-compatible with access
  control without pretending EDGAR has any.
* ``embeddings`` : vectors keyed by chunk, with the model fingerprint columns.
* ``ingest_runs``/``ingest_jobs``: durable progress so runs resume.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rag_ingest.domain.models import IngestStatus

# Must match the embedding model dimension (BGE-large-en-v1.5 = 1024).
# A mismatch is caught at write time in the index writer.
VECTOR_DIM = 1024


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    cik: Mapped[str] = mapped_column(String(32), default="")
    year: Mapped[str] = mapped_column(String(8), default="")
    source: Mapped[str] = mapped_column(String(64), default="")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default=IngestStatus.PENDING.value)
    section_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(320), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    cik: Mapped[str] = mapped_column(String(32), default="")
    year: Mapped[str] = mapped_column(String(8), default="")
    section_key: Mapped[str] = mapped_column(String(16), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    acl: Mapped[list[str]] = mapped_column(ARRAY(String(64)), default=lambda: ["public"])
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="uq_chunks_doc_index"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )


class EmbeddingRow(Base):
    __tablename__ = "embeddings"

    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"), primary_key=True
    )
    model_id: Mapped[str] = mapped_column(String(128))
    revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dim: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIM))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index(
            "ix_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class IngestRunRow(Base):
    __tablename__ = "ingest_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), default="running")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stats: Mapped[dict] = mapped_column(JSONB, default=dict)


class IngestJobRow(Base):
    __tablename__ = "ingest_jobs"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("ingest_runs.run_id", ondelete="CASCADE"), primary_key=True
    )
    doc_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default=IngestStatus.PENDING.value)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
