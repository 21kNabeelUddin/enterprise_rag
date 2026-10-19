"""Query-owned tables (step 14): conversation turns and feedback.

These live in the ``rag_query`` package but share the ingestion ``Base`` so
Alembic sees one metadata graph and one migration chain.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from rag_ingest.adapters.storage.postgres.tables import Base


class QueryTurnRow(Base):
    __tablename__ = "query_turns"

    turn_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, server_default=func.gen_random_uuid()
    )
    request_id: Mapped[str] = mapped_column(String(64), unique=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    user_id: Mapped[str] = mapped_column(String(128))
    groups: Mapped[list[str]] = mapped_column(ARRAY(String(64)), default=list)
    question: Mapped[str] = mapped_column(Text, default="")
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="error")
    citations: Mapped[list] = mapped_column(JSONB, default=list)
    timings: Mapped[dict] = mapped_column(JSONB, default=dict)
    trace: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_query_turns_session_created", "session_id", "created_at"),
        Index("ix_query_turns_user_created", "user_id", "created_at"),
    )


class AnswerFeedbackRow(Base):
    __tablename__ = "answer_feedback"

    feedback_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, server_default=func.gen_random_uuid()
    )
    turn_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("query_turns.turn_id", ondelete="CASCADE"),
    )
    user_id: Mapped[str] = mapped_column(String(128))
    rating: Mapped[int] = mapped_column(SmallInteger)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("turn_id", "user_id", name="uq_answer_feedback_turn_user"),
    )
