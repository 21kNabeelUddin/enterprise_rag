"""query history, feedback and retrieval indexes

Revision ID: 4f2c9a1b7d10
Revises: 8e9a6458c4d6
Create Date: 2026-10-19 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "4f2c9a1b7d10"
down_revision: str | None = "8e9a6458c4d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "query_turns",
        sa.Column(
            "turn_id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=True),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column(
            "groups",
            postgresql.ARRAY(sa.String(length=64)),
            nullable=False,
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("timings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("trace", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("turn_id"),
        sa.UniqueConstraint("request_id"),
    )
    op.create_index(
        "ix_query_turns_session_created",
        "query_turns",
        ["session_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_query_turns_user_created",
        "query_turns",
        ["user_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "answer_feedback",
        sa.Column(
            "feedback_id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("turn_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["turn_id"], ["query_turns.turn_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("feedback_id"),
        sa.UniqueConstraint("turn_id", "user_id", name="uq_answer_feedback_turn_user"),
    )

    # Retrieval support: metadata filters and ACL containment.
    op.create_index("ix_chunks_cik_year", "chunks", ["cik", "year"], unique=False)
    op.create_index(
        "ix_chunks_acl",
        "chunks",
        ["acl"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_embeddings_model_dim",
        "embeddings",
        ["model_id", "dim"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_embeddings_model_dim", table_name="embeddings")
    op.drop_index("ix_chunks_acl", table_name="chunks", postgresql_using="gin")
    op.drop_index("ix_chunks_cik_year", table_name="chunks")
    op.drop_table("answer_feedback")
    op.drop_index("ix_query_turns_user_created", table_name="query_turns")
    op.drop_index("ix_query_turns_session_created", table_name="query_turns")
    op.drop_table("query_turns")
