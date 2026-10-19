"""Programmatic Alembic runner.

Keeps a single source of truth for schema evolution: the CLI applies migrations
through here instead of calling ``create_all``, so the ``alembic_version`` table
always reflects reality.
"""

from __future__ import annotations

from alembic import command
from alembic.config import Config

from rag_ingest.config import Settings


def build_alembic_config(settings: Settings) -> Config:
    config = Config(str(settings.project_root / "alembic.ini"))
    config.set_main_option("script_location", str(settings.project_root / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.postgres_dsn)
    return config


def upgrade_to_head(settings: Settings) -> None:
    command.upgrade(build_alembic_config(settings), "head")
