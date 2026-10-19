"""Postgres + pgvector storage adapter."""

from rag_ingest.adapters.storage.postgres.engine import (
    build_engine,
    build_session_factory,
    init_db,
    ping,
)
from rag_ingest.adapters.storage.postgres.tables import VECTOR_DIM, Base
from rag_ingest.adapters.storage.postgres.uow import SqlAlchemyUnitOfWork

__all__ = [
    "Base",
    "VECTOR_DIM",
    "SqlAlchemyUnitOfWork",
    "build_engine",
    "build_session_factory",
    "init_db",
    "ping",
]
