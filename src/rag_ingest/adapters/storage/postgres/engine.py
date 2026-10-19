"""Engine/session factories and schema bootstrap."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from rag_ingest.adapters.storage.postgres.tables import Base


def build_engine(dsn: str, *, echo: bool = False, connect_timeout: int | None = None) -> Engine:
    connect_args = {"connect_timeout": connect_timeout} if connect_timeout is not None else {}
    return create_engine(
        dsn,
        echo=echo,
        pool_pre_ping=True,
        future=True,
        connect_args=connect_args,
    )


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create the vector extension and all tables/indexes (idempotent)."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def ping(engine: Engine) -> str:
    with engine.connect() as conn:
        return str(conn.execute(text("select version()")).scalar_one())
