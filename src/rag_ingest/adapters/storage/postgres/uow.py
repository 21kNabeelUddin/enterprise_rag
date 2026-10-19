"""SQLAlchemy unit of work.

``transaction()`` opens a session, begins a transaction and yields a scope with
document/index/job stores bound to it. On clean exit the transaction commits;
on any exception it rolls back. One document = one transaction, which is the
ACID guarantee for ingestion.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker

from rag_ingest.adapters.storage.postgres.repositories import (
    PostgresDocumentStore,
    PostgresIndexWriter,
    PostgresJobStore,
)
from rag_ingest.adapters.storage.postgres.tables import VECTOR_DIM


class TransactionScope:
    def __init__(self, session: Session, *, model_id: str, revision: str | None, dim: int) -> None:
        self.documents = PostgresDocumentStore(session)
        self.index = PostgresIndexWriter(
            session, model_id=model_id, revision=revision, dim=dim
        )
        self.jobs = PostgresJobStore(session)


class SqlAlchemyUnitOfWork:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        model_id: str = "BAAI/bge-large-en-v1.5",
        revision: str | None = None,
        dim: int = VECTOR_DIM,
    ) -> None:
        self._session_factory = session_factory
        self._model_id = model_id
        self._revision = revision
        self._dim = dim

    @contextmanager
    def transaction(self) -> Iterator[TransactionScope]:
        session = self._session_factory()
        try:
            with session.begin():
                yield TransactionScope(
                    session, model_id=self._model_id, revision=self._revision, dim=self._dim
                )
        finally:
            session.close()
