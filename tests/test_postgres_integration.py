"""Integration tests against a real Postgres + pgvector.

Skipped unless ``RAG_TEST_POSTGRES_DSN`` is set, e.g.::

    docker compose -f compose.postgres.yml up -d
    $env:RAG_TEST_POSTGRES_DSN = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    pytest tests/test_postgres_integration.py
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import func, select, text

from rag_ingest.adapters.chunkers.recursive import RecursiveTokenChunker
from rag_ingest.adapters.chunkers.section_aware import SectionAwareChunker
from rag_ingest.adapters.extractors.passthrough import PassthroughExtractor
from rag_ingest.adapters.guards import NoOpGuard
from rag_ingest.adapters.storage.postgres.engine import (
    build_engine,
    build_session_factory,
)
from rag_ingest.adapters.storage.postgres.tables import (
    ChunkRow,
    DocumentRow,
    EmbeddingRow,
    IngestJobRow,
    IngestRunRow,
)
from rag_ingest.adapters.storage.postgres.uow import SqlAlchemyUnitOfWork
from rag_ingest.config import Settings
from rag_ingest.domain.models import IngestStatus
from rag_ingest.migrate import upgrade_to_head
from rag_ingest.pipeline.orchestrator import IngestionPipeline
from tests.fakes import FakeEmbedder, ListReader, WordCounter, make_record

DSN = os.environ.get("RAG_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="RAG_TEST_POSTGRES_DSN not set")

# Must match the VECTOR(n) column, which is fixed for BGE-large-en-v1.5.
EMBED_DIM = 1024


@pytest.fixture(scope="module")
def factory():
    # apply the real migrations so the test exercises the schema path users use
    upgrade_to_head(Settings(postgres_dsn=DSN))
    engine = build_engine(DSN)
    return build_session_factory(engine)


@pytest.fixture(autouse=True)
def _clean(request):
    # only clean for tests that actually use the DB
    if "factory" not in request.fixturenames:
        return
    factory = request.getfixturevalue("factory")
    with factory() as session:
        session.execute(
            text(
                "TRUNCATE ingest_jobs, ingest_runs, embeddings, chunks, documents "
                "RESTART IDENTITY CASCADE"
            )
        )
        session.commit()


def _uow(factory, dim: int = EMBED_DIM) -> SqlAlchemyUnitOfWork:
    return SqlAlchemyUnitOfWork(factory, dim=dim)


def _pipeline(reader, uow, run_id="it-run", embedder=None) -> IngestionPipeline:
    counter = WordCounter()
    splitter = RecursiveTokenChunker(counter, target_tokens=30, overlap_tokens=5, max_tokens=500)
    return IngestionPipeline(
        reader=reader,
        extractor=PassthroughExtractor(),
        cleaners=[],
        guards=[NoOpGuard()],
        chunker=SectionAwareChunker(splitter, counter),
        embedder=embedder or FakeEmbedder(dim=EMBED_DIM),
        uow=uow,
        split="test",
        run_id=run_id,
        log_every=1000,
    )


def test_full_run_persists_and_is_idempotent(factory) -> None:
    records = [make_record("a.htm"), make_record("b.htm")]
    stats = _pipeline(ListReader(records), _uow(factory)).run()
    assert stats.documents_indexed == 2

    with factory() as session:
        assert session.scalar(select(func.count()).select_from(DocumentRow)) == 2
        embeddings = session.scalar(select(func.count()).select_from(EmbeddingRow))
        assert embeddings == stats.chunks_written
        assert session.scalar(select(func.count()).select_from(IngestRunRow)) >= 1
        statuses = session.scalars(select(IngestJobRow.status)).all()
        assert set(statuses) == {"indexed"}

    # rerun: everything unchanged -> skipped, no new chunks
    before = stats.chunks_written
    second = _pipeline(ListReader(records), _uow(factory), run_id="it-run-2").run()
    assert second.documents_skipped == 2
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(ChunkRow)) == before


def test_generated_tsvector_supports_keyword_search(factory) -> None:
    _pipeline(ListReader([make_record("fts.htm")]), _uow(factory), run_id="it-fts").run()
    with factory() as session:
        hit = session.scalar(
            text(
                "SELECT count(*) FROM chunks "
                "WHERE tsv @@ plainto_tsquery('english', 'business')"
            )
        )
    assert hit and hit > 0


def test_dimension_mismatch_rolls_back_the_whole_document(factory) -> None:
    # embedder produces dim=8 vectors but the writer/column expect 1024
    reader = ListReader([make_record("bad.htm")])
    stats = _pipeline(
        reader,
        _uow(factory, dim=1024),
        run_id="it-bad",
        embedder=FakeEmbedder(dim=8),
    ).run()

    assert stats.documents_failed == 1
    with factory() as session:
        doc = session.get(DocumentRow, "bad.htm")
        assert doc is not None and doc.status == "failed"
        assert (
            session.scalar(
                select(func.count()).select_from(ChunkRow).where(ChunkRow.doc_id == "bad.htm")
            )
            == 0
        )


def test_replace_chunks_cascades_embeddings(factory) -> None:
    doc = PassthroughExtractor().extract(
        make_record("c.htm", section_1="alpha beta gamma delta. " * 4)
    )
    counter = WordCounter()
    chunker = SectionAwareChunker(
        RecursiveTokenChunker(counter, target_tokens=5, overlap_tokens=1, max_tokens=500), counter
    )
    embedder = FakeEmbedder(dim=EMBED_DIM)
    uow = _uow(factory)

    with uow.transaction() as tx:
        tx.documents.upsert_document(doc, IngestStatus.INDEXED)
        chunks = chunker.chunk(doc)
        tx.index.replace_chunks(doc, chunks, embedder.embed([c.text for c in chunks]))
    first_count = len(chunks)
    assert first_count > 1

    with uow.transaction() as tx:
        tx.index.replace_chunks(doc, chunks[:1], embedder.embed([chunks[0].text]))
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(ChunkRow)) == 1
        assert session.scalar(select(func.count()).select_from(EmbeddingRow)) == 1
