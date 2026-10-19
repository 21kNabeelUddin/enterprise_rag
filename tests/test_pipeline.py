"""End-to-end pipeline behaviour with in-memory ports (no DB, no GPU)."""

from __future__ import annotations

from rag_ingest.adapters.chunkers.recursive import RecursiveTokenChunker
from rag_ingest.adapters.chunkers.section_aware import SectionAwareChunker
from rag_ingest.adapters.cleaners import EmptySectionDropper, UnicodeNormalizer, WhitespaceCleaner
from rag_ingest.adapters.extractors.passthrough import PassthroughExtractor
from rag_ingest.adapters.guards import NoOpGuard
from rag_ingest.pipeline.orchestrator import IngestionPipeline
from rag_ingest.ports.cleaner import GuardDecision
from tests.fakes import (
    BoomExtractor,
    FakeEmbedder,
    InMemoryUnitOfWork,
    ListReader,
    WordCounter,
    make_record,
)


class BlockingGuard:
    name = "block"

    def check(self, document) -> GuardDecision:  # noqa: ANN001
        return GuardDecision(allowed=False, reason="policy")


def _build(reader, uow, extractor=None) -> IngestionPipeline:
    counter = WordCounter()
    splitter = RecursiveTokenChunker(counter, target_tokens=30, overlap_tokens=5, max_tokens=500)
    return IngestionPipeline(
        reader=reader,
        extractor=extractor or PassthroughExtractor(),
        cleaners=[UnicodeNormalizer(), WhitespaceCleaner(), EmptySectionDropper()],
        guards=[NoOpGuard()],
        chunker=SectionAwareChunker(splitter, counter),
        embedder=FakeEmbedder(),
        uow=uow,
        split="test",
        run_id="run-1",
        log_every=1000,
    )


def test_pipeline_indexes_documents_and_writes_jobs() -> None:
    records = [make_record("a.htm"), make_record("b.htm")]
    uow = InMemoryUnitOfWork()
    stats = _build(ListReader(records), uow).run()

    assert stats.documents_seen == 2
    assert stats.documents_indexed == 2
    assert stats.documents_failed == 0
    assert stats.chunks_written >= 2
    assert set(uow.store.documents) == {"a.htm", "b.htm"}
    assert uow.store.jobs[("run-1", "a.htm")]["status"] == "indexed"
    assert uow.store.runs["run-1"]["status"] == "completed"


def test_second_run_skips_unchanged_documents() -> None:
    records = [make_record("a.htm"), make_record("b.htm")]
    uow = InMemoryUnitOfWork()
    _build(ListReader(records), uow).run()
    chunks_before = len(uow.store.chunks)

    stats = _build(ListReader(records), uow).run()
    assert stats.documents_skipped == 2
    assert stats.documents_indexed == 0
    assert len(uow.store.chunks) == chunks_before


def test_failure_is_isolated_to_one_document() -> None:
    records = [make_record("good.htm"), make_record("bad.htm")]
    uow = InMemoryUnitOfWork()
    stats = _build(ListReader(records), uow, extractor=BoomExtractor("bad.htm")).run()

    assert stats.documents_indexed == 1
    assert stats.documents_failed == 1
    assert "good.htm" in uow.store.documents
    assert uow.store.jobs[("run-1", "bad.htm")]["status"] == "failed"
    assert "kaboom" in uow.store.jobs[("run-1", "bad.htm")]["error"]


def test_guard_can_block_everything() -> None:
    uow = InMemoryUnitOfWork()
    counter = WordCounter()
    pipeline = IngestionPipeline(
        reader=ListReader([make_record("a.htm")]),
        extractor=PassthroughExtractor(),
        cleaners=[],
        guards=[BlockingGuard()],
        chunker=SectionAwareChunker(
            RecursiveTokenChunker(counter, target_tokens=30, overlap_tokens=5, max_tokens=500),
            counter,
        ),
        embedder=FakeEmbedder(),
        uow=uow,
        split="test",
        run_id="run-1",
    )
    stats = pipeline.run()
    assert stats.documents_skipped == 1
    assert stats.documents_indexed == 0
    assert "a.htm" not in uow.store.documents


def test_limit_stops_early() -> None:
    records = [make_record(f"{i}.htm") for i in range(5)]
    uow = InMemoryUnitOfWork()
    stats = _build(ListReader(records), uow).run(limit=2)
    assert stats.documents_seen == 2
    assert stats.documents_indexed == 2
