"""Composition root: build concrete objects from settings.

This is the only place that knows about concrete adapters. Everything else
receives its collaborators as ports, which is what keeps the pipeline testable
and swappable (Postgres today, Qdrant behind the same ports tomorrow).
"""

from __future__ import annotations

from rag_ingest.adapters.chunkers.recursive import RecursiveTokenChunker
from rag_ingest.adapters.chunkers.section_aware import SectionAwareChunker
from rag_ingest.adapters.chunkers.token_counter import HFTokenCounter
from rag_ingest.adapters.cleaners import (
    ContinuedHeaderCollapser,
    EmptySectionDropper,
    UnicodeNormalizer,
    WhitespaceCleaner,
)
from rag_ingest.adapters.embedders.bge import BgeEmbedder
from rag_ingest.adapters.extractors.passthrough import PassthroughExtractor
from rag_ingest.adapters.guards import NoOpGuard
from rag_ingest.adapters.sources.edgar_jsonl import EdgarJsonlReader
from rag_ingest.adapters.storage.postgres.engine import (
    build_engine,
    build_session_factory,
)
from rag_ingest.adapters.storage.postgres.uow import SqlAlchemyUnitOfWork
from rag_ingest.config import Settings
from rag_ingest.pipeline.orchestrator import IngestionPipeline


def build_settings(**overrides: object) -> Settings:
    return Settings(**overrides)


def build_unit_of_work(settings: Settings) -> SqlAlchemyUnitOfWork:
    engine = build_engine(settings.postgres_dsn)
    factory = build_session_factory(engine)
    return SqlAlchemyUnitOfWork(factory)


def build_embedder(settings: Settings) -> BgeEmbedder:
    return BgeEmbedder(
        settings.model_dir,
        device=settings.embed_device,
        batch_size=settings.embed_batch_size,
        max_seq_length=settings.model_max_seq_length,
        use_fp16=settings.embed_use_fp16,
    )


def build_pipeline(
    settings: Settings,
    split: str,
    run_id: str | None = None,
    embedder: BgeEmbedder | None = None,
) -> IngestionPipeline:
    reader = EdgarJsonlReader(settings.dataset_file(split), name=f"edgar_jsonl:{split}")
    counter = HFTokenCounter(settings.model_dir)
    splitter = RecursiveTokenChunker(
        counter,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        max_tokens=settings.chunk_max_tokens,
    )
    return IngestionPipeline(
        reader=reader,
        extractor=PassthroughExtractor(),
        cleaners=[
            UnicodeNormalizer(),
            WhitespaceCleaner(),
            ContinuedHeaderCollapser(),
            EmptySectionDropper(),
        ],
        guards=[NoOpGuard()],
        chunker=SectionAwareChunker(splitter, counter),
        embedder=embedder or build_embedder(settings),
        uow=build_unit_of_work(settings),
        split=split,
        run_id=run_id,
    )
