"""Ingestion pipeline: composes ports into a resumable, idempotent run."""

from rag_ingest.pipeline.orchestrator import IngestionPipeline

__all__ = ["IngestionPipeline"]
