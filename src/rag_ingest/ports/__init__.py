"""Ports: the interfaces the pipeline depends on (dependency inversion).

The orchestrator imports only from here. Concrete engines (BGE, Postgres,
Qdrant) live under ``rag_ingest.adapters`` and are injected at wiring time.
"""

from rag_ingest.ports.chunker import Chunker, TokenCounter
from rag_ingest.ports.cleaner import Cleaner, Guard, GuardDecision
from rag_ingest.ports.embedder import Embedder
from rag_ingest.ports.extractor import Extractor
from rag_ingest.ports.source import SourceReader
from rag_ingest.ports.storage import (
    DocumentStore,
    IndexWriter,
    JobStore,
    TransactionScope,
    UnitOfWork,
)

__all__ = [
    "Chunker",
    "Cleaner",
    "DocumentStore",
    "Embedder",
    "Extractor",
    "Guard",
    "GuardDecision",
    "IndexWriter",
    "JobStore",
    "SourceReader",
    "TokenCounter",
    "TransactionScope",
    "UnitOfWork",
]
