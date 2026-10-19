"""rag_ingest: offline ingestion pipeline for enterprise RAG.

The package is organised as ports-and-adapters (hexagonal) architecture:

- ``rag_ingest.domain``  : pure data models, no I/O
- ``rag_ingest.ports``   : interfaces (protocols) the pipeline depends on
- ``rag_ingest.adapters``: concrete implementations of those ports
- ``rag_ingest.pipeline``: orchestration that composes ports
- ``rag_ingest.cli``     : command line entry point
"""

__version__ = "0.1.0"
