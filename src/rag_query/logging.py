"""Logging helpers for the query package.

A thin re-export of the shared setup so both pipelines log identically and the
query package has one stable import path for it.
"""

from __future__ import annotations

from rag_ingest.logging_setup import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
