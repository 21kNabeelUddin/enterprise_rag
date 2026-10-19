"""Postgres-backed read models for the query pipeline."""

from __future__ import annotations

from rag_query.adapters.storage.postgres.vector_probe import StoredVectorProbe

__all__ = ["StoredVectorProbe"]
