"""Reranking port (step 9)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag_query.domain.models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Score (query, chunk) pairs and keep the best ``top_k``."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]: ...
