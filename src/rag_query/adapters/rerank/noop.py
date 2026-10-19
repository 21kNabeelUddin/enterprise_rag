"""No-op reranker: keep retrieval order.

Used in unit tests, when reranking is disabled, and as the graceful fallback if
the cross-encoder cannot be loaded.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_query.domain.models import RetrievedChunk


class NoOpReranker:
    def __init__(self, *, reason: str = "rerank disabled") -> None:
        self.reason = reason

    @property
    def enabled(self) -> bool:
        return False

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        kept = list(candidates)[: (top_k or len(candidates))]
        return [
            chunk.model_copy(update={"rank": index + 1})
            for index, chunk in enumerate(kept)
        ]
