"""Retrieval port (step 8)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from rag_query.domain.models import (
    Principal,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)


@runtime_checkable
class Retriever(Protocol):
    """Hybrid (dense + lexical) retrieval over the chunk index.

    Implementations apply the ACL groups from ``principal`` before returning
    anything. An empty ``groups`` tuple must return no rows.
    """

    def retrieve(
        self,
        query: str,
        vector: np.ndarray,
        *,
        principal: Principal,
        filters: RetrievalFilters | None = None,
        limit: int | None = None,
    ) -> RetrievalResult: ...

    def expand_neighbors(
        self,
        chunks: Sequence[RetrievedChunk],
        *,
        window: int = 1,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]:
        """Return the chunks plus adjacent chunks from the same section."""
        ...
