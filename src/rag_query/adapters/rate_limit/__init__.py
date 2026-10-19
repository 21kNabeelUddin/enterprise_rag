"""Rate-limiting adapters."""

from __future__ import annotations

from rag_query.adapters.rate_limit.memory import InMemoryRateLimiter

__all__ = ["InMemoryRateLimiter"]
