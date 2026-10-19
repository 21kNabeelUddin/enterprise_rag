"""Pure domain types for the query pipeline (no I/O, no framework imports)."""

from __future__ import annotations

from rag_query.domain.errors import (
    AuthenticationError,
    FingerprintMismatchError,
    QueryPipelineError,
)
from rag_query.domain.models import (
    ANONYMOUS_USER_ID,
    AskRequest,
    AskResult,
    ChatMessage,
    GuardVerdict,
    Principal,
    QueryStatus,
    RouteDecision,
    StoredVectorInfo,
)

__all__ = [
    "ANONYMOUS_USER_ID",
    "AskRequest",
    "AskResult",
    "AuthenticationError",
    "ChatMessage",
    "FingerprintMismatchError",
    "GuardVerdict",
    "Principal",
    "QueryPipelineError",
    "QueryStatus",
    "RouteDecision",
    "StoredVectorInfo",
]
