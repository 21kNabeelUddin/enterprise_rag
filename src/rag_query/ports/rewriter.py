"""Query rewriting port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag_query.domain.models import ChatMessage


@runtime_checkable
class QueryRewriter(Protocol):
    """Turns a (possibly context-dependent) question into a standalone query.

    ``history`` is the recent conversation, so a follow-up such as "and the
    price?" can be resolved against what came before.
    """

    def rewrite(self, query: str, history: Sequence[ChatMessage]) -> str: ...
