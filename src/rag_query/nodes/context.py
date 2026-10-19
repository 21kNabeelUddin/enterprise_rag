"""Step 10 -- build the deterministic context prompt."""

from __future__ import annotations

from typing import Any

from rag_query.domain.context import ContextBuilder
from rag_query.domain.models import QueryStatus
from rag_query.nodes.base import Node
from rag_query.state import QueryState


class ContextNode(Node):
    name = "context"

    def __init__(self, builder: ContextBuilder, *, include_history: bool = False) -> None:
        self._builder = builder
        self._include_history = include_history

    def run(self, state: QueryState) -> dict[str, Any]:
        candidates = list(state.get("candidates") or [])
        question = state.get("rewritten_query") or state.get("query") or ""
        history = (state.get("history") or []) if self._include_history else []

        bundle = self._builder.build(question, candidates, history)
        if not bundle.sources:
            return {
                "context": bundle,
                "status": QueryStatus.NO_CONTEXT,
                "message": "The retrieved passages did not fit the context budget.",
            }
        return {"context": bundle}
