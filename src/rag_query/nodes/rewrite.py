"""Step 5 -- query rewriter.

Rewrites the question into a standalone, search-optimised query using the chat
history so follow-ups resolve. The rewriter is an *enhancement*: if the LLM call
fails we keep the original query and record a warning rather than failing the
request.
"""

from __future__ import annotations

from typing import Any

from rag_query.logging import get_logger
from rag_query.nodes.base import Node
from rag_query.ports.rewriter import QueryRewriter
from rag_query.state import QueryState

log = get_logger(__name__)


class RewriteNode(Node):
    name = "rewrite"

    def __init__(self, rewriter: QueryRewriter, *, enabled: bool = True) -> None:
        self._rewriter = rewriter
        self._enabled = enabled

    def run(self, state: QueryState) -> dict[str, Any]:
        query = state.get("query") or ""

        if not self._enabled:
            return {"rewritten_query": query}

        history = state.get("history") or []
        try:
            rewritten = self._rewriter.rewrite(query, history)
        except Exception as exc:  # noqa: BLE001 - the rewriter must never be fatal
            log.exception("query rewrite failed; using the original query")
            return {"rewritten_query": query, "warnings": [f"rewrite_failed: {exc}"]}

        return {"rewritten_query": rewritten or query}
