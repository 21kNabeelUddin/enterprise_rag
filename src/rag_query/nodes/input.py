"""Step 1 -- user input.

Normalises the raw question (collapse whitespace, trim) and rejects the two
cases no later stage can do anything useful with: empty input and absurdly
long input.
"""

from __future__ import annotations

from typing import Any

from rag_query.domain.models import QueryStatus
from rag_query.nodes.base import Node
from rag_query.state import QueryState


class InputNode(Node):
    name = "input"

    def __init__(self, *, max_chars: int = 4000) -> None:
        self._max_chars = max(1, max_chars)

    def run(self, state: QueryState) -> dict[str, Any]:
        raw = state.get("raw_query") or ""
        query = " ".join(raw.split())

        if not query:
            return {
                "query": "",
                "status": QueryStatus.REJECTED,
                "message": "The query was empty.",
            }

        if len(query) > self._max_chars:
            return {
                "query": query[: self._max_chars],
                "status": QueryStatus.REJECTED,
                "message": (
                    f"The query is too long ({len(query)} characters; the limit is "
                    f"{self._max_chars})."
                ),
            }

        return {"query": query}
