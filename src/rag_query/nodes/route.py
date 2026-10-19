"""Step 4 -- router (is this answerable?).

Classifies the query before we pay for rewriting or retrieval. Non-answerable
input (small talk, off-topic, empty) ends the run with
:attr:`QueryStatus.NOT_ANSWERABLE`; the fallback node turns that into a reply.
"""

from __future__ import annotations

from typing import Any

from rag_query.domain.models import QueryStatus
from rag_query.nodes.base import Node
from rag_query.ports.router import QueryRouter
from rag_query.state import QueryState


class RouteNode(Node):
    name = "route"

    def __init__(self, router: QueryRouter) -> None:
        self._router = router

    def run(self, state: QueryState) -> dict[str, Any]:
        decision = self._router.route(state.get("query") or "")
        updates: dict[str, Any] = {"route": decision}
        if not decision.answerable:
            updates["status"] = QueryStatus.NOT_ANSWERABLE
        return updates
