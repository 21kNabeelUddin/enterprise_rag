"""Terminal node -- turn a non-answerable outcome into a user-facing message.

Steps 1-7 stop before generation, so every "no go" path (rejected, blocked,
rate-limited, not answerable) converges here to produce the reply the caller
will actually show. Exhaustive generation arrives with step 12.
"""

from __future__ import annotations

from typing import Any

from rag_query.domain.models import QueryStatus
from rag_query.nodes.base import Node
from rag_query.state import QueryState

_BY_STATUS: dict[QueryStatus, str] = {
    QueryStatus.REJECTED: "I couldn't process that input. Please rephrase your question.",
    QueryStatus.UNAUTHORIZED: "Authentication failed. Please supply a valid token.",
    QueryStatus.RATE_LIMITED: (
        "You're sending requests too quickly. Please wait a moment and retry."
    ),
    QueryStatus.BLOCKED: "I can't help with that request.",
    QueryStatus.NOT_ANSWERABLE: "I can only answer questions about the SEC filings in the corpus.",
    QueryStatus.ERROR: "Something went wrong while processing your request.",
}

_BY_INTENT: dict[str, str] = {
    "smalltalk": "Hi! I'm a filing assistant -- ask me a question about the SEC 10-K filings.",
    "empty": "Please type a question about the SEC filings in the corpus.",
    "too_short": "Could you add a little more detail? Ask a full question about the filings.",
}


class FallbackNode(Node):
    name = "fallback"

    def run(self, state: QueryState) -> dict[str, Any]:
        # A node that already produced a specific message (e.g. input validation)
        # wins; otherwise derive one from the status and router intent.
        message = state.get("message")
        if not message:
            status = state.get("status") or QueryStatus.ERROR
            route = state.get("route")
            if status is QueryStatus.NOT_ANSWERABLE and route is not None:
                message = _BY_INTENT.get(route.intent, _BY_STATUS[QueryStatus.NOT_ANSWERABLE])
            else:
                message = _BY_STATUS.get(status, _BY_STATUS[QueryStatus.ERROR])
        return {"message": message}
