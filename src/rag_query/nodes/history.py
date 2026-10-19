"""Supporting node -- load conversation history before the rewriter runs.

History is scoped by session *and* authenticated user. A failure to read history
is never fatal: the request proceeds with whatever history the caller supplied.
"""

from __future__ import annotations

from typing import Any

from rag_query.logging import get_logger
from rag_query.nodes.base import Node
from rag_query.ports.history import HistoryStore
from rag_query.state import QueryState

log = get_logger(__name__)


class HistoryNode(Node):
    name = "history"

    def __init__(self, store: HistoryStore | None, *, max_turns: int = 6) -> None:
        self._store = store
        self._max_turns = max(0, max_turns)

    def run(self, state: QueryState) -> dict[str, Any]:
        request_history = list(state.get("history") or [])
        session_id = state.get("session_id")
        principal = state.get("principal")

        if self._store is None or not session_id or principal is None or self._max_turns == 0:
            return {"history": _dedupe(request_history), "history_loaded": 0}

        try:
            stored = self._store.recent(
                session_id,
                principal.user_id,
                limit=self._max_turns,
            )
        except Exception as exc:  # noqa: BLE001 - history is an enhancement
            log.warning("could not load history for session %s: %s", session_id, exc)
            return {
                "history": _dedupe(request_history),
                "history_loaded": 0,
                "warnings": [f"history_load_failed: {exc}"],
            }

        merged = _dedupe([*stored, *request_history])
        return {"history": merged, "history_loaded": len(stored)}


def _dedupe(messages: list) -> list:
    seen: set[tuple[str, str]] = set()
    out = []
    for message in messages:
        key = (message.role, message.content)
        if key in seen:
            continue
        seen.add(key)
        out.append(message)
    return out
