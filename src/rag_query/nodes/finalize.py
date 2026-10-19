"""Step 14 -- final response: persist the turn and promote the terminal status."""

from __future__ import annotations

from typing import Any

from rag_query.domain.models import QueryStatus, TurnRecord
from rag_query.logging import get_logger
from rag_query.nodes.base import Node
from rag_query.ports.history import HistoryStore
from rag_query.state import QueryState

log = get_logger(__name__)


class FinalizeNode(Node):
    name = "finalize"

    def __init__(
        self,
        store: HistoryStore | None = None,
        *,
        persist_failures: bool = True,
    ) -> None:
        self._store = store
        self._persist_failures = persist_failures

    def run(self, state: QueryState) -> dict[str, Any]:
        status = state.get("status") or QueryStatus.ERROR
        answer = state.get("answer") or state.get("message")
        if status is QueryStatus.ENCODED:
            status = QueryStatus.ANSWERED if answer else QueryStatus.NO_CONTEXT

        updates: dict[str, Any] = {"status": status}
        if answer and not state.get("answer"):
            updates["answer"] = answer

        if self._store is None:
            return updates
        if not self._persist_failures and not status.ok:
            return updates

        principal = state.get("principal")
        turn = TurnRecord(
            request_id=state.get("request_id", ""),
            session_id=state.get("session_id"),
            user_id=principal.user_id if principal is not None else "anonymous",
            groups=principal.groups if principal is not None else (),
            question=state.get("raw_query") or state.get("query") or "",
            rewritten_query=state.get("rewritten_query"),
            answer=answer,
            status=status,
            citations=tuple(state.get("citations") or ()),
            timings_ms=dict(state.get("timings_ms") or {}),
            trace=tuple(state.get("trace") or ()),
        )

        try:
            updates["turn_id"] = self._store.save_turn(turn)
        except Exception as exc:  # noqa: BLE001 - persistence must not lose the answer
            log.exception("could not persist turn %s", turn.request_id)
            updates["warnings"] = [f"history_persist_failed: {exc}"]
        return updates
