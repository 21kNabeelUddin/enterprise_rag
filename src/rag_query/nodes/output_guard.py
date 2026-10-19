"""Step 12 -- output guardrail.

Runs citation, grounding, PII and leakage checks. A failure triggers one
regeneration with the concrete guard reason; a second failure produces a safe
answer that still returns the sources.
"""

from __future__ import annotations

from typing import Any

from rag_query.domain.models import QueryStatus
from rag_query.logging import get_logger
from rag_query.nodes.base import Node
from rag_query.ports.output_guard import OutputGuard
from rag_query.state import QueryState

log = get_logger(__name__)


class OutputGuardNode(Node):
    name = "output_guard"

    def __init__(self, guard: OutputGuard, *, max_retries: int = 1) -> None:
        self._guard = guard
        self._max_retries = max(0, max_retries)

    def run(self, state: QueryState) -> dict[str, Any]:
        answer = state.get("answer") or ""
        bundle = state.get("context")
        sources = bundle.sources if bundle is not None else ()
        query = state.get("rewritten_query") or state.get("query") or ""
        attempts = int(state.get("generation_attempts") or 0)

        try:
            verdict = self._guard.check(answer, sources, query=query)
        except Exception as exc:  # noqa: BLE001
            log.exception("output guard failed")
            return {"status": QueryStatus.ERROR, "message": f"Output guard failed: {exc}"}

        if verdict.allowed:
            return {"output_guard": verdict, "output_feedback": ""}

        if attempts <= self._max_retries:
            return {
                "output_guard": verdict,
                "output_feedback": verdict.reason,
                "warnings": [f"output_guard_retry: {verdict.reason}"],
            }

        return {
            "output_guard": verdict,
            "status": QueryStatus.OUTPUT_BLOCKED,
            "message": (
                "I found relevant filings but couldn't produce a fully verified answer. "
                "Please review the cited sources."
            ),
            "output_feedback": "",
        }
