"""Step 3 -- input guardrail.

Runs the per-user rate limit first (cheap), then every configured query guard
(prompt-injection model, PII detector, ...). Any guard that declines ends the
run with :attr:`QueryStatus.BLOCKED` before a single retrieval happens.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag_query.domain.models import GuardVerdict, QueryStatus
from rag_query.nodes.base import Node
from rag_query.ports.guard import QueryGuard
from rag_query.ports.rate_limit import RateLimiter
from rag_query.state import QueryState


class InputGuardNode(Node):
    name = "input_guard"

    def __init__(
        self,
        guards: Sequence[QueryGuard],
        *,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._guards = list(guards)
        self._rate_limiter = rate_limiter

    def run(self, state: QueryState) -> dict[str, Any]:
        principal = state.get("principal")
        if (
            self._rate_limiter is not None
            and principal is not None
            and not self._rate_limiter.allow(principal.user_id)
        ):
            verdict = GuardVerdict(
                allowed=False,
                guard="rate_limit",
                reason="per-user request quota exceeded",
            )
            return {
                "guard": verdict,
                "status": QueryStatus.RATE_LIMITED,
                "message": verdict.reason,
            }

        query = state.get("query") or ""
        verdict = self._combine([guard.check(query) for guard in self._guards])

        updates: dict[str, Any] = {"guard": verdict}
        if not verdict.allowed:
            updates["status"] = QueryStatus.BLOCKED
            updates["message"] = f"Blocked by {verdict.guard}: {verdict.reason}"
        return updates

    @staticmethod
    def _combine(verdicts: Sequence[GuardVerdict]) -> GuardVerdict:
        if not verdicts:
            return GuardVerdict(allowed=True, guard="none", reason="no guards configured")

        failing = next((v for v in verdicts if not v.allowed), None)
        scores: dict[str, float] = {}
        labels: list[str] = []
        notes: list[str] = []
        for verdict in verdicts:
            scores.update(verdict.scores)
            labels.extend(label for label in verdict.labels if label not in labels)
            if verdict.reason:
                notes.append(f"{verdict.guard}: {verdict.reason}")

        return GuardVerdict(
            allowed=failing is None,
            guard=failing.guard if failing else "all",
            reason=failing.reason if failing else "; ".join(notes),
            scores=scores,
            labels=tuple(labels),
        )
