"""Shared node machinery.

A node returns a *partial* state update. ``__call__`` decorates that with the
node name and its wall-clock time, so the final result carries a per-step
breakdown for free.
"""

from __future__ import annotations

import time
from typing import Any

from rag_query.state import QueryState


class Node:
    """Base class for a LangGraph node."""

    name: str = "node"

    def run(self, state: QueryState) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, state: QueryState) -> dict[str, Any]:
        started = time.perf_counter()
        updates = dict(self.run(state) or {})
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        timings = dict(state.get("timings_ms") or {})
        timings[self.name] = round(elapsed_ms, 3)
        updates["timings_ms"] = timings
        updates["trace"] = [self.name]
        return updates
