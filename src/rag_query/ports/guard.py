"""Input guardrail port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_query.domain.models import GuardVerdict


@runtime_checkable
class QueryGuard(Protocol):
    """Screens a raw query before any retrieval happens.

    Each guard is independent; the guard node runs them in order and blocks if
    any one declines. Guards must never raise for ordinary input -- they return
    a :class:`GuardVerdict`.
    """

    @property
    def name(self) -> str: ...

    def check(self, text: str) -> GuardVerdict: ...
