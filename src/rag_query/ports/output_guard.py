"""Output guardrail port (step 12)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag_query.domain.models import ContextSource, GuardVerdict


@runtime_checkable
class OutputGuard(Protocol):
    """Check a generated answer before it is shown to the caller.

    Implementations must not raise for ordinary answers; they return a
    :class:`GuardVerdict` and let the node decide whether to regenerate.
    """

    @property
    def name(self) -> str: ...

    def check(
        self,
        answer: str,
        sources: Sequence[ContextSource],
        *,
        query: str,
    ) -> GuardVerdict: ...
