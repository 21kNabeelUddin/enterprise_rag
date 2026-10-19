"""Answer generation port (step 11)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_query.domain.models import ContextBundle, Generation


@runtime_checkable
class Generator(Protocol):
    """Turn a built context bundle into a grounded answer.

    ``strict_feedback`` is non-empty on the regeneration attempt and tells the
    model exactly which guardrail the previous answer failed.
    """

    @property
    def model(self) -> str: ...

    def generate(
        self,
        bundle: ContextBundle,
        *,
        strict_feedback: str = "",
    ) -> Generation: ...
