"""Cleaner and guard ports.

A ``Cleaner`` is a pure text transform applied in order. A ``Guard`` inspects a
document and reports a policy decision; the default adapters are no-ops so the
public-data path is never silently altered, but the seams exist for real
policies (PII redaction, malware scanning) later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_ingest.domain.models import Document


@runtime_checkable
class Cleaner(Protocol):
    @property
    def name(self) -> str: ...

    def clean(self, document: Document) -> Document: ...


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    reason: str = ""
    redacted_count: int = 0


@runtime_checkable
class Guard(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, document: Document) -> GuardDecision: ...
