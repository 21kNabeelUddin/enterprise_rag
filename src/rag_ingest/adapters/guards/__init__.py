"""Guard adapters (safe defaults).

These are intentional no-ops/audit-only. EDGAR data is public, so PII redaction
would only damage legitimate named entities and malware scanning of text is
meaningless. Real policies (Presidio, scanners) can implement the same port
later without changing the pipeline.
"""

from rag_ingest.domain.models import Document
from rag_ingest.ports.cleaner import GuardDecision


class NoOpGuard:
    def __init__(self, *, name: str = "noop_guard", reason: str = "noop policy") -> None:
        self._name = name
        self._reason = reason

    @property
    def name(self) -> str:
        return self._name

    def check(self, document: Document) -> GuardDecision:
        return GuardDecision(allowed=True, reason=self._reason, redacted_count=0)
