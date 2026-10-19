"""PII detection guard for query text.

Deliberately regex-based: fast, transparent and good enough to *flag* common
personal data in an inbound question. By default it detects and records but does
**not** block, because asking a question that mentions an email address is
legitimate. Set ``guard_block_pii`` to make it a hard block, or swap in a real
detector (e.g. Presidio) behind the same port.
"""

from __future__ import annotations

import re

from rag_query.domain.models import GuardVerdict

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)"),
    "ipv4": re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    "api_key": re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})\b"),
}


class PiiQueryGuard:
    name = "pii"

    def __init__(
        self,
        *,
        block: bool = False,
        patterns: dict[str, re.Pattern[str]] | None = None,
    ) -> None:
        self._block = block
        self._patterns = patterns or _PATTERNS

    def check(self, text: str) -> GuardVerdict:
        findings = tuple(sorted({name for name, rx in self._patterns.items() if rx.search(text)}))
        if not findings:
            return GuardVerdict(allowed=True, guard=self.name, reason="no PII detected")

        scores = {name: 1.0 for name in findings}
        joined = ", ".join(findings)
        if self._block:
            return GuardVerdict(
                allowed=False,
                guard=self.name,
                reason=f"PII detected: {joined}",
                scores=scores,
                labels=findings,
            )
        return GuardVerdict(
            allowed=True,
            guard=self.name,
            reason=f"PII detected (recorded, not blocked): {joined}",
            scores=scores,
            labels=findings,
        )
