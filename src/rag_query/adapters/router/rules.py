"""Rule-based query router.

Decides, with no model call, whether a query is worth retrieving for. It rejects
empty/too-short input and obvious small-talk, and accepts everything else. This
is the cheap first gate; an LLM classifier can replace it behind the same port.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_query.domain.models import RouteDecision

_SMALL_TALK: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*(hi|hii+|hello|hey|yo|hiya|howdy|sup|greetings)\b[!. ]*$",
        r"^\s*good\s+(morning|afternoon|evening)\b[!. ]*$",
        r"^\s*(thanks|thank you|thx|ty|cheers|bye|goodbye|see you|see ya)\b[!. ]*$",
        r"\bhow are you\b",
        r"^\s*(who|what)\s+(are|r)\s+(you|u)\b",
        r"^\s*what can you do\b",
        r"^\s*what's up\b",
    )
)


class RuleBasedRouter:
    def __init__(
        self,
        *,
        min_tokens: int = 3,
        small_talk_patterns: Sequence[re.Pattern[str]] | None = None,
    ) -> None:
        self._min_tokens = max(1, min_tokens)
        self._small_talk = tuple(small_talk_patterns or _SMALL_TALK)

    def route(self, query: str) -> RouteDecision:
        text = (query or "").strip()
        if not text:
            return RouteDecision(answerable=False, intent="empty", reason="empty query")

        # Small talk is checked first so a greeting gets a friendly reply rather
        # than a generic "too short" one.
        if any(pattern.search(text) for pattern in self._small_talk):
            return RouteDecision(answerable=False, intent="smalltalk", reason="greeting/small talk")

        token_count = len(text.split())
        if token_count < self._min_tokens:
            return RouteDecision(
                answerable=False,
                intent="too_short",
                reason=f"only {token_count} token(s); need at least {self._min_tokens}",
            )

        return RouteDecision(answerable=True, intent="question", reason="looks answerable")
