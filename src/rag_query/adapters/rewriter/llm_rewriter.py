"""LLM query rewriter.

Turns the user's question into a single, better search query. The system prompt
is fixed by the project spec (see ``REWRITER_SYSTEM_PROMPT``); recent history is
prepended so follow-ups resolve against the conversation.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_query.domain.models import ChatMessage
from rag_query.logging import get_logger
from rag_query.ports.llm import LLMClient

log = get_logger(__name__)

# Verbatim from the project spec -- do not edit.
REWRITER_SYSTEM_PROMPT = (
    "You are an expert in query expansion and natural language processing.\n"
    "Your task is to generate an optimized search query based on the user's input\n"
    "\n"
    " Follow these guidelines:\n"
    "\n"
    "1. Analyze the input query for key concepts and intent.\n"
    "2. Identify any ambiguous terms or phrases that could be clarified.\n"
    "3. Consider common synonyms, related terms, and alternative phrasings to improve the search.\n"
    "4. If applicable, expand acronyms or abbreviations.\n"
    "5. Incorporate any relevant context or domain-specific knowledge.\n"
    "6. Ensure the expanded query maintains the original intent of the user's question.\n"
    "7. Prioritize clarity and specificity in the rewritten query.\n"
    "8. If the original query is already optimal, you may return it unchanged.\n"
    "Your goal is to produce a single, refined query that will return the best search results.\n"
    "The rewritten query should be a natural language question or statement, not a list of keywords"
)


class LlmQueryRewriter:
    def __init__(
        self,
        llm: LLMClient,
        *,
        prompt: str = REWRITER_SYSTEM_PROMPT,
        max_history_turns: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 96,
        max_chars: int = 400,
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._max_history_turns = max(0, max_history_turns)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_chars = max(40, max_chars)

    def rewrite(self, query: str, history: Sequence[ChatMessage]) -> str:
        messages: list[ChatMessage] = [ChatMessage(role="system", content=self._prompt)]
        if self._max_history_turns:
            messages.extend(list(history)[-self._max_history_turns :])
        messages.append(ChatMessage(role="user", content=query))

        raw = self._llm.chat(messages, temperature=self._temperature, max_tokens=self._max_tokens)
        return self._clean(raw, fallback=query, max_chars=self._max_chars)

    @staticmethod
    def _clean(text: str, *, fallback: str, max_chars: int) -> str:
        # The prompt asks for a single line; collapse anything the model adds.
        cleaned = " ".join((text or "").split())
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'`":
            cleaned = cleaned[1:-1].strip()
        if len(cleaned) > max_chars:
            # A 9B model occasionally answers the question instead of rewriting
            # it. That output is a terrible search query and can pollute the
            # filters, so keep the user's original words.
            log.warning(
                "rewriter output too long (%d > %d chars); using original query",
                len(cleaned),
                max_chars,
            )
            return fallback
        return cleaned or fallback
