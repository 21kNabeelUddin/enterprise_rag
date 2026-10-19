"""LLM (chat completion) port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag_query.domain.models import ChatMessage


@runtime_checkable
class LLMClient(Protocol):
    """A minimal OpenAI-style chat interface.

    The only LLM-shaped dependency in the codebase, so the local llama.cpp
    server, a vLLM server or a hosted API are all interchangeable.
    """

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...
