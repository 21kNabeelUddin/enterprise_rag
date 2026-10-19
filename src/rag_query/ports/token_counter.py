"""Token counting port (used by the context budget)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Count model tokens for a piece of text.

    The production adapter uses the local Qwen tokenizer; a heuristic counter
    keeps unit tests and non-LLM deployments dependency-free.
    """

    def count(self, text: str) -> int: ...
