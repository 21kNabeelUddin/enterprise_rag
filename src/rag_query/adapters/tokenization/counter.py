"""Token counting adapters.

The production counter uses the local Qwen tokenizer (the reranker ships the
same Qwen tokenizer family as the generator). A heuristic counter keeps tests
and CPU-only deployments working without loading a tokenizer.
"""

from __future__ import annotations

import math
from pathlib import Path

from rag_query.logging import get_logger

log = get_logger(__name__)


class HeuristicTokenCounter:
    """Approximate counter: one token per ``chars_per_token`` characters."""

    def __init__(self, *, chars_per_token: float = 4.0) -> None:
        self._chars_per_token = max(1.0, chars_per_token)

    def count(self, text: str) -> int:
        text = text or ""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self._chars_per_token))


class TransformersTokenCounter:
    """Exact counter backed by a local Hugging Face tokenizer.

    Loading is lazy and falls back to :class:`HeuristicTokenCounter` if the
    directory is missing or the tokenizer cannot be loaded, so a broken tokenizer
    never takes the service down.
    """

    def __init__(
        self,
        tokenizer_dir: str | Path | None,
        *,
        fallback: HeuristicTokenCounter | None = None,
        local_files_only: bool = True,
    ) -> None:
        self._tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None
        self._fallback = fallback or HeuristicTokenCounter()
        self._tokenizer = None

    def count(self, text: str) -> int:
        text = text or ""
        if not text:
            return 0
        tokenizer = self._ensure_tokenizer()
        if tokenizer is None:
            return self._fallback.count(text)
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001 - counting must never be fatal
            return self._fallback.count(text)

    def _ensure_tokenizer(self):  # noqa: ANN202
        if self._tokenizer is not None or self._tokenizer_dir is None:
            return self._tokenizer
        if not self._tokenizer_dir.exists():
            log.info("tokenizer dir %s missing; using heuristic counter", self._tokenizer_dir)
            self._tokenizer_dir = None
            return None
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._tokenizer_dir),
                local_files_only=True,
                use_fast=True,
            )
            log.info("loaded context tokenizer from %s", self._tokenizer_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "could not load tokenizer from %s (%s); using heuristic",
                self._tokenizer_dir,
                exc,
            )
            self._tokenizer_dir = None
        return self._tokenizer


def build_token_counter(tokenizer_dir: str | Path | None) -> TransformersTokenCounter:
    return TransformersTokenCounter(tokenizer_dir)
