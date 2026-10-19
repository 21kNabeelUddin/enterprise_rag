"""Token counter backed by a HuggingFace tokenizer.

Bound to the same tokenizer as the embedding model so chunk sizes are measured
in the exact tokens the model will see. No prefix/special tokens are counted,
since BGE adds those itself, but the caller keeps head-room via max_tokens.
"""

from __future__ import annotations

from pathlib import Path


class HFTokenCounter:
    def __init__(self, tokenizer_dir: str | Path, *, name: str = "hf_tokenizer") -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        # We only ever *count* tokens here (never truncate/pad), so lift the
        # model_max_length limit to silence the "sequence longer than
        # model_max_length" warning on long paragraphs.
        self._tokenizer.model_max_length = int(1e9)
        self._name = name
        self._cache: dict[str, int] = {}

    @property
    def name(self) -> str:
        return self._name

    def count(self, text: str) -> int:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        value = len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(self._cache) < 20_000:
            self._cache[text] = value
        return value
