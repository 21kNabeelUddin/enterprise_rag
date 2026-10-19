"""Chunker ports.

``TokenCounter`` is separated from ``Chunker`` so the splitter can be unit
tested with a cheap fake counter and only bound to the real BGE tokenizer at
wiring time.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_ingest.domain.models import Chunk, Document


@runtime_checkable
class TokenCounter(Protocol):
    """Counts model tokens for a piece of text."""

    @property
    def name(self) -> str: ...

    def count(self, text: str) -> int: ...


@runtime_checkable
class Chunker(Protocol):
    @property
    def name(self) -> str: ...

    def chunk(self, document: Document) -> list[Chunk]: ...
