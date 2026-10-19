"""Recursive, token-budgeted text splitter.

Strategy (coarse -> fine, recursing only when a unit is too big):

1. split on blank-line paragraph boundaries
2. split oversized paragraphs on sentence boundaries
3. hard character-split any single sentence still over budget

Adjacent units are then greedily merged up to ``target_tokens``; a trailing
window of at most ``overlap_tokens`` is carried into the next chunk so text
spanning a boundary is not lost. Every chunk carries its exact character
offsets into the section text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_ingest.ports.chunker import TokenCounter

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    tokens: int


@dataclass(frozen=True)
class ChunkSpan:
    start: int
    end: int
    text: str
    tokens: int


class RecursiveTokenChunker:
    def __init__(
        self,
        counter: TokenCounter,
        *,
        target_tokens: int,
        overlap_tokens: int,
        max_tokens: int,
        name: str = "recursive_token",
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be < target_tokens")
        self._counter = counter
        self._target = target_tokens
        self._overlap = overlap_tokens
        self._max = max_tokens
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def split(self, text: str) -> list[ChunkSpan]:
        if not text.strip():
            return []
        units = self._atomic_units(text)
        return self._merge(units, text)

    # -- unit construction -------------------------------------------------
    def _atomic_units(self, text: str) -> list[_Unit]:
        units: list[_Unit] = []
        cursor = 0
        for match in _PARAGRAPH.finditer(text):
            units.extend(self._paragraph_units(text, cursor, match.start()))
            cursor = match.end()
        units.extend(self._paragraph_units(text, cursor, len(text)))
        return [u for u in units if u.end > u.start and text[u.start : u.end].strip()]

    def _paragraph_units(self, text: str, start: int, end: int) -> list[_Unit]:
        if end <= start:
            return []
        tokens = self._counter.count(text[start:end])
        if tokens <= self._target:
            return [_Unit(start, end, tokens)]
        # recurse into sentences
        units: list[_Unit] = []
        cursor = start
        for match in _SENTENCE.finditer(text, start, end):
            units.extend(self._sentence_unit(text, cursor, match.start()))
            cursor = match.end()
        units.extend(self._sentence_unit(text, cursor, end))
        return units

    def _sentence_unit(self, text: str, start: int, end: int) -> list[_Unit]:
        if end <= start or not text[start:end].strip():
            return []
        tokens = self._counter.count(text[start:end])
        if tokens <= self._max:
            return [_Unit(start, end, tokens)]
        return self._hard_split(text, start, end)

    def _hard_split(self, text: str, start: int, end: int) -> list[_Unit]:
        """Binary-search the largest character cut keeping each piece <= target."""
        units: list[_Unit] = []
        lo = start
        while lo < end:
            low, high, best = lo + 1, end, lo + 1
            while low <= high:
                mid = (low + high) // 2
                if self._counter.count(text[lo:mid]) <= self._target:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            cut = max(best, lo + 1)
            if cut >= end:
                units.append(_Unit(lo, end, self._counter.count(text[lo:end])))
                break
            units.append(_Unit(lo, cut, self._counter.count(text[lo:cut])))
            lo = cut
        return units

    # -- merging -----------------------------------------------------------
    def _merge(self, units: list[_Unit], text: str) -> list[ChunkSpan]:
        spans: list[ChunkSpan] = []
        i = 0
        n = len(units)
        while i < n:
            group = [units[i]]
            tokens = units[i].tokens
            j = i + 1
            while j < n and tokens + units[j].tokens <= self._target:
                tokens += units[j].tokens
                group.append(units[j])
                j += 1

            start, end = group[0].start, group[-1].end
            body, real_start, real_end = self._trim(text, start, end)
            if body:
                spans.append(ChunkSpan(real_start, real_end, body, self._counter.count(body)))

            # carry a trailing overlap window into the next chunk
            k = j
            carried = 0
            while k - 1 > i and carried + units[k - 1].tokens <= self._overlap:
                carried += units[k - 1].tokens
                k -= 1
            i = k if k > i else j
        return spans

    @staticmethod
    def _trim(text: str, start: int, end: int) -> tuple[str, int, int]:
        raw = text[start:end]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        return raw.strip(), start + lead, end - trail
