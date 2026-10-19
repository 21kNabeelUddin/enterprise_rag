"""Drop empty / trivially short sections.

Many filings leave items blank (e.g. Item 1B "None"). Keeping them wastes
chunks and pollutes retrieval. A section is dropped when, after cleaning, it
has fewer than ``min_chars`` non-whitespace characters.
"""

from __future__ import annotations

from rag_ingest.domain.models import Document


class EmptySectionDropper:
    def __init__(self, *, min_chars: int = 25, name: str = "empty_sections") -> None:
        self._min_chars = min_chars
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def clean(self, document: Document) -> Document:
        kept = [s for s in document.sections if len(s.text.strip()) >= self._min_chars]
        return document.model_copy(update={"sections": kept})
