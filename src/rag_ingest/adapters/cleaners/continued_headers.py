"""Collapse repeated page-break headers.

10-K text repeats the Item heading on every page ("ITEM 1A. RISK FACTORS
(continued)"). Those repeats are navigation artifacts, not content, and they
confuse both chunking and keyword search. This cleaner removes only lines that
are an Item heading explicitly marked "(continued)", which is safe: it never
deletes a first occurrence or real prose.
"""

from __future__ import annotations

import re

from rag_ingest.domain.models import Document, Section

_CONTINUED = re.compile(
    r"^[ \t]*ITEM\s+\d+[A-Z]?\.\s+.*\(continued\)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


class ContinuedHeaderCollapser:
    def __init__(self, *, name: str = "continued_headers") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def clean(self, document: Document) -> Document:
        sections = []
        for s in document.sections:
            text = _CONTINUED.sub("", s.text)
            sections.append(Section(key=s.key, field=s.field, ordinal=s.ordinal, text=text))
        return document.model_copy(update={"sections": sections})
