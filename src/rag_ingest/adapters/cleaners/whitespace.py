"""Whitespace tidying.

Collapses runs of spaces/tabs, trims trailing spaces on lines and caps blank
line runs. Deliberately conservative: it never removes non-whitespace content.
"""

from __future__ import annotations

import re

from rag_ingest.domain.models import Document, Section

_MULTISPACE = re.compile(r"[ \t]{2,}")
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    text = _TRAILING.sub("", text)
    text = _MULTISPACE.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


class WhitespaceCleaner:
    def __init__(self, *, name: str = "whitespace") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def clean(self, document: Document) -> Document:
        sections = [
            Section(key=s.key, field=s.field, ordinal=s.ordinal, text=_clean_text(s.text))
            for s in document.sections
        ]
        return document.model_copy(update={"sections": sections})
