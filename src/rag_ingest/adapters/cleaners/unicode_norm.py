"""Unicode normalisation.

Filing text contains curly quotes, non-breaking spaces, dashes and stray
control characters that hurt both tokenisation and keyword search. This is a
service.cleaner.transform: document in, document out, no side effects.
"""

from __future__ import annotations

import re
import unicodedata

from rag_ingest.domain.models import Document, Section

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
_DASHES = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-"}
_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ").replace("\u202f", " ")
    text = _ZERO_WIDTH.sub("", text)
    for src, dst in {**_DASHES, **_QUOTES}.items():
        text = text.replace(src, dst)
    return text


class UnicodeNormalizer:
    def __init__(self, *, name: str = "unicode_normalizer") -> None:
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
