"""Pass-through extractor for EDGAR-CORPUS.

EDGAR-CORPUS is already extracted text (tables stripped) delivered as separate
Item sections, so no parsing/OCR is required. This adapter simply orders the
sections, normalises their keys and computes a deterministic content hash.

A Docling-based ``Extractor`` would implement the same port for PDF/HTML/Office
sources without any change to the pipeline.
"""

from __future__ import annotations

import re

from rag_ingest.domain.models import Document, RawRecord, Section, content_hash

SECTION_PREFIX = "section_"
_ORDER_RE = re.compile(r"^(\d+)([A-Za-z]*)$")


def _section_key(field: str) -> str:
    return field[len(SECTION_PREFIX) :] if field.startswith(SECTION_PREFIX) else field


def _order_key(field: str) -> tuple[int, str]:
    key = _section_key(field)
    match = _ORDER_RE.match(key)
    if match:
        return int(match.group(1)), match.group(2).upper()
    return 10**6, key


class PassthroughExtractor:
    def __init__(self, *, name: str = "passthrough") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def extract(self, record: RawRecord) -> Document:
        ordered = sorted(record.sections.items(), key=lambda kv: _order_key(kv[0]))
        sections = [
            Section(key=_section_key(field), field=field, ordinal=i, text=text or "")
            for i, (field, text) in enumerate(ordered)
        ]
        digest = content_hash(
            record.filename + "\x00" + "\x00".join(f"{s.field}={s.text}" for s in sections)
        )
        return Document(
            doc_id=record.filename,
            cik=record.cik,
            year=record.year,
            source=self._name,
            sections=sections,
            content_hash=digest,
        )
