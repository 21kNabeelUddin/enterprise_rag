"""Extractor port: raw record -> cleaned Document.

For EDGAR-CORPUS the text is already extracted, so the default adapter is a
pass-through. The port exists so a Docling/PDF extractor can be dropped in for
other sources without touching the pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_ingest.domain.models import Document, RawRecord


@runtime_checkable
class Extractor(Protocol):
    @property
    def name(self) -> str: ...

    def extract(self, record: RawRecord) -> Document: ...
