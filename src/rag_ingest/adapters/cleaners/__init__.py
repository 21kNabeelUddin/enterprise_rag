"""Cleaner adapters."""

from rag_ingest.adapters.cleaners.continued_headers import ContinuedHeaderCollapser
from rag_ingest.adapters.cleaners.empty_sections import EmptySectionDropper
from rag_ingest.adapters.cleaners.unicode_norm import UnicodeNormalizer
from rag_ingest.adapters.cleaners.whitespace import WhitespaceCleaner

__all__ = [
    "ContinuedHeaderCollapser",
    "EmptySectionDropper",
    "UnicodeNormalizer",
    "WhitespaceCleaner",
]
