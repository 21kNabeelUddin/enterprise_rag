"""Data source port."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from rag_ingest.domain.models import RawRecord


@runtime_checkable
class SourceReader(Protocol):
    """Streams raw records from a source.

    Implementations must stream (never load a whole file into memory) and be
    safe to iterate once per run.
    """

    @property
    def name(self) -> str: ...

    def read(self) -> Iterator[RawRecord]: ...
