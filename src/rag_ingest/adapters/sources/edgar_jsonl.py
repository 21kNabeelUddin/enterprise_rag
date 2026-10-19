"""Streaming reader for EDGAR-CORPUS jsonl files.

Each line is a JSON object with ``filename``, ``cik``, ``year`` and up to 15
``section_*`` string fields. The reader streams line by line so the 1.45 GiB
train file never has to fit in memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import orjson

from rag_ingest.domain.models import RawRecord

SECTION_PREFIX = "section_"


class EdgarJsonlReader:
    def __init__(self, path: str | Path, *, name: str = "edgar_jsonl") -> None:
        self._path = Path(path)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> Iterator[RawRecord]:
        if not self._path.exists():
            raise FileNotFoundError(f"dataset file not found: {self._path}")

        with self._path.open("rb") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    obj = orjson.loads(line)
                except orjson.JSONDecodeError as exc:  # pragma: no cover - corrupt input
                    raise ValueError(f"{self._path}:{line_no}: invalid json: {exc}") from exc

                sections = {k: v for k, v in obj.items() if k.startswith(SECTION_PREFIX)}
                yield RawRecord(
                    filename=str(obj["filename"]),
                    cik=str(obj.get("cik", "")),
                    year=str(obj.get("year", "")),
                    sections=sections,
                )
