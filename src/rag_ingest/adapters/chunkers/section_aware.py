"""Section-aware chunker.

The outer split is free and exact: EDGAR-CORPUS already separates a 10-K into
SEC Items (section_1, section_1A, ...). Each section is then split with the
recursive token splitter. Every chunk keeps its section key so citations can
say *which* item an answer came from, and chunk ids are deterministic.
"""

from __future__ import annotations

from rag_ingest.adapters.chunkers.recursive import RecursiveTokenChunker
from rag_ingest.domain.models import Chunk, ChunkMetadata, Document
from rag_ingest.ports.chunker import TokenCounter


class SectionAwareChunker:
    def __init__(
        self,
        splitter: RecursiveTokenChunker,
        counter: TokenCounter,
        *,
        name: str = "section_aware",
    ) -> None:
        self._splitter = splitter
        self._counter = counter
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for section in document.sections:
            if not section.text.strip():
                continue
            for span in self._splitter.split(section.text):
                metadata = ChunkMetadata(
                    doc_id=document.doc_id,
                    cik=document.cik,
                    year=document.year,
                    section_key=section.key,
                    chunk_index=index,
                    token_count=span.tokens,
                    char_start=span.start,
                    char_end=span.end,
                )
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}:{section.key}:{index}",
                        text=span.text,
                        metadata=metadata,
                    )
                )
                index += 1
        return chunks
