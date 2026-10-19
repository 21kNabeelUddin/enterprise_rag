"""Recursive token chunker: budgets, coverage, overlap, hard split."""

from __future__ import annotations

from rag_ingest.adapters.chunkers.recursive import RecursiveTokenChunker
from rag_ingest.adapters.chunkers.section_aware import SectionAwareChunker
from tests.fakes import CharCounter, WordCounter, make_document


def _splitter(counter, target: int, overlap: int, max_tokens: int) -> RecursiveTokenChunker:
    return RecursiveTokenChunker(
        counter, target_tokens=target, overlap_tokens=overlap, max_tokens=max_tokens
    )


def test_chunks_respect_target_and_cover_text() -> None:
    text = " ".join(f"w{i}." for i in range(200))
    chunker = _splitter(WordCounter(), target=20, overlap=5, max_tokens=500)
    spans = chunker.split(text)

    assert spans
    assert all(s.tokens <= 20 for s in spans)
    assert spans[0].start == 0
    assert spans[-1].end == len(text)
    for span in spans:
        assert text[span.start : span.end] == span.text
        assert span.text == span.text.strip()


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"w{i}." for i in range(200))
    spans = _splitter(WordCounter(), target=20, overlap=5, max_tokens=500).split(text)
    assert len(spans) > 2
    for prev, nxt in zip(spans, spans[1:], strict=False):
        assert nxt.start < prev.end


def test_max_tokens_is_a_hard_cap_for_pathological_units() -> None:
    # a single run with no sentence/paragraph boundary must be char-split
    text = "x" * 1000
    spans = _splitter(CharCounter(), target=100, overlap=10, max_tokens=200).split(text)
    assert spans
    assert all(s.tokens <= 200 for s in spans)
    assert "".join(s.text for s in spans) == text


def test_blank_text_yields_no_chunks() -> None:
    assert _splitter(WordCounter(), 20, 5, 500).split("   \n\n  ") == []


def test_section_aware_chunker_is_deterministic_and_keeps_sections() -> None:
    doc = make_document(
        section_1="Business. " * 50,
        section_1A="Risk. " * 50,
    )
    chunker = SectionAwareChunker(
        _splitter(WordCounter(), target=30, overlap=5, max_tokens=500), WordCounter()
    )
    chunks = chunker.chunk(doc)

    assert chunks
    expected_ids = [
        f"{doc.doc_id}:{c.metadata.section_key}:{c.metadata.chunk_index}" for c in chunks
    ]
    assert [c.chunk_id for c in chunks] == expected_ids
    assert {c.metadata.section_key for c in chunks} == {"1", "1A"}
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))

    again = chunker.chunk(doc)
    assert [c.chunk_id for c in again] == [c.chunk_id for c in chunks]
    assert [c.text for c in again] == [c.text for c in chunks]
