"""Cleaner and extractor behaviour."""

from __future__ import annotations

from rag_ingest.adapters.cleaners import (
    ContinuedHeaderCollapser,
    EmptySectionDropper,
    UnicodeNormalizer,
    WhitespaceCleaner,
)
from rag_ingest.adapters.extractors.passthrough import PassthroughExtractor
from tests.fakes import make_document, make_record


def test_unicode_normalizer_fixes_nbsp_and_quotes() -> None:
    doc = make_document(section_1="Cost\u00a0of \u201crevenue\u201d \u2014 up")
    cleaned = UnicodeNormalizer().clean(doc)
    assert "\u00a0" not in cleaned.sections[0].text
    assert "\u201c" not in cleaned.sections[0].text
    assert "Cost of" in cleaned.sections[0].text


def test_whitespace_cleaner_collapses_but_keeps_content() -> None:
    doc = make_document(section_1="a    b\t\tc\n\n\n\nd")
    cleaned = WhitespaceCleaner().clean(doc)
    assert cleaned.sections[0].text == "a b c\n\nd"


def test_continued_header_is_removed_but_first_heading_kept() -> None:
    text = "ITEM 1A. RISK FACTORS\nreal content\nITEM 1A. RISK FACTORS (continued)\nmore"
    doc = make_document(section_1A=text)
    cleaned = ContinuedHeaderCollapser().clean(doc)
    assert "(continued)" not in cleaned.sections[0].text
    assert "ITEM 1A. RISK FACTORS\n" in cleaned.sections[0].text
    assert "real content" in cleaned.sections[0].text


def test_empty_section_dropper_removes_trivial_sections() -> None:
    doc = make_document(
        section_1="Real content that is long enough to stay here.",
        section_1B="None",
    )
    cleaned = EmptySectionDropper(min_chars=25).clean(doc)
    keys = [s.key for s in cleaned.sections]
    assert keys == ["1"]


def test_extractor_orders_sections_and_normalises_keys() -> None:
    record = make_record(
        **{
            "section_7A": "quant",
            "section_1": "business",
            "section_10": "exhibits",
        }
    )
    doc = PassthroughExtractor().extract(record)
    assert [s.key for s in doc.sections] == ["1", "7A", "10"]
    assert doc.doc_id == record.filename


def test_extractor_hash_is_stable_and_order_independent() -> None:
    a = PassthroughExtractor().extract(make_record(section_1="x", section_2="y"))
    b = PassthroughExtractor().extract(make_record(section_2="y", section_1="x"))
    assert a.content_hash == b.content_hash
