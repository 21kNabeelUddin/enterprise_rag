"""Citation formatting and prompt-safety helpers.

Pure functions: no I/O, no framework imports. The formatter is shared by the
format node, the CLI and the API so a source label can never drift between
surfaces.
"""

from __future__ import annotations

import re

from rag_query.domain.models import Citation, ContextSource

# Standard Form 10-K item titles. Unknown keys fall back to "Item {key}".
SECTION_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_CONTROL_TOKEN_RE = re.compile(r"<\|[^|]{0,128}\|>")
_NO_ANSWER_MARKERS = (
    "couldn't find that in the provided filings",
    "could not find that in the provided filings",
    "not found in the provided filings",
    "not provided in the context",
    "does not contain the answer",
)
# The model does not always emit the exact marker above. In particular it may
# name what it could not find and put an abbreviation (for example "Inc.") in
# the middle: "I couldn't find the current stock price of Apple Inc. (AAPL) on
# the NASDAQ in the provided filings." Match the refusal shape, not one string.
_NO_ANSWER_PATTERNS = (
    re.compile(
        r"\b(?:could(?:n't| not)|can(?:not|'t)|unable to)\s+(?:find|locate)\b"
        r".{0,200}?\b(?:in|within)\s+the\s+(?:provided|retrieved|available)\s+"
        r"(?:filings?|context|documents?)\b"
    ),
    re.compile(
        r"\b(?:not|wasn't|weren't)\s+(?:found|provided|available)\s+in\s+the\s+"
        r"(?:provided|retrieved|available)\s+(?:filings?|context|documents?)\b"
    ),
    re.compile(
        r"\bthe\s+(?:provided\s+|retrieved\s+|available\s+)?"
        r"(?:filings?|context|documents?)\s+(?:do|does)\s+not\s+"
        r"(?:contain|include|provide)\s+(?:the\s+)?(?:answer|information|that|this)\b"
    ),
    re.compile(
        r"\b(?:answer|information)\s+(?:is|was)\s+not\s+(?:provided|available)\s+"
        r"(?:in|within)\s+the\s+(?:provided\s+|retrieved\s+)?"
        r"(?:context|filings?|documents?)\b"
    ),
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def section_title(section_key: str) -> str:
    return SECTION_TITLES.get(section_key, f"Item {section_key}")


def edgar_url(cik: str) -> str:
    """Company filings page. We do not store accession numbers, so this is the
    most precise stable link the index supports."""
    cik = (cik or "").strip() or "0"
    return (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=40"
    )


def sanitize_context_text(text: str, *, max_chars: int | None = None) -> str:
    """Strip chat-template control tokens and control characters.

    Retrieved filings are untrusted input. A document could contain a literal
    ``<|im_start|>`` sequence; if passed through to a chat-templated model it
    would be interpreted as a role boundary. Remove the token shape entirely.
    """
    cleaned = _CONTROL_TOKEN_RE.sub(" ", text or "")
    cleaned = _CONTROL_CHARS_RE.sub(" ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if max_chars is not None and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def snippet(text: str, *, max_chars: int = 280) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def is_no_answer(text: str) -> bool:
    """True when the model explicitly reports the answer is not in context."""
    lowered = " ".join((text or "").lower().split())
    if any(marker in lowered for marker in _NO_ANSWER_MARKERS):
        return True
    return any(pattern.search(lowered) is not None for pattern in _NO_ANSWER_PATTERNS)


def extract_citation_ids(text: str) -> tuple[str, ...]:
    """Return citation ids in first-appearance order, e.g. ``("S1", "S2")``."""
    seen: list[str] = []
    for match in _CITATION_RE.finditer(text or ""):
        source_id = f"S{int(match.group(1))}"
        if source_id not in seen:
            seen.append(source_id)
    return tuple(seen)


def strip_unknown_citations(text: str, valid_ids: set[str] | tuple[str, ...]) -> str:
    """Remove ``[S99]`` markers that do not correspond to a supplied source."""
    valid = set(valid_ids)

    def _replace(match: re.Match[str]) -> str:
        source_id = f"S{int(match.group(1))}"
        return match.group(0) if source_id in valid else ""

    return _CITATION_RE.sub(_replace, text or "")


def build_citation(
    source: ContextSource,
    *,
    company: str | None = None,
    snippet_chars: int = 280,
) -> Citation:
    chunk = source.chunk
    return Citation(
        source_id=source.source_id,
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        cik=chunk.cik,
        company=company,
        year=chunk.year,
        section_key=chunk.section_key,
        section_title=section_title(chunk.section_key),
        chunk_index=chunk.chunk_index,
        url=edgar_url(chunk.cik),
        snippet=snippet(sanitize_context_text(chunk.text), max_chars=snippet_chars),
        score=chunk.fused_score,
        rerank_score=chunk.rerank_score,
    )
