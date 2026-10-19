"""Deterministic query analysis for retrieval filters.

This is deliberately conservative: a wrong hard filter is worse than no filter.
We only extract metadata the user stated explicitly (CIK, a corpus year, a named
10-K item, or a company name that resolves unambiguously in the local SEC
directory).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from rag_query.domain.models import RetrievalFilters

_CIK_RE = re.compile(r"\bcik\s*(?:number|no\.?|#)?\s*[:\-]?\s*(\d{3,10})\b", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9&.'-]*")

# Explicit phrases only. "business" alone is too ambiguous to filter on.
_SECTION_PHRASES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple(
    (re.compile(pattern, re.I), keys)
    for pattern, keys in (
        (r"\bitem\s+1a\b|\brisk\s+factors?\b", ("1A",)),
        (r"\bitem\s+1b\b|\bunresolved\s+staff\s+comments?\b", ("1B",)),
        (r"\bitem\s+1\b|\bbusiness\s+section\b", ("1",)),
        (r"\bitem\s+2\b|\bproperties\s+section\b", ("2",)),
        (r"\bitem\s+3\b|\blegal\s+proceedings?\b|\blitigation\s+section\b", ("3",)),
        (r"\bitem\s+4\b|\bmine\s+safety\b", ("4",)),
        (r"\bitem\s+7a\b|\bmarket\s+risk\b", ("7A",)),
        (r"\bitem\s+7\b|\bmd\s*&\s*a\b|\bmanagement'?s?\s+discussion\b", ("7",)),
        (r"\bitem\s+8\b|\bfinancial\s+statements?\b", ("8",)),
        (r"\bitem\s+9a\b|\bcontrols?\s+and\s+procedures?\b", ("9A",)),
        (r"\bitem\s+9b\b|\bother\s+information\b", ("9B",)),
        (r"\bitem\s+10\b|\bdirectors?\s+and\s+executive\s+officers?\b", ("10",)),
        (r"\bitem\s+11\b|\bexecutive\s+compensation\b", ("11",)),
        (r"\bitem\s+12\b|\bsecurity\s+ownership\b", ("12",)),
        (r"\bitem\s+13\b|\brelated\s+(?:party\s+)?transactions?\b", ("13",)),
        (r"\bitem\s+14\b|\baccountant\s+fees\b", ("14",)),
        (r"\bitem\s+15\b|\bexhibits?\b", ("15",)),
    )
)

_SUFFIXES = (
    " incorporated",
    " corporation",
    " corp",
    " inc",
    " ltd",
    " limited",
    " plc",
    " company",
    " co",
    " holdings",
    " holding",
    " group",
    " trust",
    " technologies",
    " technology",
    " systems",
    " solutions",
    " and",
)

# Single-word aliases that also appear in ordinary questions. Matching these
# would invent a company filter (and silently hide the right filings). Proper
# multi-word or ticker matches are still allowed.
_GENERIC_ALIASES = frozenset(
    """
    a about above after again against all am an and any are as at be because
    been before below between both but by can compare company companies
    corporation corp could describe did do does doing down during each eight
    energy enterprise even ever every few filing filings find first five for
    from fund further general give global group had has have having he her here
    hers him his holding holdings how i if in inc industries international
    into is it its item just ltd md mda me more most my national new nice no not
    of off on once one only or other our ours out over own please plc report
    reports reliance risk risks same section securities sentinel services seven
    she should show six so some solutions such summarize systems tell ten than
    that the their theirs them then there these they this those three through
    to too trust two under until up us very was we were what when where which
    while who whom why will with would you your yours
    """.split()  # noqa: SIM905 - readable word list
)

_GENERIC_TICKERS = frozenset(
    {"A", "AI", "ALL", "AND", "ARE", "CEO", "CFO", "EU", "FOR", "GDP", "I",
     "IPO", "IT", "MD", "SEC", "THE", "UK", "US", "USA", "YOU"}
)


class CompanyDirectory:
    """CIK lookup built from SEC's public ``company_tickers.json``.

    The directory is optional: when the file is absent, retrieval simply works
    without company-name filters and citations fall back to the CIK.
    """

    def __init__(
        self,
        name_to_ciks: Mapping[str, Iterable[str]] | None = None,
        ticker_to_cik: Mapping[str, str] | None = None,
        name_by_cik: Mapping[str, str] | None = None,
        *,
        max_ngram: int = 4,
    ) -> None:
        self._name_to_ciks: dict[str, tuple[str, ...]] = {}
        for key, ciks in (name_to_ciks or {}).items():
            if not key.strip():
                continue
            values = (ciks,) if isinstance(ciks, str) else tuple(ciks)
            aliases = {_normalise_name(key), *aliases_for_company(key)}
            for alias in aliases:
                if not alias:
                    continue
                bucket = list(self._name_to_ciks.get(alias, ()))
                for cik in values:
                    if str(cik) not in bucket:
                        bucket.append(str(cik))
                self._name_to_ciks[alias] = tuple(bucket)
        self._ticker_to_cik = {
            key.strip().upper(): str(cik) for key, cik in (ticker_to_cik or {}).items()
        }
        self._name_by_cik = {str(k): v for k, v in (name_by_cik or {}).items()}
        self._max_ngram = max(1, max_ngram)

    @property
    def empty(self) -> bool:
        return not self._name_to_ciks and not self._ticker_to_cik

    def name_for_cik(self, cik: str) -> str | None:
        return self._name_by_cik.get(str(cik))

    def resolve(self, query: str) -> tuple[str, ...]:
        """Return CIKs for company names/tickers mentioned in ``query``."""
        if self.empty:
            return ()
        normalised = _normalise_name(query)
        tokens = normalised.split()
        found: list[str] = []
        for size in range(min(self._max_ngram, len(tokens)), 0, -1):
            for start in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[start : start + size])
                if size == 1 and phrase in _GENERIC_ALIASES:
                    continue
                for cik in self._name_to_ciks.get(phrase, ()):
                    if cik not in found:
                        found.append(cik)

        # Explicit uppercase tickers (AAPL, MSFT, TSLA). Only consult the
        # directory, so ordinary acronyms such as SEC are ignored.
        for match in re.finditer(r"\b[A-Z]{2,5}\b", query or ""):
            ticker = match.group(0)
            if ticker in _GENERIC_TICKERS:
                continue
            cik = self._ticker_to_cik.get(ticker)
            if cik and cik not in found:
                found.append(cik)
        return tuple(found)


class MetadataExtractor:
    """Extract conservative retrieval filters from a natural-language query."""

    def __init__(
        self,
        companies: CompanyDirectory | None = None,
        *,
        corpus_years: Iterable[str] = ("2020",),
    ) -> None:
        self._companies = companies or CompanyDirectory()
        self._corpus_years = {str(year) for year in corpus_years}

    def extract(
        self,
        query: str,
        *,
        resolve_companies: bool = True,
    ) -> RetrievalFilters:
        text = query or ""
        cik_match = _CIK_RE.search(text)
        # An explicit CIK wins outright. Do not also resolve company names:
        # "CIK 320193" must not be widened by an incidental word match.
        if cik_match:
            ciks = [cik_match.group(1)]
        elif resolve_companies:
            ciks = list(self._companies.resolve(text))
        else:
            ciks = []

        years = tuple(
            dict.fromkeys(
                match.group(0)
                for match in _YEAR_RE.finditer(text)
                if match.group(0) in self._corpus_years
            )
        )

        sections: list[str] = []
        for pattern, keys in _SECTION_PHRASES:
            if pattern.search(text):
                for key in keys:
                    if key not in sections:
                        sections.append(key)

        return RetrievalFilters(
            ciks=tuple(ciks),
            years=years,
            section_keys=tuple(sections),
        )


def _normalise_name(value: str) -> str:
    value = (value or "").lower().replace("’", "'").replace("&", " and ")
    value = re.sub(r"'s\b", "", value)
    # Drop abbreviation periods at word ends ("Apple Inc." -> "apple inc")
    # while preserving domain dots ("amazon.com").
    value = re.sub(r"(?<=[a-z])\.(?=\s|$)", "", value)
    value = re.sub(r"[^a-z0-9.' ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def aliases_for_company(title: str) -> tuple[str, ...]:
    """Generate the lookup aliases for an SEC company title."""
    normalised = _normalise_name(title)
    if not normalised:
        return ()
    aliases = {normalised}
    stripped = normalised
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)].strip(" ,.")
                if stripped:
                    aliases.add(stripped)
                changed = True
                break
    # "amazon.com" -> "amazon" is a common spoken form.
    if stripped.endswith(".com"):
        aliases.add(stripped[: -len(".com")])
    return tuple(
        alias
        for alias in aliases
        if len(alias) >= 2
        and not (len(alias.split()) == 1 and alias in _GENERIC_ALIASES)
    )
