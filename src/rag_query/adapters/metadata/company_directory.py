"""Load SEC's public company_tickers.json into a CIK directory.

The file is optional. If it is absent the pipeline still works; it simply cannot
resolve a company name to a CIK filter and citations show the CIK only.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_query.domain.query_analysis import (
    CompanyDirectory,
    aliases_for_company,
)
from rag_query.logging import get_logger

log = get_logger(__name__)


def load_company_directory(path: str | Path | None) -> CompanyDirectory:
    if not path:
        return CompanyDirectory()
    file_path = Path(path)
    if not file_path.exists():
        log.info("company directory %s not found; company-name filters disabled", file_path)
        return CompanyDirectory()

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read company directory %s: %s", file_path, exc)
        return CompanyDirectory()

    name_to_ciks: dict[str, set[str]] = {}
    ticker_to_cik: dict[str, str] = {}
    name_by_cik: dict[str, str] = {}

    rows = payload.values() if isinstance(payload, dict) else payload
    for row in rows:
        if not isinstance(row, dict):
            continue
        cik = str(row.get("cik_str") or row.get("cik") or "").strip()
        title = str(row.get("title") or row.get("name") or "").strip()
        ticker = str(row.get("ticker") or "").strip()
        if not cik:
            continue
        if title:
            name_by_cik[cik] = title
        if ticker:
            ticker_to_cik[ticker.upper()] = cik
        for alias in aliases_for_company(title):
            name_to_ciks.setdefault(alias, set()).add(cik)

    directory = CompanyDirectory(
        name_to_ciks={key: tuple(sorted(value)) for key, value in name_to_ciks.items()},
        ticker_to_cik=ticker_to_cik,
        name_by_cik=name_by_cik,
    )
    log.info(
        "loaded company directory: %d aliases, %d tickers",
        len(name_to_ciks),
        len(ticker_to_cik),
    )
    return directory
