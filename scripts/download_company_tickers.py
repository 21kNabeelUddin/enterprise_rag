#!/usr/bin/env python
"""Download SEC's public CIK -> company/ticker directory.

    .\\.venv\\Scripts\\python.exe .\\scripts\\download_company_tickers.py

The SEC requires a descriptive User-Agent with contact information; pass your
own with ``--user-agent`` when deploying this beyond a local demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

SEC_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_OUT = Path("data/company_tickers.json")
DEFAULT_UA = "enterprise-rag/0.1 (local research; contact: admin@example.com)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--user-agent", default=DEFAULT_UA)
    args = parser.parse_args()

    response = httpx.get(
        SEC_URL,
        headers={"User-Agent": args.user_agent, "Accept": "application/json"},
        timeout=60.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload:
        raise SystemExit("unexpected response shape from SEC")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(payload)} companies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
