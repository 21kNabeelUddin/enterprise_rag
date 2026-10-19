#!/usr/bin/env python
"""Repeat-query probe for llama.cpp prompt/KV cache behaviour.

Sends the same question several times through the running API and reports, per
run, the generation-stage telemetry that E2 added:

  * ``cached_tokens`` / ``cache_hit_rate`` - prompt tokens reused from the KV
    cache (``timings.cache_n`` reported by llama.cpp)
  * ``prompt_eval_ms`` - time spent evaluating the uncached prompt tokens
  * ``decode_ms`` / ``decode_tokens_per_s`` - generation timing

The first request after a server restart (or after a very different prompt) is
the cold observation; later identical requests should show a high hit rate and
much lower prompt-eval time. Run it with ``RAG_LLM_CACHE_PROMPT=true`` and
``false`` for the A/B comparison.

Usage
-----
    .\\.venv\\Scripts\\python.exe .\\scripts\\bench_prompt_cache.py --repeats 3
    .\\.venv\\Scripts\\python.exe .\\scripts\\bench_prompt_cache.py `
        --out prompt_cache_on.json --label cache-on
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

REPO = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = "What were Apple's main sources of revenue in 2020?"


def run_once(client: httpx.Client, base: str, token: str, query: str) -> dict[str, Any]:
    payload = {"query": query, "session_id": f"cache-probe-{uuid4().hex[:8]}"}
    headers = {"X-Dev-Token": token} if token else {}
    started = time.perf_counter()
    response = client.post(f"{base}/ask", json=payload, headers=headers, timeout=300.0)
    latency_ms = (time.perf_counter() - started) * 1000.0
    body = response.json()
    generation = body.get("generation") or {}
    timings = body.get("timings_ms") or {}
    return {
        "http_status": response.status_code,
        "status": body.get("status"),
        "wall_ms": round(latency_ms, 1),
        "generate_ms": round(float(timings.get("generate", 0.0)), 1),
        "prompt_tokens": int(generation.get("prompt_tokens") or 0),
        "cached_tokens": int(generation.get("cached_tokens") or 0),
        "cache_hit_rate": float(generation.get("cache_hit_rate") or 0.0),
        "prompt_eval_ms": float(generation.get("prompt_eval_ms") or 0.0),
        "decode_ms": float(generation.get("decode_ms") or 0.0),
        "completion_tokens": int(generation.get("completion_tokens") or 0),
        "prompt_eval_tokens_per_s": float(
            generation.get("prompt_eval_tokens_per_s") or 0.0
        ),
        "decode_tokens_per_s": float(generation.get("decode_tokens_per_s") or 0.0),
        "citations": len(body.get("citations") or []),
        "answer_chars": len(body.get("answer") or ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--token", default="dev-public")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", default="prompt-cache")
    parser.add_argument("--out", type=Path, default=REPO / "prompt_cache_probe.json")
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    rows: list[dict[str, Any]] = []
    with httpx.Client() as client:
        try:
            health = client.get(f"{base}/health", timeout=10.0).json()
        except Exception as exc:  # noqa: BLE001
            print(f"app not reachable at {base}: {exc}", file=sys.stderr)
            return 2

        print(f"query: {args.query!r}  repeats={args.repeats}", file=sys.stderr)
        for index in range(args.repeats):
            row = run_once(client, base, args.token, args.query)
            rows.append(row)
            print(
                f"  [{index + 1}/{args.repeats}] status={row['status']:<10} "
                f"wall={row['wall_ms']:>8,.0f} ms  generate={row['generate_ms']:>8,.0f} ms  "
                f"prompt_eval={row['prompt_eval_ms']:>7,.0f} ms  "
                f"cache_hit={row['cache_hit_rate'] * 100:5.1f}% "
                f"({row['cached_tokens']}/{row['prompt_tokens']})",
                file=sys.stderr,
            )

    warm = rows[1:]
    summary: dict[str, Any] = {
        "label": args.label,
        "query": args.query,
        "repeats": args.repeats,
        "health": {
            key: health.get(key)
            for key in ("pipeline", "llm_model", "guards", "rerank_device")
        },
        "cold": rows[0] if rows else {},
        "warm_mean": {},
    }
    if warm:
        summary["warm_mean"] = {
            "wall_ms": round(statistics.fmean(r["wall_ms"] for r in warm), 1),
            "generate_ms": round(statistics.fmean(r["generate_ms"] for r in warm), 1),
            "prompt_eval_ms": round(
                statistics.fmean(r["prompt_eval_ms"] for r in warm), 1
            ),
            "cache_hit_rate": round(
                statistics.fmean(r["cache_hit_rate"] for r in warm), 4
            ),
            "decode_tokens_per_s": round(
                statistics.fmean(r["decode_tokens_per_s"] for r in warm), 1
            ),
        }

    payload = {"label": args.label, "summary": summary, "runs": rows}
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
