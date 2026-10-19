#!/usr/bin/env python
"""Behavioural + latency evaluation for Pipeline 2, driven over the HTTP API.

Runs a small labelled question set (``eval/questions.jsonl``) against a running
app and writes machine-readable results (``eval_results.json``) plus a summary
table. No models are loaded here: start the app first, then run this script, so
the measured latency is what a caller actually experiences.

Question categories and what counts as correct
---------------------------------------------
``answerable``  status ``answered``, not the "couldn't find" phrase, >= 1
                citation, and the output guard allowed the answer.
``refuse``      status ``no_context`` / ``not_answerable``, or the generator
                returned the exact no-answer phrase (no hallucinated answer).
``block``       status ``blocked`` (input guard tripped).

Usage
-----
    .\\.venv\\Scripts\\python.exe .\\scripts\\eval_rag.py
    .\\.venv\\Scripts\\python.exe .\\scripts\\eval_rag.py --only apple-risk-factors tesla-liquidity
    .\\.venv\\Scripts\\python.exe .\\scripts\\eval_rag.py --out eval_results_baseline.json
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

from rag_query.domain.citations import extract_citation_ids, is_no_answer

REPO = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = REPO / "eval" / "questions.jsonl"
DEFAULT_OUT = REPO / "eval_results.json"


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def classify(expected: str, body: dict[str, Any]) -> tuple[str, bool]:
    """Return ``(actual_outcome, correct)`` for one response."""
    status = str(body.get("status", ""))
    answer = body.get("answer") or ""
    citations = body.get("citations") or []
    guard = body.get("guard") or {}
    output_guard = body.get("output_guard") or {}

    if status == "blocked":
        return "blocked", expected == "block"
    if expected == "block":
        return "not_blocked", False

    refused = status in ("no_context", "not_answerable") or is_no_answer(answer)
    if expected == "refuse":
        return ("refused", True) if refused else ("answered", False)

    # expected == "answerable"
    answered = status == "answered" and not is_no_answer(answer)
    grounded = bool(output_guard.get("allowed", True))
    ok = answered and bool(citations) and grounded and bool(guard.get("allowed", True))
    if not answered:
        return ("refused", False)
    if not citations:
        return ("answered_no_citation", False)
    if not grounded:
        return ("ungrounded", False)
    return "answered", ok


def citation_stats(body: dict[str, Any]) -> tuple[int, int, bool]:
    """(cited ids, supplied ids, all cited ids are real)."""
    answer = body.get("answer") or ""
    cited = set(extract_citation_ids(answer))
    supplied = {c["source_id"] for c in (body.get("citations") or [])}
    return len(cited), len(supplied), cited.issubset(supplied)


def run_question(
    client: httpx.Client,
    base_url: str,
    token: str,
    q: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    payload = {"query": q["query"], "session_id": f"eval-{q['id']}-{uuid4().hex[:6]}"}
    headers = {"X-Dev-Token": token} if token else {}
    started = time.perf_counter()
    response = client.post(f"{base_url}/ask", json=payload, headers=headers, timeout=timeout_s)
    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - surface the raw status
        body = {"status": "http_error", "message": response.text[:500]}

    actual, correct = classify(q["category"], body)
    cited, supplied, citation_valid = citation_stats(body)
    timings = {k: float(v) for k, v in (body.get("timings_ms") or {}).items()}
    retrieval = body.get("retrieval") or {}
    context = body.get("context") or {}
    generation = body.get("generation") or {}
    return {
        "id": q["id"],
        "category": q["category"],
        "query": q["query"],
        "notes": q.get("notes", ""),
        "http_status": response.status_code,
        "status": body.get("status"),
        "expected": q["category"],
        "actual": actual,
        "correct": correct,
        "latency_ms": round(latency_ms, 1),
        "timings_ms": timings,
        "prompt_tokens": int(generation.get("prompt_tokens") or 0),
        "completion_tokens": int(generation.get("completion_tokens") or 0),
        "cached_tokens": int(generation.get("cached_tokens") or 0),
        "cache_hit_rate": float(generation.get("cache_hit_rate") or 0.0),
        "prompt_eval_tokens": int(generation.get("prompt_eval_tokens") or 0),
        "prompt_eval_ms": float(generation.get("prompt_eval_ms") or 0.0),
        "decode_ms": float(generation.get("decode_ms") or 0.0),
        "prompt_eval_tokens_per_s": float(
            generation.get("prompt_eval_tokens_per_s") or 0.0
        ),
        "decode_tokens_per_s": float(generation.get("decode_tokens_per_s") or 0.0),
        "finish_reason": generation.get("finish_reason") or "",
        "answer_chars": len(body.get("answer") or ""),
        "citations": len(body.get("citations") or []),
        "cited_ids": cited,
        "supplied_ids": supplied,
        "citation_valid": citation_valid,
        "input_guard_allowed": (body.get("guard") or {}).get("allowed"),
        "output_guard_allowed": (body.get("output_guard") or {}).get("allowed"),
        "no_answer_phrase": is_no_answer(body.get("answer") or ""),
        "retrieval": {
            "dense": retrieval.get("dense_candidates"),
            "lexical": retrieval.get("lexical_candidates"),
            "fused": retrieval.get("fused_candidates"),
            "reranked": retrieval.get("reranked"),
            "expanded": retrieval.get("expanded"),
        },
        "context": context,
        "warnings": body.get("warnings") or [],
        "answer_head": (body.get("answer") or "")[:300],
    }


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def _latency(rows: list[dict[str, Any]], key: str = "latency_ms") -> dict[str, float]:
    values = sorted(r[key] for r in rows)
    if not values:
        return {}
    return {
        "mean": round(statistics.fmean(values), 1),
        "p50": round(statistics.median(values), 1),
        "min": round(values[0], 1),
        "max": round(values[-1], 1),
        "p95": round(values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))], 1),
    }


def summarize(rows: list[dict[str, Any]], health: dict[str, Any]) -> dict[str, Any]:
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cat.setdefault(row["expected"], []).append(row)

    answerable = by_cat.get("answerable", [])
    refuse = by_cat.get("refuse", [])
    block = by_cat.get("block", [])

    answered_rows = [r for r in rows if r["actual"] == "answered"]
    step_names = [
        "history", "input_guard", "route", "rewrite", "encode", "retrieve",
        "rerank", "context", "generate", "output_guard", "finalize",
    ]
    step_means = {}
    answered_step_means = {}
    for step in step_names:
        values = [r["timings_ms"][step] for r in rows if step in r["timings_ms"]]
        if values:
            step_means[step] = round(statistics.fmean(values), 1)
        answered_values = [
            r["timings_ms"][step] for r in answered_rows if step in r["timings_ms"]
        ]
        if answered_values:
            answered_step_means[step] = round(statistics.fmean(answered_values), 1)

    prompt_tokens = [r["prompt_tokens"] for r in answered_rows if r["prompt_tokens"] > 0]
    completion_tokens = [
        r["completion_tokens"] for r in answered_rows if r["completion_tokens"] > 0
    ]
    cached_tokens = sum(r["cached_tokens"] for r in answered_rows)
    prompt_eval_ms = sum(r["prompt_eval_ms"] for r in answered_rows)
    prompt_eval_tokens = sum(r["prompt_eval_tokens"] for r in answered_rows)
    decode_ms = sum(r["decode_ms"] for r in answered_rows)
    generation_ms = sum(
        r["timings_ms"].get("generate", 0.0)
        for r in answered_rows
        if r["completion_tokens"] > 0
    )
    tokens = {
        "prompt_mean": round(statistics.fmean(prompt_tokens), 1) if prompt_tokens else 0.0,
        "completion_mean": (
            round(statistics.fmean(completion_tokens), 1) if completion_tokens else 0.0
        ),
        "cached_mean": (
            round(cached_tokens / len(answered_rows), 1) if answered_rows else 0.0
        ),
        "cached_total": cached_tokens,
        "prompt_eval_tokens_total": prompt_eval_tokens,
        "prompt_eval_ms_total": round(prompt_eval_ms, 1),
        "cache_hit_rate": (
            round(cached_tokens / sum(prompt_tokens), 4) if prompt_tokens else 0.0
        ),
        "prompt_eval_tokens_per_s": (
            round(prompt_eval_tokens / (prompt_eval_ms / 1000.0), 1)
            if prompt_eval_ms > 0 and prompt_eval_tokens
            else 0.0
        ),
        "decode_tokens_per_s": (
            round(sum(completion_tokens) / (decode_ms / 1000.0), 1)
            if decode_ms > 0 and completion_tokens
            else 0.0
        ),
        "generation_tokens_per_s": (
            round(sum(completion_tokens) / (generation_ms / 1000.0), 1)
            if generation_ms > 0 and completion_tokens
            else 0.0
        ),
    }

    return {
        "counts": {
            "total": len(rows),
            "answerable": len(answerable),
            "refuse": len(refuse),
            "block": len(block),
        },
        "accuracy": {
            "answer_rate": _pct(sum(r["correct"] for r in answerable), len(answerable)),
            "refusal_accuracy": _pct(sum(r["correct"] for r in refuse), len(refuse)),
            "guard_detection_rate": _pct(sum(r["correct"] for r in block), len(block)),
            "overall_behavior_accuracy": _pct(sum(r["correct"] for r in rows), len(rows)),
            "grounding_pass_rate": _pct(
                sum(1 for r in answered_rows if r["output_guard_allowed"] is not False),
                len(answered_rows),
            ),
            "citation_validity_rate": _pct(
                sum(1 for r in answered_rows if r["citation_valid"]),
                len(answered_rows),
            ),
            "answer_citation_rate": _pct(
                sum(1 for r in answered_rows if r["citations"] > 0),
                len(answered_rows),
            ),
            "guard_false_positive_rate": _pct(
                sum(1 for r in rows if r["expected"] != "block" and r["status"] == "blocked"),
                len([r for r in rows if r["expected"] != "block"]),
            ),
        },
        "latency_ms": {
            "all": _latency(rows),
            "answerable": _latency(answerable) if answerable else {},
            "per_step_mean": step_means,
            "per_step_mean_answerable": answered_step_means,
        },
        "tokens": tokens,
        "health": health,
    }


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    head = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join(
        "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip() for row in rows
    )
    return f"{head}\n{sep}\n{body}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--token", default="dev-public")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", help="restrict to these question ids")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    questions = [q for q in load_questions(args.questions) if not args.only or q["id"] in args.only]
    if not questions:
        print("no questions selected", file=sys.stderr)
        return 2

    base = args.base_url.rstrip("/")
    with httpx.Client() as client:
        try:
            health = client.get(f"{base}/health", timeout=10.0).json()
        except Exception as exc:  # noqa: BLE001
            print(f"app not reachable at {base}: {exc}", file=sys.stderr)
            return 2

        print(f"evaluating {len(questions)} questions against {base} ...", file=sys.stderr)
        rows: list[dict[str, Any]] = []
        for i, q in enumerate(questions, 1):
            row = run_question(client, base, args.token, q, args.timeout)
            rows.append(row)
            mark = "PASS" if row["correct"] else "FAIL"
            print(
                f"  [{i}/{len(questions)}] {row['id']:<24} {row['category']:<10} "
                f"{mark:<4} status={row['status']:<14} {row['latency_ms']:>8,.0f} ms",
                file=sys.stderr,
            )

    summary = summarize(rows, health)
    payload = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_url": base,
        "summary": summary,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(
        _table(
            ["question", "category", "expected -> actual", "status", "latency ms", "cites", "ok"],
            [
                [
                    r["id"],
                    r["category"],
                    f"{r['expected']} -> {r['actual']}",
                    str(r["status"]),
                    f"{r['latency_ms']:,.0f}",
                    str(r["citations"]),
                    "yes" if r["correct"] else "NO",
                ]
                for r in rows
            ],
        )
    )
    print()
    acc = summary["accuracy"]
    lat = summary["latency_ms"]
    print("ACCURACY (behavioural, labelled set)")
    for key, value in acc.items():
        print(f"  {key:<28} {value:>6.1f} %")
    print("LATENCY")
    all_lat = lat["all"]
    print(
        f"  all      mean={all_lat['mean']:,.0f}  p50={all_lat['p50']:,.0f}  "
        f"p95={all_lat['p95']:,.0f}  min={all_lat['min']:,.0f}  max={all_lat['max']:,.0f} ms"
    )
    if lat["answerable"]:
        a = lat["answerable"]
        print(
            f"  answered mean={a['mean']:,.0f}  p50={a['p50']:,.0f}  "
            f"p95={a['p95']:,.0f}  min={a['min']:,.0f}  max={a['max']:,.0f} ms"
        )
    print("  per-step mean ms (answerable):", lat["per_step_mean_answerable"])
    tok = summary["tokens"]
    print(
        f"TOKENS  prompt/query={tok['prompt_mean']:,.0f}  "
        f"completion/query={tok['completion_mean']:,.0f}  "
        f"cache_hit={tok['cache_hit_rate'] * 100:,.1f}%  "
        f"prompt_eval={tok['prompt_eval_tokens_per_s']:,.1f} tok/s  "
        f"decode={tok['decode_tokens_per_s']:,.1f} tok/s"
    )
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
