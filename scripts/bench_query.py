#!/usr/bin/env python
"""Benchmark Pipeline 2 (steps 1-14) and report per-step latency.

Runs a curated set of questions -- from empty/1-token up to ~250-word prompts,
plus intent, PII and history variants -- through the *real* pipeline. The
service is built once so the embedding model, reranker, prompt guard and LLM
client are loaded a single time; only the per-request node work is timed.

Usage
-----
    .\\.venv\\Scripts\\python.exe .\\scripts\\bench_query.py --repeat 1
    .\\.venv\\Scripts\\python.exe .\\scripts\\bench_query.py --retrieval-only --repeat 3
    .\\.venv\\Scripts\\python.exe .\\scripts\\bench_query.py --only short-1 medium-1 long-1
    .\\.venv\\Scripts\\python.exe .\\scripts\\bench_query.py --no-guard --out bench.json

Notes
-----
* The per-user rate limit is raised for the run, otherwise a burst of questions
  would start returning ``rate_limited`` and skew the numbers.
* Prompt Guard 2 is enabled by default (use ``--no-guard`` to disable it).
* ``--retrieval-only`` skips the LLM, guards and history: it times query
  embedding, hybrid retrieval, reranking/neighbour expansion and context
  assembly only. This is the fast way to measure steps 8-10 in isolation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from rag_query.config import QuerySettings
from rag_query.domain.models import (
    AskRequest,
    ChatMessage,
    Principal,
    QueryStatus,
    RetrievalFilters,
)
from rag_query.domain.query_analysis import MetadataExtractor
from rag_query.nodes.context import ContextNode
from rag_query.nodes.rerank import RerankNode
from rag_query.nodes.retrieve import RetrieveNode
from rag_query.wiring import (
    build_authenticator,
    build_company_directory,
    build_context_builder,
    build_embedder,
    build_reranker,
    build_retriever,
    build_service,
    build_token_counter,
)

STEP_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "steps 1-7",
        ("input", "authenticate", "history", "input_guard", "route", "rewrite", "encode"),
    ),
    (
        "steps 8-14",
        ("retrieve", "rerank", "context", "generate", "output_guard", "format", "finalize"),
    ),
)
STEPS: tuple[str, ...] = tuple(step for _, steps in STEP_GROUPS for step in steps)

# --- question corpus ------------------------------------------------------
LONG_1 = (
    "In its 2020 annual report, Apple described a broad set of principal risks, "
    "including its dependence on a concentrated supply chain, the possibility of "
    "manufacturing disruptions, exposure to tariffs and international trade "
    "disputes, foreign-exchange volatility, and the difficulty of forecasting "
    "demand for new products. Please summarise each of these areas and explain "
    "which ones Apple said were most likely to materially affect its business, "
    "results of operations, or financial condition going forward."
)

LONG_2 = (
    "Compare how Apple and Microsoft described the risks arising from the "
    "COVID-19 pandemic in their 2020 annual reports, specifically with respect "
    "to manufacturing and logistics on the one hand and cloud demand and remote "
    "work on the other. For each company, identify the language used to "
    "characterise uncertainty, the product lines or segments most exposed, and "
    "any mitigation measures the management discussion and analysis proposed. "
    "Finally, note any differences in tone or emphasis between the two filings."
)

XLONG_1 = (
    "I am preparing an internal briefing on how large technology companies "
    "framed their principal risks during the pandemic year of 2020, and I would "
    "like a detailed comparison drawn strictly from the annual reports filed "
    "with the SEC. Please begin by summarising the risk factors that Apple "
    "disclosed in its 2020 Form 10-K, paying particular attention to supply "
    "chain concentration, manufacturing and logistics disruptions, tariffs and "
    "trade tensions, foreign-exchange movements, and the reliability of demand "
    "forecasts. Then do the same for Microsoft, focusing instead on cloud "
    "infrastructure capacity, cybersecurity and data privacy, competition in "
    "enterprise software, and the sustainability of subscription revenue. After "
    "that, outline the risks Tesla identified, especially around production "
    "ramp, battery supply, regulatory credits, and the concentration of its "
    "operations in a small number of facilities. For each company, state "
    "explicitly whether management characterised the pandemic as a discrete "
    "event or as an ongoing condition, and quote or paraphrase the specific "
    "language used wherever it helps clarify the emphasis."
)

XLONG_2 = (
    "For a diligence exercise, I need an exhaustive, source-grounded review of "
    "the principal risk factors disclosed by Apple, Microsoft, and Tesla in "
    "their 2020 annual reports. Start with a short executive summary of no more "
    "than five sentences. Then produce a structured section for each company "
    "that covers, at a minimum, the following themes: the concentration of "
    "revenue in a small number of products or customers; dependence on "
    "third-party suppliers and single-source components; exposure to "
    "manufacturing, logistics, and workforce disruptions; legal and regulatory "
    "proceedings, including antitrust and privacy investigations; "
    "intellectual-property and licensing disputes; foreign-exchange and "
    "interest-rate movements; the effect of tariffs and shifting trade policy; "
    "cybersecurity incidents and data-protection obligations; environmental and "
    "climate-related requirements; and the challenge of accurately forecasting "
    "demand for new or refreshed products. Where the filings quantify a risk, "
    "report the figure and the period; where they describe a trend, note "
    "whether management expected it to continue. After the company sections, "
    "add a comparative table that ranks the twelve most frequently recurring "
    "themes across all three filings and indicates, for each, which companies "
    "raised it. Conclude with a list of the five risks you would weight most "
    "heavily when assessing downside exposure for the coming fiscal year, and "
    "briefly justify each ranking with reference to the language in the "
    "relevant annual report."
)


@dataclass(frozen=True)
class BenchQuestion:
    qid: str
    bucket: str
    note: str
    query: str
    history: tuple[ChatMessage, ...] = ()


QUESTIONS: tuple[BenchQuestion, ...] = (
    BenchQuestion("empty-space", "empty", "whitespace only", "    "),
    BenchQuestion("empty-newline", "empty", "newline/tab only", "\n\t  "),
    BenchQuestion("tiny-smalltalk", "tiny", "1 token, smalltalk", "hi"),
    BenchQuestion("tiny-tooshort", "tiny", "1 token, too_short", "risks?"),
    BenchQuestion("tiny-bare", "tiny", "3 tokens, answerable", "apple 2020 risks"),
    BenchQuestion("short-1", "short", "~8 words", "What were Apple's main risk factors in 2020?"),
    BenchQuestion(
        "short-2", "short", "~9 words",
        "How did Microsoft describe the pandemic in 2020?",
    ),
    BenchQuestion("short-3", "short", "~8 words", "What did Tesla say about liquidity in 2020?"),
    BenchQuestion(
        "short-pii", "short", "contains PII",
        "My email is alice@example.com, what risks does Apple face?",
    ),
    BenchQuestion(
        "medium-1", "medium", "~20 words",
        "Summarise the principal risks Apple disclosed in its 2020 annual report, "
        "focusing on supply chain and regulatory matters.",
    ),
    BenchQuestion(
        "medium-2", "medium", "~23 words",
        "Compare the risk factors Microsoft and Apple disclosed in 2020 regarding "
        "cloud computing, cybersecurity, and international operations.",
    ),
    BenchQuestion(
        "medium-3", "medium", "~22 words",
        "Did Apple or Microsoft disclose more material legal proceedings in 2020, "
        "and what categories did those proceedings fall into?",
    ),
    BenchQuestion("long-1", "long", "~70 words", LONG_1),
    BenchQuestion("long-2", "long", "~80 words", LONG_2),
    BenchQuestion("xlong-1", "xlong", "~180 words", XLONG_1),
    BenchQuestion("xlong-2", "xlong", "~250 words", XLONG_2),
    BenchQuestion(
        "other-nonenglish", "other", "non-English",
        "Quels sont les principaux risques d'Apple en 2020 ?",
    ),
    BenchQuestion(
        "other-multipart", "other", "multi-part, 3 companies",
        "First, what were Apple's main supply-chain risks in 2020; second, how did "
        "Microsoft describe pandemic-related demand shifts; and third, what did "
        "Tesla report about production capacity?",
    ),
    BenchQuestion(
        "other-followup", "other", "follow-up with chat history",
        "and how did that change in 2020?",
        history=(
            ChatMessage(role="user", content="What were Apple's main risk factors?"),
            ChatMessage(
                role="assistant",
                content=(
                    "Apple highlighted supply-chain concentration, tariffs, and demand uncertainty."
                ),
            ),
        ),
    ),
)


@dataclass
class Run:
    qid: str
    bucket: str
    note: str
    iteration: int
    chars: int
    words: int
    status: str
    trace: list[str]
    timings: dict[str, float]
    total_ms: float
    dim: int | None
    warnings: list[str] = field(default_factory=list)
    dense: int = 0
    lexical: int = 0
    reranked: int = 0
    sources: int = 0
    context_tokens: int = 0
    answer_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_hit_rate: float = 0.0
    prompt_eval_ms: float = 0.0
    decode_ms: float = 0.0
    prompt_eval_tokens_per_s: float = 0.0
    decode_tokens_per_s: float = 0.0


def _run_once(service: Any, q: BenchQuestion, iteration: int) -> Run:
    result = service.ask(
        AskRequest(query=q.query, history=list(q.history)),
        include_vector=False,
    )
    timings = {k: float(v) for k, v in result.timings_ms.items()}
    retrieval = result.retrieval
    context = result.context
    generation = result.generation
    return Run(
        qid=q.qid,
        bucket=q.bucket,
        note=q.note,
        iteration=iteration,
        chars=len(q.query.strip()),
        words=len(q.query.split()),
        status=result.status.value,
        trace=list(result.trace),
        timings=timings,
        total_ms=float(sum(timings.values())),
        dim=result.embedding_dim,
        warnings=list(result.warnings),
        dense=retrieval.dense_candidates if retrieval is not None else 0,
        lexical=retrieval.lexical_candidates if retrieval is not None else 0,
        reranked=retrieval.reranked if retrieval is not None else 0,
        sources=context.sources if context is not None else 0,
        context_tokens=context.tokens if context is not None else 0,
        answer_chars=len(result.answer or ""),
        prompt_tokens=generation.prompt_tokens if generation is not None else 0,
        completion_tokens=generation.completion_tokens if generation is not None else 0,
        cached_tokens=generation.cached_tokens if generation is not None else 0,
        cache_hit_rate=generation.cache_hit_rate if generation is not None else 0.0,
        prompt_eval_ms=generation.prompt_eval_ms if generation is not None else 0.0,
        decode_ms=generation.decode_ms if generation is not None else 0.0,
        prompt_eval_tokens_per_s=(
            generation.prompt_eval_tokens_per_s if generation is not None else 0.0
        ),
        decode_tokens_per_s=(
            generation.decode_tokens_per_s if generation is not None else 0.0
        ),
    )


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": sorted(values)[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))],
    }


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows
    )
    return f"{line}\n{sep}\n{body}"


def _report(runs: list[Run], questions: list[BenchQuestion]) -> str:
    out: list[str] = []
    by_id: dict[str, list[Run]] = {}
    for run in runs:
        by_id.setdefault(run.qid, []).append(run)

    # --- per-question tables ------------------------------------------------
    for label, steps in STEP_GROUPS:
        out.append(
            f"PER-QUESTION {label} (ms per step; mean of repeats, '-' = step did not run)"
        )
        rows: list[list[str]] = []
        for q in questions:
            group = by_id.get(q.qid, [])
            if not group:
                continue
            statuses = {r.status for r in group}
            status = statuses.pop() if len(statuses) == 1 else "/".join(sorted(statuses))
            cells = [q.qid, q.bucket, str(group[0].chars), str(group[0].words), status]
            for step in steps:
                vals = [r.timings[step] for r in group if step in r.timings]
                cells.append(f"{statistics.fmean(vals):,.0f}" if vals else "-")
            totals = [r.total_ms for r in group]
            cells.append(f"{statistics.fmean(totals):,.0f}")
            rows.append(cells)
        out.append(
            _table(
                ["question", "bucket", "chars", "words", "status", *steps, "total"],
                rows,
            )
        )
        out.append("")

    # --- per-bucket totals --------------------------------------------------
    out.append("PER-BUCKET (mean total ms)")
    buckets: list[str] = []
    for q in questions:
        if q.bucket not in buckets:
            buckets.append(q.bucket)
    rows = []
    for bucket in buckets:
        group = [r for r in runs if r.bucket == bucket]
        encoded = [r for r in group if "encode" in r.trace]
        retrieve = [r for r in group if "retrieve" in r.trace]
        rows.append([
            bucket,
            str(len(group)),
            f"{statistics.fmean([r.total_ms for r in group]):,.0f}",
            f"{statistics.fmean([r.timings.get('rewrite', 0.0) for r in encoded]):,.0f}"
            if encoded
            else "-",
            f"{statistics.fmean([r.timings.get('encode', 0.0) for r in encoded]):,.0f}"
            if encoded
            else "-",
            f"{statistics.fmean([r.timings.get('retrieve', 0.0) for r in retrieve]):,.0f}"
            if retrieve
            else "-",
            f"{statistics.fmean([r.timings.get('rerank', 0.0) for r in retrieve]):,.0f}"
            if retrieve
            else "-",
            f"{statistics.fmean([r.timings.get('generate', 0.0) for r in group]):,.0f}"
            if any("generate" in r.timings for r in group)
            else "-",
        ])
    out.append(
        _table(
            ["bucket", "n", "total", "rewrite", "encode", "retrieve", "rerank", "generate"],
            rows,
        )
    )

    # --- per-step summary over full (encoded) runs --------------------------
    full = [r for r in runs if "encode" in r.trace]
    out.append("")
    out.append(f"PER-STEP over {len(full)} encoded runs (of {len(runs)} total)")
    rows = []
    total_mean = statistics.fmean([r.total_ms for r in full]) if full else 0.0
    for step in STEPS:
        vals = [r.timings[step] for r in full if step in r.timings]
        st = _stats(vals)
        if not st:
            rows.append([step, "0", "-", "-", "-", "-", "-"])
            continue
        share = f"{100.0 * st['mean'] / total_mean:5.1f}%" if total_mean else "-"
        rows.append([
            step, str(int(st["n"])), f"{st['mean']:,.1f}", f"{st['median']:,.1f}",
            f"{st['min']:,.1f}", f"{st['max']:,.1f}", share,
        ])
    rows.append(["TOTAL", "", f"{total_mean:,.1f}", "", "", "", "100.0%"])
    out.append(_table(["step", "n", "mean", "median", "min", "max", "share"], rows))

    # --- generation tokens and prompt-cache reuse ---------------------------
    gen_full = [r for r in full if r.completion_tokens > 0]
    if gen_full:
        prompt_total = sum(r.prompt_tokens for r in gen_full) or 1
        cached_total = sum(r.cached_tokens for r in gen_full)
        eval_ms = sum(r.prompt_eval_ms for r in gen_full)
        decode_ms = sum(r.decode_ms for r in gen_full)
        completion_total = sum(r.completion_tokens for r in gen_full)
        prompt_tps = (prompt_total - cached_total) / (eval_ms / 1000.0) if eval_ms else 0.0
        decode_tps = completion_total / (decode_ms / 1000.0) if decode_ms else 0.0
        out.append("")
        out.append(f"GENERATION over {len(gen_full)} runs with token telemetry")
        out.append(
            f"  prompt/query={statistics.fmean([r.prompt_tokens for r in gen_full]):,.0f}  "
            f"completion/query={statistics.fmean([r.completion_tokens for r in gen_full]):,.0f}  "
            f"cache_hit={100.0 * cached_total / prompt_total:5.1f}%  "
            f"prompt_eval={prompt_tps:,.1f} tok/s  decode={decode_tps:,.1f} tok/s"
        )
    return "\n".join(out)


def _run_retrieval_once(
    q: BenchQuestion,
    iteration: int,
    *,
    embedder: Any,
    retrieve_node: RetrieveNode,
    rerank_node: RerankNode,
    context_node: ContextNode,
    principal: Principal,
) -> Run:
    timings: dict[str, float] = {}
    state: dict[str, Any] = {
        "principal": principal,
        "query": q.query,
        "rewritten_query": q.query,
        "filters": RetrievalFilters(),
    }

    started = perf_counter()
    try:
        vector = embedder.embed([q.query])[0]
    except Exception as exc:  # noqa: BLE001 - benchmark should report, not crash
        timings["encode"] = (perf_counter() - started) * 1000.0
        state["status"] = QueryStatus.ERROR
        state["warnings"] = [f"embed_failed: {exc}"]
        return _retrieval_run(q, iteration, embedder, state, timings)
    timings["encode"] = (perf_counter() - started) * 1000.0
    state["query_vector"] = vector

    started = perf_counter()
    state.update(retrieve_node.run(state))
    timings["retrieve"] = (perf_counter() - started) * 1000.0
    if state.get("status") in (QueryStatus.ERROR, QueryStatus.NO_CONTEXT):
        return _retrieval_run(q, iteration, embedder, state, timings)

    started = perf_counter()
    state.update(rerank_node.run(state))
    timings["rerank"] = (perf_counter() - started) * 1000.0
    if state.get("status") in (QueryStatus.ERROR, QueryStatus.NO_CONTEXT):
        return _retrieval_run(q, iteration, embedder, state, timings)

    started = perf_counter()
    state.update(context_node.run(state))
    timings["context"] = (perf_counter() - started) * 1000.0
    return _retrieval_run(q, iteration, embedder, state, timings)


def _retrieval_run(
    q: BenchQuestion,
    iteration: int,
    embedder: Any,
    state: dict[str, Any],
    timings: dict[str, float],
) -> Run:
    summary = state.get("retrieval_summary")
    bundle = state.get("context")
    status = state.get("status")
    return Run(
        qid=q.qid,
        bucket=q.bucket,
        note=q.note,
        iteration=iteration,
        chars=len(q.query.strip()),
        words=len(q.query.split()),
        status=status.value if isinstance(status, QueryStatus) else QueryStatus.ANSWERED.value,
        trace=["encode", "retrieve", "rerank", "context"],
        timings=timings,
        total_ms=float(sum(timings.values())),
        dim=int(embedder.dim),
        warnings=list(state.get("warnings") or []),
        dense=summary.dense_candidates if summary is not None else 0,
        lexical=summary.lexical_candidates if summary is not None else 0,
        reranked=summary.reranked if summary is not None else 0,
        sources=len(bundle.sources) if bundle is not None else 0,
        context_tokens=bundle.token_count if bundle is not None else 0,
    )


def _run_retrieval_only(
    settings: QuerySettings,
    questions: list[BenchQuestion],
    repeat: int,
) -> tuple[list[Run], dict[str, Any]]:
    print("building retrieval stack (BGE + reranker + tokenizer)...", file=sys.stderr)
    embedder = build_embedder(settings)
    retriever = build_retriever(settings, embedder)
    reranker = build_reranker(settings)
    if hasattr(reranker, "warmup"):
        reranker.warmup()  # type: ignore[attr-defined]
    company_directory = build_company_directory(settings)
    extractor = MetadataExtractor(company_directory, corpus_years=settings.corpus_years)
    counter = build_token_counter(settings.context_tokenizer_dir)
    counter.count("warmup")
    builder = build_context_builder(settings, counter)
    retrieve_node = RetrieveNode(
        retriever,
        filter_extractor=extractor,
        top_k=settings.retrieval_fused_k,
        per_company_k=settings.retrieval_per_company_k,
    )
    rerank_node = RerankNode(
        reranker,
        top_k=settings.rerank_top_k,
        expander=retriever,
        neighbor_window=settings.retrieval_neighbor_window,
    )
    context_node = ContextNode(builder, include_history=False)
    principal = build_authenticator(settings).authenticate("dev-public")

    print("warm-up...", file=sys.stderr)
    _run_retrieval_once(
        QUESTIONS[5],
        -1,
        embedder=embedder,
        retrieve_node=retrieve_node,
        rerank_node=rerank_node,
        context_node=context_node,
        principal=principal,
    )

    runs: list[Run] = []
    total = len(questions) * repeat
    done = 0
    for q in questions:
        for i in range(repeat):
            run = _run_retrieval_once(
                q,
                i,
                embedder=embedder,
                retrieve_node=retrieve_node,
                rerank_node=rerank_node,
                context_node=context_node,
                principal=principal,
            )
            runs.append(run)
            done += 1
            print(
                f"  [{done}/{total}] {q.qid:<16} status={run.status:<15} "
                f"total={run.total_ms:,.0f}ms",
                file=sys.stderr,
            )
    meta = {
        "embedding_model": embedder.fingerprint.model_id,
        "embedding_dim": embedder.dim,
        "reranker": str(settings.rerank_model_dir),
        "reranker_enabled": bool(getattr(reranker, "enabled", False)),
        "context_token_budget": settings.context_token_budget,
    }
    return runs, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repeat", type=int, default=1, help="runs per question (default 1)")
    parser.add_argument(
        "--no-guard", dest="guard", action="store_false", help="disable the prompt-guard classifier"
    )
    parser.set_defaults(guard=True)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="time embedding/retrieval/rerank/context only (no LLM or guards)",
    )
    parser.add_argument(
        "--no-index-check", action="store_true", help="skip the Postgres fingerprint query"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("bench_query_results.json"), help="JSON output path"
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of tables")
    parser.add_argument("--only", nargs="*", help="restrict to these question ids")
    args = parser.parse_args(argv)

    questions = [q for q in QUESTIONS if not args.only or q.qid in args.only]

    settings = QuerySettings(
        guard_enabled=args.guard and not args.retrieval_only,
        rate_limit_per_minute=1_000_000,
        rate_limit_burst=1_000_000,
        vector_check=not args.no_index_check,
    )

    if args.retrieval_only:
        runs, meta = _run_retrieval_only(settings, questions, max(1, args.repeat))
        payload = {
            "config": {
                "mode": "retrieval-only",
                "guard_enabled": False,
                "index_check": not args.no_index_check,
                "repeat": args.repeat,
                **meta,
            },
            "runs": [run.__dict__ for run in runs],
        }
    else:
        print(
            f"building service (guard={'on' if args.guard else 'off'}, "
            f"index_check={'off' if args.no_index_check else 'on'})...",
            file=sys.stderr,
        )
        service = build_service(settings)
        meta = service.metadata

        # Warm-up: first embed/LLM/guard/reranker call pays lazy CUDA + kernel
        # init and PG2 lazily loads its weights. Do not time it.
        print("warm-up...", file=sys.stderr)
        _run_once(service, QUESTIONS[5], iteration=-1)

        runs = []
        total = len(questions) * args.repeat
        done = 0
        for q in questions:
            for i in range(args.repeat):
                run = _run_once(service, q, i)
                runs.append(run)
                done += 1
                print(
                    f"  [{done}/{total}] {q.qid:<16} status={run.status:<15} "
                    f"total={run.total_ms:,.0f}ms",
                    file=sys.stderr,
                )
        payload = {
            "config": {
                "mode": "full",
                "guard_enabled": args.guard,
                "index_check": not args.no_index_check,
                "repeat": args.repeat,
                "embedding_model": meta.get("embedding_model"),
                "embedding_dim": meta.get("embedding_dim"),
                "llm_model": meta.get("llm_model"),
                "guards": meta.get("guards"),
                "reranker": meta.get("reranker"),
                "rerank_enabled": meta.get("rerank_enabled"),
            },
            "runs": [run.__dict__ for run in runs],
        }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_report(runs, questions))
        print("")
        print(
            f"mode={'retrieval-only' if args.retrieval_only else 'full'}  "
            f"guard={'off' if args.retrieval_only or not args.guard else 'on'}  "
            f"index_check={not args.no_index_check}  repeat={args.repeat}  "
            f"embed={meta.get('embedding_model')}  "
            f"llm={meta.get('llm_model') or '-'}"
        )

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} ({len(runs)} runs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
