"""Composition root for the query pipeline.

The only module that binds ports to concrete adapters. Everything else receives
its collaborators, which is what keeps the pipeline unit-testable and lets the
local model server or Postgres be swapped without touching a node.

The full service wires steps 1-14. Passing ``with_retrieval=False`` builds the
legacy steps 1-7 graph, which the early tests and diagnostics still use.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import Engine

from rag_ingest.adapters.embedders.bge import BgeEmbedder, BgeQueryEmbedder
from rag_ingest.adapters.storage.postgres.engine import build_engine
from rag_ingest.ports.embedder import Embedder
from rag_query.adapters.auth.dev_tokens import DevTokenAuthenticator
from rag_query.adapters.generation.llm_generator import LlmAnswerGenerator
from rag_query.adapters.guards.output import (
    CitationOutputGuard,
    CompositeOutputGuard,
    GroundingOutputGuard,
    LeakageOutputGuard,
    PiiOutputGuard,
)
from rag_query.adapters.guards.pii import PiiQueryGuard
from rag_query.adapters.guards.prompt_guard import PromptGuardModel, PromptGuardQueryGuard
from rag_query.adapters.llm.openai_compat import OpenAICompatibleLLM
from rag_query.adapters.metadata.company_directory import load_company_directory
from rag_query.adapters.rate_limit.memory import InMemoryRateLimiter
from rag_query.adapters.rerank.cross_encoder import CrossEncoderReranker
from rag_query.adapters.rerank.noop import NoOpReranker
from rag_query.adapters.rewriter.llm_rewriter import LlmQueryRewriter
from rag_query.adapters.router.rules import RuleBasedRouter
from rag_query.adapters.storage.postgres.history import PostgresHistoryStore
from rag_query.adapters.storage.postgres.retriever import PostgresHybridRetriever
from rag_query.adapters.storage.postgres.vector_probe import StoredVectorProbe
from rag_query.adapters.tokenization.counter import build_token_counter
from rag_query.config import QuerySettings
from rag_query.domain.context import ContextBuilder
from rag_query.domain.models import StoredVectorInfo
from rag_query.domain.query_analysis import CompanyDirectory, MetadataExtractor
from rag_query.graph import build_graph
from rag_query.logging import get_logger
from rag_query.nodes import (
    AuthenticateNode,
    ContextNode,
    EncodeNode,
    FallbackNode,
    FinalizeNode,
    FormatAnswerNode,
    GenerateNode,
    HistoryNode,
    InputGuardNode,
    InputNode,
    OutputGuardNode,
    RerankNode,
    RetrieveNode,
    RewriteNode,
    RouteNode,
)
from rag_query.ports.generator import Generator
from rag_query.ports.guard import QueryGuard
from rag_query.ports.history import HistoryStore
from rag_query.ports.llm import LLMClient
from rag_query.ports.output_guard import OutputGuard
from rag_query.ports.reranker import Reranker
from rag_query.ports.retriever import Retriever
from rag_query.ports.token_counter import TokenCounter
from rag_query.service import QueryService

log = get_logger(__name__)


def build_settings(**overrides: object) -> QuerySettings:
    return QuerySettings(**overrides)


def build_embedder(settings: QuerySettings) -> BgeQueryEmbedder:
    base = BgeEmbedder(
        settings.model_dir,
        device=settings.embed_device,
        batch_size=settings.embed_batch_size,
        max_seq_length=settings.model_max_seq_length,
        use_fp16=settings.embed_use_fp16,
    )
    return BgeQueryEmbedder(base)


def build_authenticator(settings: QuerySettings) -> DevTokenAuthenticator:
    return DevTokenAuthenticator(settings.dev_tokens, anonymous=settings.anonymous)


def build_guards(settings: QuerySettings) -> list[QueryGuard]:
    guards: list[QueryGuard] = []

    if settings.guard_enabled:
        model_dir = Path(settings.guard_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"prompt-guard model not found at {model_dir}. Point RAG_GUARD_MODEL_DIR "
                "at a local copy of Llama-Prompt-Guard-2-86M, or set RAG_GUARD_ENABLED=false."
            )
        model = PromptGuardModel(model_dir, device=settings.guard_device)
        guards.append(PromptGuardQueryGuard(model, threshold=settings.guard_threshold))

    guards.append(PiiQueryGuard(block=settings.guard_block_pii))
    return guards


def build_rate_limiter(settings: QuerySettings) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(
        per_minute=settings.rate_limit_per_minute,
        burst=settings.rate_limit_burst,
    )


def build_router(settings: QuerySettings) -> RuleBasedRouter:
    return RuleBasedRouter(min_tokens=settings.min_query_tokens)


def build_llm(settings: QuerySettings) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
        enable_thinking=settings.llm_enable_thinking,
        cache_prompt=settings.llm_cache_prompt,
    )


def build_rewriter(settings: QuerySettings, llm: LLMClient) -> LlmQueryRewriter:
    return LlmQueryRewriter(
        llm,
        max_history_turns=settings.rewrite_history_turns,
        max_tokens=settings.rewrite_max_tokens,
        max_chars=settings.rewrite_max_chars,
    )


def build_vector_probe(
    settings: QuerySettings,
    engine: Engine | None = None,
) -> StoredVectorProbe:
    """Read-only probe over the ingestion index, cached for a short TTL."""
    engine = engine or build_engine(
        settings.postgres_dsn,
        connect_timeout=settings.db_connect_timeout_s,
    )
    return StoredVectorProbe(engine, ttl_s=settings.vector_probe_ttl_s)


def build_retriever(
    settings: QuerySettings,
    embedder: Embedder,
    *,
    engine: Engine | None = None,
) -> PostgresHybridRetriever:
    engine = engine or build_engine(
        settings.postgres_dsn,
        connect_timeout=settings.db_connect_timeout_s,
    )
    return PostgresHybridRetriever(
        engine,
        model_id=embedder.fingerprint.model_id,
        dim=embedder.dim,
        dense_k=settings.retrieval_dense_k,
        lexical_k=settings.retrieval_lexical_k,
        fused_k=settings.retrieval_fused_k,
        rrf_k=settings.retrieval_rrf_k,
        dense_weight=settings.retrieval_dense_weight,
        lexical_weight=settings.retrieval_lexical_weight,
        lexical_min_hits=settings.retrieval_lexical_min_hits,
        lexical_max_terms=settings.retrieval_lexical_max_terms,
        statement_timeout_ms=settings.retrieval_statement_timeout_ms,
        hnsw_ef_search=settings.retrieval_hnsw_ef_search,
        hnsw_iterative_scan=settings.retrieval_hnsw_iterative_scan,
    )


def build_reranker(settings: QuerySettings) -> Reranker:
    if not settings.rerank_enabled:
        return NoOpReranker(reason="rerank disabled by settings")
    model_dir = Path(settings.rerank_model_dir)
    if not model_dir.exists():
        log.warning(
            "reranker model %s not found; running without cross-encoder reranking",
            model_dir,
        )
        return NoOpReranker(reason=f"model missing: {model_dir}")
    if settings.rerank_device.startswith("cuda") and not _cuda_available():
        log.warning(
            "rerank_device=%s but CUDA is unavailable; keeping retrieval order "
            "(set RAG_RERANK_DEVICE=cpu to force the slow CPU path)",
            settings.rerank_device,
        )
        return NoOpReranker(reason="CUDA unavailable")
    return CrossEncoderReranker(
        model_dir,
        device=settings.rerank_device,
        batch_size=settings.rerank_batch_size,
        max_length=settings.rerank_max_length,
        fallback_on_error=True,
        allow_cpu_fallback=settings.rerank_allow_cpu_fallback,
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch is optional for tests/CPU installs
        return False


def build_generator(settings: QuerySettings, llm: LLMClient) -> LlmAnswerGenerator:
    return LlmAnswerGenerator(
        llm,
        model=settings.llm_model,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
        max_retries=settings.generation_max_retries,
        retry_sleep_s=settings.generation_retry_sleep_s,
    )


def build_output_guard(
    settings: QuerySettings,
    embedder: Embedder,
) -> OutputGuard:
    if not settings.output_guard_enabled:
        return CompositeOutputGuard([])
    guards: list[object] = [
        CitationOutputGuard(min_citations=settings.output_guard_min_citations),
        GroundingOutputGuard(
            embedder,
            threshold=settings.output_guard_grounding_threshold,
            min_grounded_ratio=settings.output_guard_grounding_min_ratio,
        ),
        LeakageOutputGuard(),
    ]
    if settings.output_guard_pii_block:
        guards.append(PiiOutputGuard())
    return CompositeOutputGuard(guards)


def build_history_store(
    settings: QuerySettings,
    *,
    engine: Engine | None = None,
) -> HistoryStore | None:
    if not settings.history_enabled:
        return None
    engine = engine or build_engine(
        settings.postgres_dsn,
        connect_timeout=settings.db_connect_timeout_s,
    )
    return PostgresHistoryStore(engine)


def build_company_directory(settings: QuerySettings) -> CompanyDirectory:
    return load_company_directory(settings.company_map_path)


def build_context_builder(
    settings: QuerySettings,
    counter: TokenCounter,
) -> ContextBuilder:
    return ContextBuilder(
        counter,
        token_budget=settings.context_token_budget,
        max_sources=settings.context_max_sources,
        max_history_messages=settings.context_max_history_messages,
        chars_per_token=settings.context_chars_per_token,
        min_source_tokens=settings.context_min_source_tokens,
        history_budget_ratio=settings.context_history_budget_ratio,
    )


def _validate_index(info: StoredVectorInfo, embedder: Embedder) -> None:
    if not info.count:
        log.warning("vector index is empty; retrieval will return no context")
        return
    fingerprint = embedder.fingerprint
    if fingerprint.model_id not in info.model_ids or embedder.dim not in info.dims:
        raise RuntimeError(
            f"Encoder {fingerprint.model_id!r} (dim {embedder.dim}) does not match the "
            f"indexed vectors {list(info.model_ids)} (dims {list(info.dims)})."
        )


def _check_index_at_startup(probe: Any, embedder: Embedder) -> None:
    try:
        try:
            info = probe.describe(force=True)
        except TypeError:
            info = probe.describe()
    except Exception as exc:  # noqa: BLE001 - DB down is surfaced by retrieval
        log.warning("index fingerprint check could not run at startup: %s", exc)
        return
    _validate_index(info, embedder)
    log.info(
        "startup index check ok: %s rows, models=%s dims=%s",
        info.count,
        list(info.model_ids),
        list(info.dims),
    )


def build_service(
    settings: QuerySettings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    guards: Sequence[QueryGuard] | None = None,
    vector_probe: StoredVectorProbe | None = None,
    with_vector_check: bool | None = None,
    with_retrieval: bool | None = None,
    engine: Engine | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    generator: Generator | None = None,
    output_guard: OutputGuard | None = None,
    history_store: HistoryStore | None = None,
    token_counter: TokenCounter | None = None,
    company_directory: CompanyDirectory | None = None,
    authenticator: DevTokenAuthenticator | None = None,
) -> QueryService:
    """Build the whole pipeline with real adapters.

    Every collaborator can be injected to run the service against fakes. The
    full pipeline is selected by ``retrieval_enabled`` (default true); pass
    ``with_retrieval=False`` to build the legacy steps 1-7 graph.
    """
    settings = settings or build_settings()
    embedder = embedder or build_embedder(settings)
    llm = llm or build_llm(settings)
    guards = list(guards) if guards is not None else build_guards(settings)
    authenticator = authenticator or build_authenticator(settings)

    full_pipeline = settings.retrieval_enabled if with_retrieval is None else with_retrieval
    check_index = settings.vector_check if with_vector_check is None else with_vector_check

    if not full_pipeline:
        if vector_probe is None and check_index:
            vector_probe = build_vector_probe(settings, engine)
        graph = build_graph(
            input_node=InputNode(max_chars=settings.max_query_chars),
            authenticate_node=AuthenticateNode(authenticator),
            guard_node=InputGuardNode(guards, rate_limiter=build_rate_limiter(settings)),
            route_node=RouteNode(build_router(settings)),
            rewrite_node=RewriteNode(
                build_rewriter(settings, llm),
                enabled=settings.rewrite_enabled,
            ),
            encode_node=EncodeNode(
                embedder,
                expected_dim=settings.vector_dim,
                vector_probe=vector_probe,
                check_index=check_index,
            ),
            fallback_node=FallbackNode(),
        )
        metadata: dict[str, Any] = {
            "pipeline": "steps-1-7",
            "embedding_model": embedder.fingerprint.model_id,
            "embedding_dim": embedder.dim,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "guards": [guard.name for guard in guards],
            "rewrite_enabled": settings.rewrite_enabled,
            "index_checked": vector_probe is not None,
        }
        return QueryService(
            graph,
            settings=settings,
            metadata=metadata,
            authenticator=authenticator,
            history_store=history_store,
        )

    engine = engine or build_engine(
        settings.postgres_dsn,
        connect_timeout=settings.db_connect_timeout_s,
    )
    if vector_probe is None and check_index:
        vector_probe = build_vector_probe(settings, engine)
    if check_index and settings.vector_check_at_startup and vector_probe is not None:
        _check_index_at_startup(vector_probe, embedder)

    retriever = retriever or build_retriever(settings, embedder, engine=engine)
    reranker = reranker or build_reranker(settings)
    if hasattr(reranker, "warmup"):
        try:
            warmed = bool(reranker.warmup())  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - startup must still succeed
            log.warning("reranker warmup raised: %s", exc)
            warmed = False
        if not warmed:
            log.warning("cross-encoder reranking unavailable; keeping retrieval order")
    generator = generator or build_generator(settings, llm)
    output_guard = output_guard or build_output_guard(settings, embedder)
    token_counter = token_counter or build_token_counter(settings.context_tokenizer_dir)
    try:
        # Loading the tokenizer costs ~1 s; pay it at startup, not inside the
        # first context-build step.
        token_counter.count("warmup")
    except Exception as exc:  # noqa: BLE001 - the builder has a character fallback
        log.warning("token counter warmup failed: %s", exc)
    company_directory = company_directory or build_company_directory(settings)
    if history_store is None and settings.history_enabled:
        history_store = build_history_store(settings, engine=engine)

    filter_extractor = MetadataExtractor(
        company_directory,
        corpus_years=settings.corpus_years,
    )
    context_builder = build_context_builder(settings, token_counter)

    graph = build_graph(
        input_node=InputNode(max_chars=settings.max_query_chars),
        authenticate_node=AuthenticateNode(authenticator),
        history_node=HistoryNode(history_store, max_turns=settings.history_max_turns),
        guard_node=InputGuardNode(guards, rate_limiter=build_rate_limiter(settings)),
        route_node=RouteNode(build_router(settings)),
        rewrite_node=RewriteNode(build_rewriter(settings, llm), enabled=settings.rewrite_enabled),
        encode_node=EncodeNode(
            embedder,
            expected_dim=settings.vector_dim,
            vector_probe=None,
            check_index=False,
        ),
        fallback_node=FallbackNode(),
        retrieve_node=RetrieveNode(
            retriever,
            filter_extractor=filter_extractor,
            top_k=settings.retrieval_fused_k,
            per_company_k=settings.retrieval_per_company_k,
        ),
        rerank_node=RerankNode(
            reranker,
            top_k=settings.rerank_top_k,
            expander=retriever,
            neighbor_window=settings.retrieval_neighbor_window,
        ),
        context_node=ContextNode(
            context_builder,
            include_history=settings.context_include_history,
        ),
        generate_node=GenerateNode(generator),
        output_guard_node=OutputGuardNode(
            output_guard,
            max_retries=settings.output_guard_max_retries,
        ),
        format_node=FormatAnswerNode(company_directory),
        finalize_node=FinalizeNode(
            history_store,
            persist_failures=settings.history_persist_failures,
        ),
    )

    metadata = {
        "pipeline": "steps-1-14",
        "embedding_model": embedder.fingerprint.model_id,
        "embedding_dim": embedder.dim,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "guards": [guard.name for guard in guards],
        "rewrite_enabled": settings.rewrite_enabled,
        "index_checked": vector_probe is not None,
        "retrieval": {
            "dense_k": settings.retrieval_dense_k,
            "lexical_k": settings.retrieval_lexical_k,
            "fused_k": settings.retrieval_fused_k,
            "rrf_k": settings.retrieval_rrf_k,
            "top_k": settings.rerank_top_k,
            "neighbor_window": settings.retrieval_neighbor_window,
        },
        "reranker": getattr(reranker, "model_name", type(reranker).__name__),
        "rerank_device": getattr(reranker, "device", settings.rerank_device),
        "rerank_enabled": bool(getattr(reranker, "enabled", settings.rerank_enabled)),
        "context_token_budget": settings.context_token_budget,
        "context_tokenizer": getattr(token_counter, "name", type(token_counter).__name__),
        "output_guard_enabled": settings.output_guard_enabled,
        "history_enabled": history_store is not None,
        "company_filter_enabled": not company_directory.empty,
    }
    return QueryService(
        graph,
        settings=settings,
        metadata=metadata,
        authenticator=authenticator,
        history_store=history_store,
        feedback_admin_group=settings.feedback_admin_group,
    )
