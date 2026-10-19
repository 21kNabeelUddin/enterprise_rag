"""Typed configuration for the query pipeline.

Shares the ``RAG_`` environment prefix and ``.env`` file with the ingestion
package, so one file configures the whole system. Complex values (the dev token
map) can be overridden with JSON via the environment.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_query.domain.models import Principal

# src/rag_query/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dev_tokens() -> dict[str, Principal]:
    """Stand-in SSO: a fixed map of dev tokens to principals.

    Replace with a real identity provider by implementing the
    :class:`~rag_query.ports.auth.Authenticator` port; nothing else changes.
    """
    return {
        "dev-public": Principal(
            user_id="public-dev",
            groups=("public",),
            display_name="Public dev user",
            token_label="dev-public",
        ),
        "dev-finance": Principal(
            user_id="finance-dev",
            groups=("public", "finance"),
            display_name="Finance dev user",
            token_label="dev-finance",
        ),
        "dev-admin": Principal(
            user_id="admin-dev",
            groups=("public", "finance", "admin"),
            display_name="Admin dev user",
            token_label="dev-admin",
        ),
        "dev-guest": Principal(
            user_id="guest-dev",
            groups=(),
            display_name="Guest (auth ok, no groups)",
            token_label="dev-guest",
        ),
    }


def _default_anonymous() -> Principal:
    # EDGAR is public data, so an unauthenticated demo caller may read it.
    return Principal(
        user_id="anonymous",
        groups=("public",),
        display_name="Anonymous",
    )


def _default_rerank_dir() -> Path:
    return PROJECT_ROOT / "models" / "Qwen3-Reranker-0.6B"


def _default_guard_dir() -> Path:
    # Verified Prompt Guard 2 snapshot (binary BENIGN/MALICIOUS). The older
    # models/llama-prompt-guard-2-86m directory holds Prompt Guard 1 weights.
    return PROJECT_ROOT / "models" / "llama-prompt-guard-2"


class QuerySettings(BaseSettings):
    """Runtime configuration for Pipeline 2. Immutable by convention."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths ---
    project_root: Path = PROJECT_ROOT
    model_dir: Path = PROJECT_ROOT / "models" / "bge-large-en-v1.5"
    guard_model_dir: Path = Field(default_factory=_default_guard_dir)
    rerank_model_dir: Path = Field(default_factory=_default_rerank_dir)
    context_tokenizer_dir: Path = Field(default_factory=_default_rerank_dir)
    company_map_path: Path = PROJECT_ROOT / "data" / "company_tickers.json"

    # --- encoder (must match ingestion) ---
    vector_dim: int = 1024
    embed_device: str = "cuda"
    embed_batch_size: int = 32
    embed_use_fp16: bool = True
    model_max_seq_length: int = 512
    # Verify the stored index uses this encoder. The check runs once at startup
    # (cached for ``vector_probe_ttl_s``), not on every request.
    vector_check: bool = True
    vector_check_at_startup: bool = True
    vector_probe_ttl_s: float = 300.0

    # --- step 1: input ---
    max_query_chars: int = 4000

    # --- step 3: input guardrail ---
    # Prompt Guard 2 verified on-device: benign queries ~0.0004 malicious,
    # injection/jailbreak prompts >= 0.99. Runs on CPU to leave VRAM for the LLM.
    guard_enabled: bool = True
    guard_device: str = "cpu"  # 86M model on CPU keeps the GPU for LLM + BGE
    guard_threshold: float = 0.5
    guard_block_pii: bool = False  # detect+record by default, do not block
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 20

    # --- step 4: router ---
    min_query_tokens: int = 3

    # --- step 2: authenticate ---
    dev_tokens: dict[str, Principal] = Field(default_factory=_default_dev_tokens)
    anonymous: Principal = Field(default_factory=_default_anonymous)

    # --- step 5: rewriter (local llama.cpp, OpenAI-compatible) ---
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen3.5-9b-local"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    llm_timeout_s: float = 180.0
    llm_enable_thinking: bool = False
    # llama.cpp prefix/KV cache. On by default; exposed so cache-on/off can be
    # benchmarked without editing code. A hit is reported in Generation.
    llm_cache_prompt: bool = True
    rewrite_enabled: bool = True
    rewrite_max_tokens: int = 96
    rewrite_max_chars: int = 400
    rewrite_history_turns: int = 4

    # --- step 8: hybrid retrieval ---
    retrieval_enabled: bool = True
    retrieval_dense_k: int = 50
    retrieval_lexical_k: int = 50
    retrieval_fused_k: int = 40
    # Comparison questions are retrieved per company; this is the pool size
    # per company before the cross-encoder re-ranks the merged set.
    retrieval_per_company_k: int = 8
    retrieval_rrf_k: int = 60
    retrieval_dense_weight: float = 1.0
    retrieval_lexical_weight: float = 0.8
    retrieval_lexical_min_hits: int = 8
    retrieval_lexical_max_terms: int = 8
    retrieval_statement_timeout_ms: int = 15_000
    retrieval_hnsw_ef_search: int = 100
    retrieval_hnsw_iterative_scan: bool = True
    # Add adjacent chunks from the same section after reranking. 0 disables.
    retrieval_neighbor_window: int = 1
    corpus_years: tuple[str, ...] = ("2020",)

    # --- step 9: cross-encoder rerank ---
    # Warm GPU inference is ~0.15 s for 8 candidates; CPU is ~9 s/candidate.
    # If CUDA is unavailable or the reranker OOMs next to the LLM, the pipeline
    # keeps retrieval order instead of falling back to CPU.
    rerank_enabled: bool = True
    rerank_device: str = "cuda"
    rerank_batch_size: int = 8
    rerank_max_length: int = 512
    rerank_top_k: int = 8
    rerank_allow_cpu_fallback: bool = False

    # --- step 10: context builder ---
    context_token_budget: int = 6500
    context_max_sources: int = 12
    context_max_history_messages: int = 6
    # History is used by the rewriter. Including previous assistant answers in
    # the generation prompt makes small local models re-answer the previous
    # question; keep it off by default and rely on a standalone rewrite.
    context_include_history: bool = False
    context_chars_per_token: float = 4.0
    context_min_source_tokens: int = 60
    context_history_budget_ratio: float = 0.2

    # --- step 11: grounded generation ---
    generation_temperature: float = 0.1
    generation_max_tokens: int = 768
    generation_max_retries: int = 1
    generation_retry_sleep_s: float = 0.5

    # --- step 12: output guard ---
    output_guard_enabled: bool = True
    output_guard_min_citations: int = 1
    output_guard_grounding_threshold: float = 0.45
    output_guard_grounding_min_ratio: float = 0.6
    output_guard_max_retries: int = 1
    output_guard_pii_block: bool = True

    # --- step 14: history and feedback ---
    history_enabled: bool = True
    history_max_turns: int = 6
    history_persist_failures: bool = True
    feedback_admin_group: str = "admin"

    # --- storage ---
    postgres_dsn: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    db_connect_timeout_s: int = 5

    # --- misc ---
    log_level: str = "INFO"


def get_query_settings(**overrides: object) -> QuerySettings:
    """Load query settings, optionally overriding fields (handy in tests)."""
    return QuerySettings(**overrides)
