"""Typed application configuration.

Every tunable lives here so stages never hard-code constants. Values come from
environment variables (``RAG_`` prefix) or a local ``.env`` file, falling back
to the defaults below.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/rag_ingest/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# EDGAR-CORPUS ships split files named train/validate/test, while HuggingFace
# configs call them train/validation/test. Keep one canonical mapping.
SPLIT_TO_FILE = {
    "train": "train",
    "validation": "validate",
    "test": "test",
}


class Settings(BaseSettings):
    """Runtime configuration. Immutable by convention."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths ---
    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = PROJECT_ROOT / "dataset"
    model_dir: Path = PROJECT_ROOT / "models" / "bge-large-en-v1.5"

    # --- source ---
    edgar_repo_id: str = "eloukas/edgar-corpus"
    edgar_year: str = "2020"

    # --- chunking (BGE-large-en-v1.5 caps at 512 tokens) ---
    chunk_target_tokens: int = 384
    chunk_overlap_tokens: int = 48
    chunk_max_tokens: int = 500

    # --- embedding ---
    embed_device: str = "cuda"
    embed_batch_size: int = 32
    embed_use_fp16: bool = True
    model_max_seq_length: int = 512

    # --- storage ---
    postgres_dsn: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    # --- misc ---
    log_level: str = "INFO"

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _overlap_lt_target(cls, v: int, info) -> int:
        target = info.data.get("chunk_target_tokens", 384)
        if v >= target:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_target_tokens")
        return v

    @field_validator("chunk_max_tokens")
    @classmethod
    def _max_ge_target(cls, v: int, info) -> int:
        target = info.data.get("chunk_target_tokens", 384)
        if v < target:
            raise ValueError("chunk_max_tokens must be >= chunk_target_tokens")
        return v

    def dataset_file(self, split: str) -> Path:
        """Locate a split file.

        Accepts both layouts: ``dataset/test.jsonl`` (flat, as shipped to the
        user) and ``dataset/2020/test.jsonl`` (as ``hf_hub_download`` writes it).
        """
        filename = SPLIT_TO_FILE.get(split, split)
        flat = self.dataset_dir / f"{filename}.jsonl"
        if flat.exists():
            return flat
        return self.dataset_dir / self.edgar_year / f"{filename}.jsonl"


def get_settings(**overrides: object) -> Settings:
    """Load settings, optionally overriding fields (handy in tests)."""
    return Settings(**overrides)
