"""BGE-large-en-v1.5 embedder.

Loads the local sentence-transformers copy, runs batched fp16 inference on the
GPU and returns L2-normalised float32 vectors. The model fingerprint (dim,
normalisation, max length, pooling) is recorded so the query side can prove it
used the identical contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rag_ingest.domain.models import ModelFingerprint
from rag_ingest.logging_setup import get_logger

log = get_logger(__name__)

# BGE v1.5 passages need no prefix. Kept explicit for the future query side.
DEFAULT_QUERY_PREFIX = ""


class BgeEmbedder:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cuda",
        batch_size: int = 32,
        max_seq_length: int = 512,
        use_fp16: bool = True,
        model_id: str = "BAAI/bge-large-en-v1.5",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(str(model_dir), device=device)
        self._model.max_seq_length = max_seq_length
        self._use_fp16 = use_fp16 and device != "cpu"
        if self._use_fp16:
            self._model.half()
        self._batch_size = batch_size

        dim_getter = getattr(self._model, "get_embedding_dimension", None)
        if dim_getter is None:  # older sentence-transformers
            dim_getter = self._model.get_sentence_embedding_dimension
        dim = dim_getter()
        if not dim:
            raise RuntimeError("could not determine embedding dimension from model")
        self._dim = int(dim)
        self._fingerprint = ModelFingerprint(
            model_id=model_id,
            revision=None,
            dim=self._dim,
            normalize=True,
            max_seq_length=max_seq_length,
            pooling="cls",
            query_prefix=DEFAULT_QUERY_PREFIX,
        )
        log.info(
            "loaded embedder %s dim=%d device=%s fp16=%s",
            model_id,
            self._dim,
            device,
            self._use_fp16,
        )

    @property
    def fingerprint(self) -> ModelFingerprint:
        return self._fingerprint

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class BgeQueryEmbedder:
    """Query-side embedder: applies the documented instruction prefix.

    Provided now so the eventual question pipeline cannot silently drift from
    ingestion. Ingestion must use :class:`BgeEmbedder` (passages, no prefix).
    """

    def __init__(self, embedder: BgeEmbedder, prefix: str = DEFAULT_QUERY_PREFIX) -> None:
        self._embedder = embedder
        self._prefix = prefix

    @property
    def fingerprint(self) -> ModelFingerprint:
        return self._embedder.fingerprint

    @property
    def dim(self) -> int:
        return self._embedder.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._embedder.embed([self._prefix + t for t in texts])
