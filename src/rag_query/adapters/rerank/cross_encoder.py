"""Qwen3 cross-encoder reranker.

``Qwen3-Reranker-0.6B`` is a causal LM with a sentence-transformers
``CrossEncoder`` head (logit score; higher is better). It is loaded lazily and
degrades to a no-op reranker on missing weights, OOM or inference failure, so a
model problem cannot take the query path down.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_query.adapters.rerank.noop import NoOpReranker
from rag_query.domain.models import RetrievedChunk
from rag_query.logging import get_logger

log = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 512,
        fallback_on_error: bool = True,
        allow_cpu_fallback: bool = False,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._device = device
        self._batch_size = max(1, batch_size)
        self._max_length = max(64, max_length)
        self._fallback_on_error = fallback_on_error
        self._allow_cpu_fallback = allow_cpu_fallback
        self._model: Any | None = None
        self._disabled = False
        self._fallback = NoOpReranker(reason="cross-encoder unavailable")

    @property
    def enabled(self) -> bool:
        return not self._disabled

    def warmup(self) -> bool:
        """Load and exercise the model once, outside the request path.

        The first Qwen3-Reranker call is dominated by CUDA kernel setup (~17 s
        on the laptop GPU); warm inference is ~0.15 s for 8 candidates. If the
        GPU cannot fit the reranker next to the LLM, disable reranking rather
        than silently falling back to a multi-second CPU path.
        """
        if self._disabled:
            return False
        started = time.perf_counter()
        try:
            model = self._ensure_model()
            # Exercise a full batch, not just one pair: the first batch of 8
            # pays kernel-selection and allocator costs that a single-pair
            # warmup would leave on the first user request.
            pairs = [("warmup query", "warmup passage")] * self._batch_size
            model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            log.info(
                "reranker warmup ok in %.2fs (batch=%d, device=%s)",
                time.perf_counter() - started,
                self._batch_size,
                self._device,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                _empty_cuda_cache()
            log.warning("reranker warmup failed; disabling cross-encoder: %s", exc)
            self._disabled = True
            self._model = None
            return False

    @property
    def model_name(self) -> str:
        return self._model_dir.name

    @property
    def device(self) -> str:
        return self._device

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        candidates = list(candidates)
        if not candidates:
            return []
        limit = top_k or len(candidates)
        if self._disabled:
            return self._fallback.rerank(query, candidates, top_k=limit)

        try:
            model = self._ensure_model()
            pairs = [(query, chunk.text) for chunk in candidates]
            scores = model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            ranked = sorted(
                zip(candidates, scores, strict=True),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )[:limit]
            return [
                chunk.model_copy(update={"rerank_score": float(score), "rank": index + 1})
                for index, (chunk, score) in enumerate(ranked)
            ]
        except Exception as exc:  # noqa: BLE001
            oom = "out of memory" in str(exc).lower()
            if oom and self._allow_cpu_fallback and self._device != "cpu":
                log.warning("reranker CUDA OOM; retrying on CPU as configured")
                return self._rerank_on_cpu(query, candidates, limit)
            if oom:
                # A CPU cross-encoder takes seconds per batch. Retrieval order
                # is a better product than a 30-second request.
                self._disabled = True
                self._model = None
                _empty_cuda_cache()
                log.warning(
                    "reranker CUDA OOM; disabled for this process "
                    "(free GPU memory for the LLM and restart to re-enable)"
                )
                return self._fallback.rerank(query, candidates, top_k=limit)
            if not self._fallback_on_error:
                raise
            log.exception("reranker failed; falling back to retrieval order")
            return self._fallback.rerank(query, candidates, top_k=limit)

    def _rerank_on_cpu(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        try:
            self._device = "cpu"
            self._model = None
            _empty_cuda_cache()
            return self.rerank(query, candidates, top_k=limit)
        except Exception:  # noqa: BLE001
            self._disabled = True
            log.exception("reranker CPU fallback failed; using retrieval order")
            return self._fallback.rerank(query, candidates, top_k=limit)

    def _ensure_model(self):  # noqa: ANN202
        if self._model is not None:
            return self._model
        if not self._model_dir.exists():
            raise FileNotFoundError(f"reranker model not found at {self._model_dir}")
        from sentence_transformers import CrossEncoder

        log.info(
            "loading reranker %s device=%s max_length=%d",
            self._model_dir,
            self._device,
            self._max_length,
        )
        self._model = CrossEncoder(
            str(self._model_dir),
            device=self._device,
            max_length=self._max_length,
        )
        return self._model


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort cleanup
        pass
