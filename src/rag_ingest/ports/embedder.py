"""Embedder port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from rag_ingest.domain.models import ModelFingerprint


@runtime_checkable
class Embedder(Protocol):
    """Turns chunk text into vectors.

    ``embed`` returns an ``(n, dim)`` float32 array in the same order as the
    input. Implementations decide batching/device internally.
    """

    @property
    def fingerprint(self) -> ModelFingerprint: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...
