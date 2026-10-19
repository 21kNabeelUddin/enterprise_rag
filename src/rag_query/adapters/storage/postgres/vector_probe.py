"""Reads the model fingerprint the index was actually built with.

The encoder must produce vectors from the *same* model as the stored vectors.
Comparing a BGE-large query vector against vectors from another model silently
returns garbage, so we fail loudly instead.

The probe is cached for a short TTL. The full-table aggregate used to run on
every request (~0.5 s); now it runs once at startup and then at most once per
``ttl_s`` seconds, while step 8's own model/dim filter protects every query.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from sqlalchemy import Engine, text

from rag_query.domain.models import StoredVectorInfo
from rag_query.logging import get_logger

log = get_logger(__name__)


class StoredVectorProbe:
    def __init__(
        self,
        engine: Engine,
        *,
        ttl_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._ttl_s = max(0.0, ttl_s)
        self._clock = clock
        self._cached: StoredVectorInfo | None = None
        self._cached_at = 0.0
        self._lock = threading.Lock()

    def describe(self, *, force: bool = False) -> StoredVectorInfo:
        now = self._clock()
        if not force and self._cached is not None and now - self._cached_at < self._ttl_s:
            return self._cached

        with self._lock:
            now = self._clock()
            if not force and self._cached is not None and now - self._cached_at < self._ttl_s:
                return self._cached
            info = self._read()
            self._cached = info
            self._cached_at = now
            return info

    def invalidate(self) -> None:
        with self._lock:
            self._cached = None
            self._cached_at = 0.0

    def _read(self) -> StoredVectorInfo:
        stmt = text(
            "SELECT count(*) AS n, "
            "array_agg(DISTINCT model_id) AS model_ids, "
            "array_agg(DISTINCT dim) AS dims "
            "FROM embeddings"
        )
        with self._engine.connect() as conn:
            row = conn.execute(stmt).one()
        info = StoredVectorInfo(
            count=int(row.n or 0),
            model_ids=tuple(row.model_ids or ()),
            dims=tuple(int(dim) for dim in (row.dims or ())),
        )
        log.debug("index vectors: %s", info)
        return info
