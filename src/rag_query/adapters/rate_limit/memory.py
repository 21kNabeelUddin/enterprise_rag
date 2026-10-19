"""In-process token-bucket rate limiter.

Good enough for a single-process dev/demo deployment. A multi-replica setup
would implement the same port over Redis.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class InMemoryRateLimiter:
    def __init__(
        self,
        *,
        per_minute: int = 60,
        burst: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = float(burst if burst is not None else max(1, per_minute))
        self._refill_per_second = max(0.0, per_minute / 60.0)
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = self._clock()
            tokens, last = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - last) * self._refill_per_second)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True
