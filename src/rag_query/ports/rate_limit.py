"""Per-caller rate-limiting port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RateLimiter(Protocol):
    """Token-bucket style quota, keyed by caller (usually ``user_id``)."""

    def allow(self, key: str) -> bool:
        """Return ``True`` if the caller may proceed, ``False`` if over quota."""
