"""Authentication port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_query.domain.models import Principal


@runtime_checkable
class Authenticator(Protocol):
    """Resolves a bearer token to a :class:`Principal`.

    Implementations raise :class:`~rag_query.domain.errors.AuthenticationError`
    when a token is present but invalid. A ``None`` token may resolve to an
    anonymous principal, depending on policy.
    """

    def authenticate(self, token: str | None) -> Principal: ...
