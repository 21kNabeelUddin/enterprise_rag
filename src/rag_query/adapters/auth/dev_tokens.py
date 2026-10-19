"""Dev-token authenticator: a fixed token -> principal map.

A deliberate stub for step 2. It models the *shape* of real SSO (a token in,
a principal with groups out) without pretending to verify signatures. Swapping
in OIDC/JWT means writing another :class:`~rag_query.ports.auth.Authenticator`.
"""

from __future__ import annotations

from collections.abc import Mapping

from rag_query.domain.errors import AuthenticationError
from rag_query.domain.models import Principal
from rag_query.logging import get_logger

log = get_logger(__name__)


class DevTokenAuthenticator:
    def __init__(
        self,
        principals: Mapping[str, Principal],
        *,
        anonymous: Principal,
    ) -> None:
        self._principals = dict(principals)
        self._anonymous = anonymous

    def authenticate(self, token: str | None) -> Principal:
        token = (token or "").strip()
        if not token:
            return self._anonymous

        principal = self._principals.get(token)
        if principal is None:
            log.warning("rejected unknown token %r", token[:8])
            raise AuthenticationError("unknown or expired token")

        log.debug("authenticated %s groups=%s", principal.user_id, principal.groups)
        return principal
