"""Step 2 -- authenticate.

Resolves the request's bearer token to a principal. The groups on that principal
are what step 8 will use as the retrieval ACL filter, so they are resolved here,
before any retrieval work happens.
"""

from __future__ import annotations

from typing import Any

from rag_query.domain.errors import AuthenticationError
from rag_query.domain.models import QueryStatus
from rag_query.nodes.base import Node
from rag_query.ports.auth import Authenticator
from rag_query.state import QueryState


class AuthenticateNode(Node):
    name = "authenticate"

    def __init__(self, authenticator: Authenticator) -> None:
        self._authenticator = authenticator

    def run(self, state: QueryState) -> dict[str, Any]:
        try:
            principal = self._authenticator.authenticate(state.get("token"))
        except AuthenticationError as exc:
            return {"status": QueryStatus.UNAUTHORIZED, "message": str(exc)}
        return {"principal": principal}
