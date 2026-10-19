"""Step 7 -- encoder.

Embeds the rewritten query with the *same* model ingestion used. Two belt-and-
braces checks guard against silent model drift:

* the vector dimension must equal the configured ``vector_dim``; and
* optionally, the model id and dimension must appear in the vectors already
  stored in Postgres (see :class:`~rag_query.adapters.storage.postgres.StoredVectorProbe`).

A mismatch is fatal for the request, never silent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rag_ingest.ports.embedder import Embedder
from rag_query.domain.models import QueryStatus
from rag_query.logging import get_logger
from rag_query.nodes.base import Node
from rag_query.state import QueryState

log = get_logger(__name__)


class EncodeNode(Node):
    name = "encode"

    def __init__(
        self,
        embedder: Embedder,
        *,
        expected_dim: int | None = None,
        vector_probe: Any | None = None,
        check_index: bool = True,
    ) -> None:
        self._embedder = embedder
        self._expected_dim = expected_dim
        self._probe = vector_probe
        self._check_index = check_index

    def run(self, state: QueryState) -> dict[str, Any]:
        text = state.get("rewritten_query") or state.get("query") or ""
        if not text:
            return {"status": QueryStatus.ERROR, "message": "No text available to embed."}

        try:
            vector = np.asarray(self._embedder.embed([text])[0], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the graph
            log.exception("query embedding failed")
            return {"status": QueryStatus.ERROR, "message": f"Embedding failed: {exc}"}

        dim = int(vector.shape[0])
        if self._expected_dim and dim != self._expected_dim:
            return {
                "status": QueryStatus.ERROR,
                "message": f"Encoder produced dim {dim}, expected {self._expected_dim}.",
            }

        fingerprint = self._embedder.fingerprint

        # The index check is best-effort: a genuine mismatch is fatal, but an
        # unreachable database is an infrastructure problem that step 8
        # (retrieval) will surface -- it should not fail the encoder.
        warnings: list[str] = []
        if self._check_index and self._probe is not None:
            try:
                info = self._probe.describe()
            except Exception as exc:  # noqa: BLE001
                log.warning("index fingerprint check could not run: %s", exc)
                warnings.append(f"index_check_skipped: {exc}")
            else:
                if info.count and (
                    fingerprint.model_id not in info.model_ids or dim not in info.dims
                ):
                    return {
                        "status": QueryStatus.ERROR,
                        "message": (
                            f"Encoder {fingerprint.model_id!r} (dim {dim}) does not match the "
                            f"indexed vectors {list(info.model_ids)} (dims {list(info.dims)})."
                        ),
                    }

        return {
            "query_vector": vector,
            "embedding_dim": dim,
            "fingerprint": fingerprint,
            "status": QueryStatus.ENCODED,
            "warnings": warnings,
        }
