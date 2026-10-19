"""Postgres hybrid retriever (step 8).

Dense kNN (pgvector HNSW) and lexical BM25-ish ranking (Postgres ``ts_rank_cd``)
are fetched independently, fused with Reciprocal Rank Fusion, then resolved to
full chunk rows. Fusing in Python keeps the ranking maths testable and lets the
lexical side retry with a relaxed term set when the strict AND query is too
narrow.

Why not one giant SQL statement?
* ``OR`` over common terms scans the whole 1.3M-row chunk table (~2.9 s).
* RRF is a handful of arithmetic operations; doing it in Python is clearer and
  costs one small ``WHERE chunk_id = ANY(...)`` query.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import Engine, String, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Session, sessionmaker

from rag_query.domain.models import (
    Principal,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)
from rag_query.logging import get_logger

log = get_logger(__name__)

_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from
    had has have having he her here hers him his how i if in into is it its just
    me more most my no not of on or our ours out over own same she should so some
    such than that the their theirs them then there these they this those through
    to too under until up very was we were what when where which while who whom
    why will with would you your yours about above after again against all am
    any because before below between both during each few further once only other
    s t than then once
    """.split()  # noqa: SIM905 - a readable word list beats 100 quoted tokens
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&.'-]*")
# Query-intent words that make an AND tsquery too narrow without adding subject
# matter. Company suffixes are included because names are also extracted into
# the CIK filter, where they belong.
_QUERY_FILLERS = frozenset(
    """
    primary main key major important specifically regarding concerning facing
    inc incorporated corp corporation company companies annual report reports
    please summarize summarise summary explain describe disclosed disclose
    discuss discussed identify state note compare comparison versus vs
    say says said tell tells mention mentioned ask asked
    """.split()  # noqa: SIM905 - a readable list is clearer than quoted literals
)
_ARRAY_FILTERS = {
    "ciks": "c.cik = ANY(CAST(:ciks AS varchar[]))",
    "years": "c.year = ANY(CAST(:years AS varchar[]))",
    "section_keys": "c.section_key = ANY(CAST(:section_keys AS varchar[]))",
    "doc_ids": "c.doc_id = ANY(CAST(:doc_ids AS varchar[]))",
}


@dataclass(frozen=True)
class _DenseRow:
    chunk_id: str
    dense_score: float


@dataclass(frozen=True)
class _LexicalRow:
    chunk_id: str
    lexical_score: float


@dataclass(frozen=True)
class _RankedRow:
    chunk_id: str
    rrf_score: float
    dense_score: float | None
    lexical_score: float | None


class PostgresHybridRetriever:
    def __init__(
        self,
        engine: Engine,
        *,
        model_id: str,
        dim: int,
        dense_k: int = 50,
        lexical_k: int = 50,
        fused_k: int = 40,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 0.8,
        lexical_min_hits: int = 8,
        lexical_max_terms: int = 8,
        statement_timeout_ms: int = 15_000,
        hnsw_ef_search: int = 100,
        hnsw_iterative_scan: bool = True,
    ) -> None:
        self._session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
        self._model_id = model_id
        self._dim = int(dim)
        self._dense_k = max(1, dense_k)
        self._lexical_k = max(1, lexical_k)
        self._fused_k = max(1, fused_k)
        self._rrf_k = max(1, rrf_k)
        self._dense_weight = max(0.0, dense_weight)
        self._lexical_weight = max(0.0, lexical_weight)
        self._lexical_min_hits = max(1, lexical_min_hits)
        self._lexical_max_terms = max(1, lexical_max_terms)
        self._statement_timeout_ms = max(1_000, statement_timeout_ms)
        self._hnsw_ef_search = max(1, hnsw_ef_search)
        self._hnsw_iterative_scan = hnsw_iterative_scan

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def retrieve(
        self,
        query: str,
        vector: np.ndarray,
        *,
        principal: Principal,
        filters: RetrievalFilters | None = None,
        limit: int | None = None,
    ) -> RetrievalResult:
        groups = tuple(principal.groups)
        filters = filters or RetrievalFilters()
        if not groups:
            log.info("retrieval denied: principal %s has no groups", principal.user_id)
            return RetrievalResult(filters=filters)

        top_k = max(1, limit or self._fused_k)

        with self._session_factory() as session:
            self._configure_session(session)
            dense_rows = self._dense(session, vector, groups, filters)
            lexical_rows, lexical_query = self._lexical_with_relaxation(
                session, query, groups, filters
            )
            fused = self._fuse(dense_rows, lexical_rows)
            selected = fused[:top_k]
            chunks = self._fetch_chunks(session, [row.chunk_id for row in selected])

        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        results: list[RetrievedChunk] = []
        for rank, row in enumerate(selected, start=1):
            chunk = by_id.get(row.chunk_id)
            if chunk is None:
                continue
            results.append(
                chunk.model_copy(
                    update={
                        "dense_score": row.dense_score,
                        "lexical_score": row.lexical_score,
                        "fused_score": row.rrf_score,
                        "rank": rank,
                    }
                )
            )

        log.info(
            "retrieval query=%r dense=%d lexical=%d fused=%d returned=%d groups=%s lex=%r",
            query[:80],
            len(dense_rows),
            len(lexical_rows),
            len(fused),
            len(results),
            list(groups),
            lexical_query,
        )
        return RetrievalResult(
            candidates=tuple(results),
            dense_candidates=len(dense_rows),
            lexical_candidates=len(lexical_rows),
            fused_candidates=len(fused),
            lexical_query=lexical_query,
            filters=filters,
        )

    def expand_neighbors(
        self,
        chunks: Sequence[RetrievedChunk],
        *,
        window: int = 1,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]:
        if window <= 0 or not chunks:
            return list(chunks)

        groups = tuple(principal.groups) if principal is not None else ()
        if principal is not None and not groups:
            return list(chunks)

        primary_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        windows: dict[tuple[str, str, int], list[RetrievedChunk]] = {}

        with self._session_factory() as session:
            self._configure_session(session, apply_hnsw=False)
            for chunk in chunks:
                low = max(0, chunk.chunk_index - window)
                high = chunk.chunk_index + window
                cache_key = (chunk.doc_id, chunk.section_key, low)
                if cache_key in windows:
                    continue
                sql = (
                    "SELECT chunk_id, doc_id, cik, year, section_key, chunk_index, "
                    "text, token_count "
                    "FROM chunks c "
                    "WHERE c.doc_id = :doc_id AND c.section_key = :section_key "
                    "AND c.chunk_index BETWEEN :low AND :high "
                )
                params: dict[str, Any] = {
                    "doc_id": chunk.doc_id,
                    "section_key": chunk.section_key,
                    "low": low,
                    "high": high,
                }
                if groups:
                    sql += "AND c.acl && CAST(:groups AS varchar[]) "
                    params["groups"] = list(groups)
                sql += "ORDER BY c.chunk_index"
                rows = session.execute(text(sql), params).all()
                windows[cache_key] = [_row_to_chunk(row, primary_by_id) for row in rows]

        seen: set[str] = set()
        expanded: list[RetrievedChunk] = []
        for chunk in chunks:
            low = max(0, chunk.chunk_index - window)
            window_chunks = windows.get((chunk.doc_id, chunk.section_key, low), ())
            # Keep the reranked hit first so the strongest source becomes S1,
            # followed by its adjacent context in document order.
            ordered = [chunk, *(c for c in window_chunks if c.chunk_id != chunk.chunk_id)]
            for candidate in ordered:
                if candidate.chunk_id in seen:
                    continue
                seen.add(candidate.chunk_id)
                expanded.append(candidate)
        return expanded

    # --- SQL stages ------------------------------------------------------
    def _dense(
        self,
        session: Session,
        vector: np.ndarray,
        groups: Sequence[str],
        filters: RetrievalFilters,
    ) -> list[_DenseRow]:
        filter_sql, filter_params = _filter_clause(filters)
        sql = (
            "SELECT e.chunk_id, "
            "1 - (e.embedding <=> CAST(:qvec AS vector)) AS dense_score "
            "FROM embeddings e "
            "JOIN chunks c ON c.chunk_id = e.chunk_id "
            "WHERE e.model_id = :model_id AND e.dim = :dim "
            "AND c.acl && CAST(:groups AS varchar[]) "
            f"{filter_sql} "
            "ORDER BY e.embedding <=> CAST(:qvec AS vector) "
            "LIMIT :dense_k"
        )
        stmt = text(sql).bindparams(
            bindparam("groups", type_=ARRAY(String)),
        )
        params: dict[str, Any] = {
            "qvec": _vector_literal(vector),
            "model_id": self._model_id,
            "dim": self._dim,
            "groups": list(groups),
            "dense_k": self._dense_k,
            **filter_params,
        }
        rows = session.execute(stmt, params).all()
        return [_DenseRow(row.chunk_id, float(row.dense_score)) for row in rows]

    def _lexical_with_relaxation(
        self,
        session: Session,
        query: str,
        groups: Sequence[str],
        filters: RetrievalFilters,
    ) -> tuple[list[_LexicalRow], str]:
        variants = _lexical_variants(query, max_terms=self._lexical_max_terms)
        if not variants:
            return [], ""

        best: list[_LexicalRow] = []
        best_query = ""
        for index, terms in enumerate(variants):
            lex_query = " ".join(terms)
            rows = self._lexical(session, lex_query, groups, filters)
            if len(rows) > len(best):
                best = rows
                best_query = lex_query
            if len(rows) >= self._lexical_min_hits:
                log.debug("lexical variant %d (%s) returned %d rows", index, terms, len(rows))
                return rows, lex_query
        log.debug(
            "lexical relaxation exhausted; best variant %r returned %d rows",
            best_query,
            len(best),
        )
        return best, best_query

    def _lexical(
        self,
        session: Session,
        lex_query: str,
        groups: Sequence[str],
        filters: RetrievalFilters,
    ) -> list[_LexicalRow]:
        filter_sql, filter_params = _filter_clause(filters)
        sql = (
            "SELECT c.chunk_id, "
            "ts_rank_cd(c.tsv, q) AS lexical_score "
            "FROM chunks c, plainto_tsquery('english', :lex_query) AS q "
            "WHERE c.tsv @@ q "
            "AND c.acl && CAST(:groups AS varchar[]) "
            f"{filter_sql} "
            "ORDER BY lexical_score DESC "
            "LIMIT :lexical_k"
        )
        stmt = text(sql).bindparams(bindparam("groups", type_=ARRAY(String)))
        params: dict[str, Any] = {
            "lex_query": lex_query,
            "groups": list(groups),
            "lexical_k": self._lexical_k,
            **filter_params,
        }
        rows = session.execute(stmt, params).all()
        return [_LexicalRow(row.chunk_id, float(row.lexical_score)) for row in rows]

    def _fetch_chunks(self, session: Session, chunk_ids: Sequence[str]) -> list[RetrievedChunk]:
        if not chunk_ids:
            return []
        sql = (
            "SELECT chunk_id, doc_id, cik, year, section_key, chunk_index, text, token_count "
            "FROM chunks WHERE chunk_id = ANY(CAST(:ids AS varchar[]))"
        )
        stmt = text(sql).bindparams(bindparam("ids", type_=ARRAY(String)))
        rows = session.execute(stmt, {"ids": list(chunk_ids)}).all()
        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                doc_id=row.doc_id,
                cik=row.cik,
                year=row.year,
                section_key=row.section_key,
                chunk_index=int(row.chunk_index),
                text=row.text,
                token_count=int(row.token_count),
            )
            for row in rows
        ]

    def _fuse(
        self,
        dense: Sequence[_DenseRow],
        lexical: Sequence[_LexicalRow],
    ) -> list[_RankedRow]:
        dense_rank = {
            row.chunk_id: (index + 1, row.dense_score)
            for index, row in enumerate(dense)
        }
        lexical_rank = {
            row.chunk_id: (index + 1, row.lexical_score)
            for index, row in enumerate(lexical)
        }
        fused: list[_RankedRow] = []
        for chunk_id in set(dense_rank) | set(lexical_rank):
            dense_position, dense_score = dense_rank.get(chunk_id, (0, 0.0))
            lexical_position, lexical_score = lexical_rank.get(chunk_id, (0, 0.0))
            score = 0.0
            if dense_position:
                score += self._dense_weight / (self._rrf_k + dense_position)
            if lexical_position:
                score += self._lexical_weight / (self._rrf_k + lexical_position)
            fused.append(
                _RankedRow(
                    chunk_id=chunk_id,
                    rrf_score=score,
                    dense_score=dense_score if dense_position else None,
                    lexical_score=lexical_score if lexical_position else None,
                )
            )
        fused.sort(key=lambda row: (row.rrf_score, row.dense_score or 0.0), reverse=True)
        return fused

    def _configure_session(self, session: Session, *, apply_hnsw: bool = True) -> None:
        try:
            # pgvector registers its GUCs only once the extension library has
            # been loaded in this backend. A trivial vector cast does that.
            session.execute(text("SELECT '[0]'::vector"))
            session.execute(
                text("SELECT set_config('statement_timeout', :value, true)"),
                {"value": f"{self._statement_timeout_ms}ms"},
            )
            if not apply_hnsw:
                return
            session.execute(
                text("SELECT set_config('hnsw.ef_search', :value, true)"),
                {"value": str(self._hnsw_ef_search)},
            )
            if self._hnsw_iterative_scan:
                session.execute(
                    text("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
                )
                session.execute(
                    text("SELECT set_config('hnsw.max_scan_tuples', '20000', true)")
                )
        except Exception as exc:  # noqa: BLE001 - tuning hints must not break search
            log.warning("could not apply retrieval session settings: %s", exc)


def _row_to_chunk(row: Any, primary_by_id: dict[str, RetrievedChunk]) -> RetrievedChunk:
    primary = primary_by_id.get(row.chunk_id)
    if primary is not None:
        return primary
    return RetrievedChunk(
        chunk_id=row.chunk_id,
        doc_id=row.doc_id,
        cik=row.cik,
        year=row.year,
        section_key=row.section_key,
        chunk_index=int(row.chunk_index),
        text=row.text,
        token_count=int(row.token_count),
        is_neighbor=True,
    )


def _filter_clause(filters: RetrievalFilters) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for field, clause in _ARRAY_FILTERS.items():
        values = getattr(filters, field)
        if values:
            clauses.append(clause)
            params[field] = list(values)
    if not clauses:
        return "", {}
    return "AND " + " AND ".join(clauses), params


def _lexical_variants(query: str, *, max_terms: int) -> list[list[str]]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(query or ""):
        token = match.group(0).lower().strip(".'-")
        if token and token not in _STOPWORDS and (len(token) > 1 or token.isdigit()):
            tokens.append(token)
    terms: list[str] = []
    for token in tokens:
        if token not in terms:
            terms.append(token)
    terms = terms[:max_terms]
    if not terms:
        return []

    variants: list[list[str]] = []

    def add(candidate: list[str]) -> None:
        if len(candidate) >= 2 and candidate not in variants:
            variants.append(candidate)

    # Strict AND first, then drop boilerplate. A rewritten question often
    # starts with "primary ... facing ... specifically", and keeping those
    # words can make the tsquery match nothing even when the subject terms are
    # common in the corpus.
    add(terms)
    core = [term for term in terms if term not in _QUERY_FILLERS]
    add(core or terms)

    # Finally relax from the tail. Subject words appear early in rewritten
    # questions; trailing qualifiers are the least costly to lose. Never go
    # below two terms: one common term (e.g. "risk") returns arbitrary chunks.
    current = list(core or terms)
    while len(current) > 2:
        current = current[:-1]
        add(current)
    return variants


def _vector_literal(vector: np.ndarray) -> str:
    values = np.asarray(vector, dtype=np.float32).ravel()
    return "[" + ",".join(f"{float(value):.7f}" for value in values) + "]"
