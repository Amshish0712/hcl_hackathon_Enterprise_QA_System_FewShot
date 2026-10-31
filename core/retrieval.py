"""Hybrid FAISS + BM25 retrieval with mandatory, current document-access checks."""

import math
import re
import sqlite3

import numpy as np

from core import database
from core.chat_config import ChatError
from core.document_management import ManagementError, _actor, _document
from core.embeddings import MODEL_NAME, embed_question
from core.index_errors import IndexingError
from core.permissions import can_access
from core.vector_store import vector_transaction
from core.trust import finite_number, finite_score


TOP_K = 4
CANDIDATE_POOL = 20
RRF_K = 60
BM25_K1, BM25_B = 1.5, 0.75
TOKEN = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset("""a an and any are as at be been by can could do does did for from had has have how i if in into
is it its me my must of on or our should so than that the their them then there these they this those to us was
we were what when where which who whom why will with within would you your""".split())
SCORE_FIELDS = ("similarity_score", "keyword_coverage", "bm25_score", "fused_score")
RANK_FIELDS = ("semantic_rank", "keyword_rank")
CHUNK_FROM = """
    FROM chunks c JOIN documents d ON d.id = c.document_id
    JOIN vector_mapping v ON v.chunk_id = c.id AND v.document_id = d.id
    WHERE d.extraction_status = 'completed' AND d.indexing_status = 'indexed'
          AND d.embedding_model = ?
"""
CHUNK_QUERY = """SELECT c.id AS chunk_id, c.document_id, c.chunk_text, c.section_title, c.page_number,
                       d.original_filename AS filename, d.uploaded_by """ + CHUNK_FROM


def terms(text: str) -> list[str]:
    """Lowercase keyword terms with stop words removed and a light plural stem."""
    result = []
    for token in TOKEN.findall(text.lower()):
        if token in STOP_WORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        result.append(token)
    return result


def chat_actor(db, user_id):
    try:
        actor = _actor(db, user_id)
    except ManagementError:
        raise ChatError("Sign in with a valid account to use document chat.") from None
    if not can_access(actor["role"], "Chat"):
        raise ChatError("Your account does not have permission to use Chat.")
    return actor


def _permitted_rows(db, user_id, rows):
    """Apply the existing document-preview policy to rows already scoped in SQL."""
    permitted, decisions = {}, {}
    for row in rows:
        if row["document_id"] not in decisions:
            try:
                _document(db, user_id, row["document_id"])
                decisions[row["document_id"]] = True
            except ManagementError:
                decisions[row["document_id"]] = False
        if decisions[row["document_id"]] and row["chunk_text"].strip():
            permitted[row["chunk_id"]] = {key: row[key] for key in (
                "chunk_id", "document_id", "chunk_text", "filename", "section_title", "page_number",
            )}
    return permitted


def _scoped_query(actor, query, parameters):
    if actor["role"] == "editor":
        query += " AND d.uploaded_by = ?"
        parameters.append(actor["id"])
    return query, parameters


def _authorized_chunks(db, user_id, chunk_ids):
    actor = chat_actor(db, user_id)
    rows = []
    for offset in range(0, len(chunk_ids), 400):
        batch = chunk_ids[offset:offset + 400]
        placeholders = ",".join("?" for _ in batch)
        query, parameters = _scoped_query(actor, CHUNK_QUERY + f" AND c.id IN ({placeholders})", [MODEL_NAME, *batch])
        rows.extend(db.execute(query, parameters))
    return _permitted_rows(db, user_id, rows)


def _accessible_chunks(db, user_id):
    actor = chat_actor(db, user_id)
    query, parameters = _scoped_query(actor, CHUNK_QUERY, [MODEL_NAME])
    return _permitted_rows(db, user_id, db.execute(query + " ORDER BY c.id", parameters).fetchall())


def accessible_chunk_count(user_id: int) -> int:
    with database.connection() as db:
        actor = chat_actor(db, user_id)
        query, parameters = _scoped_query(actor, "SELECT COUNT(*) " + CHUNK_FROM, [MODEL_NAME])
        return db.execute(query, parameters).fetchone()[0]


def authorize_evidence(user_id: int, evidence: list[dict]) -> list[dict]:
    """Rehydrate from SQLite; never trust caller-supplied passage text or metadata."""
    ids = [item.get("chunk_id") for item in evidence if type(item.get("chunk_id")) is int]
    with database.connection() as db:
        allowed = _authorized_chunks(db, user_id, ids)
    result = []
    for chunk_id in ids:
        if chunk_id in allowed and not any(item["chunk_id"] == chunk_id for item in result):
            original = next(item for item in evidence if item.get("chunk_id") == chunk_id)
            # Only numeric retrieval metadata is carried over; text always comes from SQLite.
            metadata = {"similarity_score": finite_score(original.get("similarity_score"))}
            metadata.update({name: finite_number(original.get(name)) for name in SCORE_FIELDS[1:]})
            metadata.update({name: original[name] for name in RANK_FIELDS if type(original.get(name)) is int})
            if original.get("source_label") in ("S1", "S2", "S3", "S4"):
                metadata["source_label"] = original["source_label"]
            result.append(dict(allowed[chunk_id], rank=len(result) + 1, **metadata))
    return result


def _semantic_ranking(index, vector, accessible):
    """Nearest accessible chunks by cosine similarity (inner product of unit vectors)."""
    candidates = min(CANDIDATE_POOL, index.ntotal)
    while True:
        scores, ids = index.search(vector, candidates)
        raw = {int(chunk_id): float(score) for chunk_id, score in zip(ids[0], scores[0]) if chunk_id >= 0}
        ranking = [chunk_id for chunk_id in raw if chunk_id in accessible][:CANDIDATE_POOL]
        # Inaccessible nearest neighbors cannot starve the caller's allowed documents.
        if len(ranking) >= CANDIDATE_POOL or candidates >= index.ntotal:
            return ranking, raw
        candidates = min(index.ntotal, candidates * 2)


def _bm25_ranking(question, accessible):
    """Okapi BM25 over accessible chunk text plus filename and section title."""
    query_terms = list(dict.fromkeys(terms(question)))
    documents = {chunk_id: terms(" ".join(filter(None, (row["filename"], row["section_title"], row["chunk_text"]))))
                 for chunk_id, row in accessible.items()}
    coverage, scores = {}, {}
    if not query_terms or not documents:
        return [], scores, coverage
    average = sum(map(len, documents.values())) / len(documents)
    frequency = {term: sum(term in set(tokens) for tokens in documents.values()) for term in query_terms}
    for chunk_id, tokens in documents.items():
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        matched = [term for term in query_terms if term in counts]
        coverage[chunk_id] = len(matched) / len(query_terms)
        score = 0.0
        for term in matched:
            idf = math.log(1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            tf = counts[term]
            score += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / average))
        if score > 0:
            scores[chunk_id] = score
    ranking = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:CANDIDATE_POOL]
    return ranking, scores, coverage


def _search(user_id: int, question: str, include_debug: bool):
    if not accessible_chunk_count(user_id):
        return [], []
    try:
        vector = embed_question(question)
        with vector_transaction() as transaction:
            chat_actor(transaction.db, user_id)
            index = transaction.load()
            if not index.ntotal:
                return [], []
            accessible = _accessible_chunks(transaction.db, user_id)
            semantic, raw = _semantic_ranking(index, vector, accessible)
            keyword, bm25, coverage = _bm25_ranking(question, accessible)
            # Reciprocal Rank Fusion: deterministic, scale-free combination of both rankings.
            fused = {}
            for ranking in (semantic, keyword):
                for position, chunk_id in enumerate(ranking, 1):
                    fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (RRF_K + position)
            for chunk_id in fused:
                if chunk_id not in raw:
                    # Keyword-only hits still get a true cosine score for the relevance gate.
                    raw[chunk_id] = float(np.dot(vector[0], index.reconstruct(chunk_id)))
            order = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], -raw[chunk_id], chunk_id))
            result, debug, seen_text = [], [], set()
            for position, chunk_id in enumerate(order, 1):
                text_key = " ".join(accessible[chunk_id]["chunk_text"].split()).casefold()
                duplicate = text_key in seen_text
                selected = not duplicate and len(result) < TOP_K
                seen_text.add(text_key)
                details = {
                    "similarity_score": finite_score(raw[chunk_id]),
                    "keyword_coverage": finite_number(coverage.get(chunk_id, 0.0)),
                    "bm25_score": finite_number(bm25.get(chunk_id, 0.0)),
                    "fused_score": finite_number(fused[chunk_id]),
                    "semantic_rank": semantic.index(chunk_id) + 1 if chunk_id in semantic else None,
                    "keyword_rank": keyword.index(chunk_id) + 1 if chunk_id in keyword else None,
                }
                if selected:
                    result.append(dict(accessible[chunk_id], rank=len(result) + 1, **details))
                if include_debug and position <= CANDIDATE_POOL:
                    debug.append(dict(details, candidate_rank=position, chunk_id=chunk_id,
                                      document_id=accessible[chunk_id]["document_id"],
                                      filename=accessible[chunk_id]["filename"],
                                      preview=accessible[chunk_id]["chunk_text"][:160],
                                      passed_access=True, duplicate=duplicate, selected=selected))
            if include_debug:
                for chunk_id, score in raw.items():
                    if chunk_id not in accessible:
                        debug.append({"chunk_id": chunk_id, "similarity_score": finite_score(score),
                                      "passed_access": False, "selected": False})
            return result, debug
    except (IndexingError, OSError, sqlite3.Error, RuntimeError):
        raise ChatError("Document retrieval is unavailable. Check the embedding model and restore a matching SQLite/FAISS index if needed, then retry.") from None


def retrieve_evidence(user_id: int, question: str) -> list[dict]:
    return _search(user_id, question, include_debug=False)[0]


def retrieve_with_debug(user_id: int, question: str) -> tuple[list[dict], list[dict]]:
    """Admin diagnostics: fused candidates, raw scores, and access-filter outcomes."""
    return _search(user_id, question, include_debug=True)
