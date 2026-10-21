"""Read approved, clinic-authored knowledge for grounded chatbot answers.

This module deliberately does not invoke an LLM. Returned answers are authored
and published in MySQL and are returned to the user without model rewriting.
"""

from __future__ import annotations

import re
from typing import Any

def _get_db_connection():
    """Lazily import the DB connector so pure ranking tests need no driver."""
    from src.database.database import get_db_connection
    return get_db_connection()


# Keep free-form answers inside the clinic/health domain even if an unrelated
# record is accidentally added to this table.
_ALLOWED_CATEGORIES = (
    "APPOINTMENT", "CLINIC", "DOCTOR", "FAQ", "HEALTH", "MEDICAL",
    "PHARMACY", "POLICY", "SAFETY", "SCREENING",
)

_STOP_WORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "before", "can", "could",
    "did", "do", "does", "for", "from", "give", "hello", "help",
    "how", "i", "in", "into", "is", "it", "its", "me", "my", "of",
    "on", "or", "please", "tell", "than", "that", "the", "their",
    "them", "then", "there", "this", "to", "us", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "explain", "describe", "know", "want", "need",
}


def meaningful_tokens(value: Any) -> set[str]:
    """Return normalized, meaningful query/content tokens."""
    tokens = set(re.findall(r"[a-z0-9]+", str(value or "").lower()))
    return {token for token in tokens if len(token) >= 2 and token not in _STOP_WORDS}


def score_knowledge_row(query: str, row: dict[str, Any]) -> tuple[float, bool]:
    """Score a candidate and indicate whether its lexical overlap is sufficient.

    Strong title/keyword matches can answer short searches. For matches that occur
    only in the body, require at least 75% of meaningful query terms to be present.
    This is a conservative lexical relevance gate, not semantic/vector retrieval.
    """
    query_tokens = meaningful_tokens(query)
    if not query_tokens:
        return 0.0, False

    title_tokens = meaningful_tokens(row.get("title"))
    keyword_tokens = meaningful_tokens(row.get("keywords"))
    answer_tokens = meaningful_tokens(row.get("answer"))

    denominator = len(query_tokens)
    title_coverage = len(query_tokens & title_tokens) / denominator
    keyword_coverage = len(query_tokens & keyword_tokens) / denominator
    answer_coverage = len(query_tokens & answer_tokens) / denominator

    score = (4.0 * title_coverage) + (3.0 * keyword_coverage) + answer_coverage
    relevant = (
        title_coverage >= 0.5
        or keyword_coverage >= 0.5
        or answer_coverage >= 0.75
    )
    return score, relevant


def search_clinic_knowledge(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Find published knowledge items that materially overlap the query.

    Raises database errors to the service layer so it can distinguish an
    unavailable knowledge source from a successful search with no matches.
    """
    query_tokens = sorted(meaningful_tokens(query))[:12]
    if not query_tokens:
        return []

    # Search candidates with parameterized SQL. Tokenization above prevents SQL
    # wildcard/control characters from coming from user-supplied search terms.
    conditions: list[str] = []
    params: list[str] = []
    for token in query_tokens:
        conditions.append(
            "(LOWER(title) LIKE %s OR LOWER(answer) LIKE %s "
            "OR LOWER(COALESCE(keywords, '')) LIKE %s)"
        )
        pattern = f"%{token}%"
        params.extend([pattern, pattern, pattern])

    category_placeholders = ", ".join(["%s"] * len(_ALLOWED_CATEGORIES))
    sql = f"""
        SELECT id, title, answer, keywords, category
        FROM clinic_knowledge
        WHERE is_published = 1
          AND UPPER(category) IN ({category_placeholders})
          AND ({' OR '.join(conditions)})
        ORDER BY updated_at DESC, id DESC
        LIMIT 100
    """

    connection = None
    cursor = None
    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(sql, tuple(_ALLOWED_CATEGORIES) + tuple(params))
        candidates = cursor.fetchall() or []
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in candidates:
        if str(row.get("category") or "").strip().upper() not in _ALLOWED_CATEGORIES:
            continue
        score, relevant = score_knowledge_row(query, row)
        if relevant:
            ranked.append((score, row))

    ranked.sort(key=lambda item: item[0], reverse=True)
    safe_limit = max(1, min(int(limit), 10))

    # Return only fields needed by the chatbot; never pass database keywords or
    # internal metadata into the user-facing response.
    return [
        {
            "id": row.get("id"),
            "title": str(row.get("title") or "").strip(),
            "answer": str(row.get("answer") or "").strip(),
            "category": str(row.get("category") or "GENERAL").strip(),
        }
        for _, row in ranked[:safe_limit]
        if str(row.get("answer") or "").strip()
    ]
