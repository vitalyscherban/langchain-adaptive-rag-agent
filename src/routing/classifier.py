"""Query complexity classification.

Classifies a query as ``simple`` or ``complex`` using lightweight heuristics
(no LLM call needed, keeping classification itself token-free):

* Question length and word count.
* Number of distinct question clauses (multi-hop indicators like "and",
  "also", multiple "?").
* Presence of comparison / reasoning keywords ("compare", "why", "explain",
  "difference between", "relationship", etc.) that typically require pulling
  together more context.

This keeps the classifier fast, free, and deterministic, which matters
because it runs on *every* query before any retrieval happens.
"""

from __future__ import annotations

import re
from enum import Enum

_COMPLEX_KEYWORDS = {
    "compare",
    "comparison",
    "difference",
    "differences",
    "why",
    "explain",
    "relationship",
    "trade-off",
    "tradeoff",
    "pros and cons",
    "advantages and disadvantages",
    "how does",
    "how do",
    "step by step",
    "walk me through",
    "multiple",
    "several",
    "across",
    "combine",
    "summarize",
    "summarise",
    "analyze",
    "analyse",
    "root cause",
}

_MULTI_HOP_CONNECTORS = {" and ", " also ", " as well as ", " in addition ", " then "}

_SIMPLE_WORD_COUNT_THRESHOLD = 12
_SIMPLE_CHAR_COUNT_THRESHOLD = 80


class QueryComplexity(str, Enum):
    """Discrete complexity buckets driving retrieval depth and model choice."""

    SIMPLE = "simple"
    COMPLEX = "complex"


def classify_query(query: str) -> QueryComplexity:
    """Classify ``query`` as :class:`QueryComplexity.SIMPLE` or ``COMPLEX``.

    A query is treated as complex if it is long, contains multiple
    question/connector clauses, or uses reasoning/comparison keywords.
    Otherwise it is treated as simple (short, single-fact lookups).

    Args:
        query: The raw user question.

    Returns:
        The detected :class:`QueryComplexity`.
    """

    normalized = query.strip().lower()
    if not normalized:
        return QueryComplexity.SIMPLE

    word_count = len(normalized.split())
    question_marks = normalized.count("?")
    has_multi_hop_connector = any(conn in f" {normalized} " for conn in _MULTI_HOP_CONNECTORS)
    has_complex_keyword = any(_keyword_matches(normalized, kw) for kw in _COMPLEX_KEYWORDS)

    is_complex = (
        word_count > _SIMPLE_WORD_COUNT_THRESHOLD
        or len(normalized) > _SIMPLE_CHAR_COUNT_THRESHOLD
        or question_marks > 1
        or has_multi_hop_connector
        or has_complex_keyword
    )

    return QueryComplexity.COMPLEX if is_complex else QueryComplexity.SIMPLE


def _keyword_matches(text: str, keyword: str) -> bool:
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text) is not None
