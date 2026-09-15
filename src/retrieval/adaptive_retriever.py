"""Adaptive retriever: embedding search + MMR re-ranking + dynamic top-k.

Combines:

* Dynamic ``top_k`` selection driven by :class:`~src.routing.classifier.QueryComplexity`
  (1-3 chunks for simple queries, 5-8 for complex ones).
* Maximal Marginal Relevance (MMR) re-ranking to reduce redundancy among the
  retrieved chunks, fetching a larger candidate pool (``fetch_k``) before
  narrowing down to ``top_k``.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.routing.classifier import QueryComplexity, classify_query


@dataclass(frozen=True)
class RetrievalResult:
    """Documents retrieved for a query, plus the metadata used to get them."""

    documents: list[Document]
    complexity: QueryComplexity
    top_k: int
    fetch_k: int


class AdaptiveRetriever:
    """Retrieves documents with complexity-driven, MMR-reranked top-k."""

    def __init__(self, vectorstore: Chroma, settings: Settings | None = None) -> None:
        self.vectorstore = vectorstore
        self.settings = settings or get_settings()

    def determine_top_k(self, complexity: QueryComplexity) -> int:
        """Pick a concrete ``top_k`` within the configured range for ``complexity``.

        Uses the midpoint of the configured min/max range, rounded down, so
        behavior is deterministic rather than random.
        """

        if complexity is QueryComplexity.SIMPLE:
            low, high = self.settings.simple_top_k_min, self.settings.simple_top_k_max
        else:
            low, high = self.settings.complex_top_k_min, self.settings.complex_top_k_max
        return (low + high) // 2 or low

    def retrieve(self, query: str, complexity: QueryComplexity | None = None) -> RetrievalResult:
        """Retrieve documents for ``query`` using adaptive, MMR-reranked search.

        Args:
            query: The user question.
            complexity: Pre-computed complexity; classified from ``query`` if
                omitted.

        Returns:
            A :class:`RetrievalResult` with the retrieved documents and the
            parameters used.
        """

        complexity = complexity or classify_query(query)
        top_k = self.determine_top_k(complexity)
        fetch_k = max(top_k * self.settings.mmr_fetch_k_multiplier, top_k)

        documents = self.vectorstore.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=fetch_k,
            lambda_mult=self.settings.mmr_lambda,
        )
        return RetrievalResult(
            documents=documents, complexity=complexity, top_k=top_k, fetch_k=fetch_k
        )
