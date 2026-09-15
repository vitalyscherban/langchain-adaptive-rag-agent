"""Tests for the adaptive retriever's top-k logic (no live vector store)."""

from __future__ import annotations

from src.config import Settings
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.routing.classifier import QueryComplexity


def _settings() -> Settings:
    return Settings(
        simple_top_k_min=1,
        simple_top_k_max=3,
        complex_top_k_min=5,
        complex_top_k_max=8,
        mmr_fetch_k_multiplier=4,
    )


def test_determine_top_k_simple_uses_midpoint() -> None:
    retriever = AdaptiveRetriever(vectorstore=None, settings=_settings())
    assert retriever.determine_top_k(QueryComplexity.SIMPLE) == 2


def test_determine_top_k_complex_uses_midpoint() -> None:
    retriever = AdaptiveRetriever(vectorstore=None, settings=_settings())
    assert retriever.determine_top_k(QueryComplexity.COMPLEX) == 6


def test_determine_top_k_never_zero_for_degenerate_range() -> None:
    settings = Settings(simple_top_k_min=1, simple_top_k_max=1)
    retriever = AdaptiveRetriever(vectorstore=None, settings=settings)
    assert retriever.determine_top_k(QueryComplexity.SIMPLE) == 1


def test_retrieve_calls_mmr_search_with_dynamic_fetch_k() -> None:
    class _FakeVectorStore:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult):
            self.calls.append(
                {"query": query, "k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
            )
            return []

    fake_store = _FakeVectorStore()
    retriever = AdaptiveRetriever(vectorstore=fake_store, settings=_settings())

    result = retriever.retrieve("What is the refund window?", complexity=QueryComplexity.SIMPLE)

    assert result.top_k == 2
    assert result.fetch_k == 8
    assert fake_store.calls[0]["k"] == 2
    assert fake_store.calls[0]["fetch_k"] == 8
