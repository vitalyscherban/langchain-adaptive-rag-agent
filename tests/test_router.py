"""Tests for the query complexity classifier and model router."""

from __future__ import annotations

from src.config import Settings
from src.routing.classifier import QueryComplexity, classify_query
from src.routing.model_router import ModelRouter


def test_short_factual_query_is_simple() -> None:
    assert classify_query("What is your refund policy?") is QueryComplexity.SIMPLE


def test_short_lookup_query_is_simple() -> None:
    assert classify_query("What port does the API use?") is QueryComplexity.SIMPLE


def test_comparison_query_is_complex() -> None:
    result = classify_query("Compare the pros and cons of plan A versus plan B in detail.")
    assert result is QueryComplexity.COMPLEX


def test_multi_hop_query_is_complex() -> None:
    query = (
        "Explain how billing works and also describe what happens if a payment "
        "fails and how that affects my production resources."
    )
    assert classify_query(query) is QueryComplexity.COMPLEX


def test_empty_query_is_simple() -> None:
    assert classify_query("   ") is QueryComplexity.SIMPLE


def test_router_routes_simple_query_to_cheap_model() -> None:
    settings = Settings(cheap_model="cheap-x", strong_model="strong-y")
    router = ModelRouter(settings)

    decision = router.route("What is your refund policy?")

    assert decision.complexity is QueryComplexity.SIMPLE
    assert decision.model_name == "cheap-x"


def test_router_routes_complex_query_to_strong_model() -> None:
    settings = Settings(cheap_model="cheap-x", strong_model="strong-y")
    router = ModelRouter(settings)

    decision = router.route(
        "Compare the security and billing implications of switching data residency."
    )

    assert decision.complexity is QueryComplexity.COMPLEX
    assert decision.model_name == "strong-y"


def test_router_top_k_ranges() -> None:
    settings = Settings(
        simple_top_k_min=1, simple_top_k_max=3, complex_top_k_min=5, complex_top_k_max=8
    )
    router = ModelRouter(settings)

    assert router.top_k_range(QueryComplexity.SIMPLE) == (1, 3)
    assert router.top_k_range(QueryComplexity.COMPLEX) == (5, 8)
