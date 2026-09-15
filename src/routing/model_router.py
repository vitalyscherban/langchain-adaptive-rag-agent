"""Model router: picks a cheap or strong LLM based on query complexity.

Supports any OpenAI-compatible API (OpenAI itself, Azure OpenAI gateways,
local proxies like LiteLLM/vLLM) by honoring ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings
from src.routing.classifier import QueryComplexity, classify_query


@dataclass(frozen=True)
class RoutingDecision:
    """Result of routing a query to a specific model."""

    complexity: QueryComplexity
    model_name: str


class ModelRouter:
    """Routes queries to a cheap or strong chat model based on complexity."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def route(self, query: str) -> RoutingDecision:
        """Classify ``query`` and return the routing decision (no LLM built yet)."""

        complexity = classify_query(query)
        model_name = (
            self.settings.cheap_model
            if complexity is QueryComplexity.SIMPLE
            else self.settings.strong_model
        )
        return RoutingDecision(complexity=complexity, model_name=model_name)

    def get_llm(self, query: str, **overrides) -> tuple[ChatOpenAI, RoutingDecision]:
        """Route ``query`` and construct the corresponding chat model.

        Args:
            query: The user query used to determine complexity/model.
            **overrides: Extra kwargs forwarded to :class:`ChatOpenAI`
                (e.g. ``temperature``).

        Returns:
            A tuple of ``(chat_model, routing_decision)``.
        """

        decision = self.route(query)
        llm = ChatOpenAI(
            model=decision.model_name,
            api_key=self.settings.openai_api_key or None,
            base_url=self.settings.openai_base_url,
            **overrides,
        )
        return llm, decision

    def top_k_range(self, complexity: QueryComplexity) -> tuple[int, int]:
        """Return the ``(min, max)`` top-k range for a given complexity."""

        if complexity is QueryComplexity.SIMPLE:
            return self.settings.simple_top_k_min, self.settings.simple_top_k_max
        return self.settings.complex_top_k_min, self.settings.complex_top_k_max
