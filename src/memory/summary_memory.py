"""Conversation memory backed by ``ConversationSummaryBufferMemory``.

Keeps chat history compact by automatically summarizing older turns once the
running token count exceeds ``settings.memory_max_token_limit``, avoiding
resending the full transcript on every request.
"""

from __future__ import annotations

from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.language_models import BaseLanguageModel

from src.config import Settings, get_settings


def build_conversation_memory(
    llm: BaseLanguageModel,
    settings: Settings | None = None,
) -> ConversationSummaryBufferMemory:
    """Create a summarizing conversation memory.

    Args:
        llm: The language model used to generate summaries once the buffer
            exceeds the token limit. A cheap model is typically sufficient
            here since summarization is a low-stakes task.
        settings: Optional pre-loaded settings.

    Returns:
        A configured ``ConversationSummaryBufferMemory`` instance.
    """

    settings = settings or get_settings()
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=settings.memory_max_token_limit,
        memory_key="chat_history",
        return_messages=True,
    )
