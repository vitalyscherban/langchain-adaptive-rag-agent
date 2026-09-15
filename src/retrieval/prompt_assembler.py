"""Prompt assembly with a cached/reused static system prompt.

The system prompt is defined once as a module-level constant and reused
across every request, so it is never regenerated per-call. Only the
compressed retrieval context and the user question vary. When the underlying
chat model / API supports prompt caching (e.g. OpenAI's automatic prompt
caching for repeated prefixes), keeping this prefix identical across calls
maximizes cache hits and further reduces cost.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

STATIC_SYSTEM_PROMPT = (
    "You are a concise, accurate knowledge/support assistant. Answer the "
    "user's question using ONLY the provided context. If the context does "
    "not contain the answer, say you don't know rather than guessing. Keep "
    "answers focused and avoid restating the context verbatim."
)


class PromptAssembler:
    """Assembles the final message list sent to the chat model."""

    def __init__(self, system_prompt: str = STATIC_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def assemble(
        self,
        question: str,
        context_text: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> list[BaseMessage]:
        """Build the full message list for one query.

        Args:
            question: The user's current question.
            context_text: Compressed, budget-enforced retrieval context.
            chat_history: Optional prior turns (already summarized/trimmed by
                the conversation memory) to include for continuity.

        Returns:
            A list of messages ready to pass to a chat model.
        """

        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        if chat_history:
            messages.extend(chat_history)

        user_content = (
            f"Context:\n{context_text}\n\nQuestion: {question}"
            if context_text
            else f"Question: {question}"
        )
        messages.append(HumanMessage(content=user_content))
        return messages
