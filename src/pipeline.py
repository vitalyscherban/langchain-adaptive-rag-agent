"""End-to-end orchestration of the adaptive RAG pipeline.

Wires together ingestion, retrieval, compression, prompt assembly, model
routing, memory, and usage tracking into a single ``AdaptiveRagPipeline``
used by both the CLI (:mod:`src.main`) and the evaluation script
(:mod:`evals.benchmark`).
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loaders import load_documents_from_dir
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.retrieval.compressor import ContextCompressor, count_tokens
from src.retrieval.prompt_assembler import PromptAssembler
from src.retrieval.vectorstore import build_vectorstore, get_embeddings
from src.routing.classifier import classify_query
from src.routing.model_router import ModelRouter
from src.tracking.usage_tracker import RequestUsage, UsageTracker


@dataclass
class QueryOutcome:
    """Full result of running one query through the pipeline."""

    answer: str
    usage: RequestUsage


class AdaptiveRagPipeline:
    """Token-optimized adaptive RAG pipeline (ingest + query)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embeddings: Embeddings = get_embeddings(self.settings)
        self.vectorstore: Chroma = build_vectorstore(self.settings, self.embeddings)
        self.retriever = AdaptiveRetriever(self.vectorstore, self.settings)
        self.compressor = ContextCompressor(
            embeddings=self.embeddings, kind="embeddings_filter", settings=self.settings
        )
        self.prompt_assembler = PromptAssembler()
        self.router = ModelRouter(self.settings)
        self.tracker = UsageTracker(self.settings)

    def ingest(self, directory: str) -> int:
        """Load, chunk, and index all documents in ``directory``.

        Returns:
            The number of chunks indexed.
        """

        documents = load_documents_from_dir(directory)
        chunks = chunk_documents(documents, self.settings, self.embeddings)
        if chunks:
            self.vectorstore.add_documents(chunks)
        return len(chunks)

    def query(self, question: str, use_compression: bool = True) -> QueryOutcome:
        """Run one question through the full adaptive pipeline.

        Args:
            question: The user's question.
            use_compression: Whether to apply contextual compression + token
                budget enforcement. Disabling this (used by the eval script's
                "baseline" mode) sends raw retrieved chunks instead.

        Returns:
            A :class:`QueryOutcome` with the generated answer and usage stats.
        """

        complexity = classify_query(question)
        retrieval = self.retriever.retrieve(question, complexity=complexity)

        context_text, context_docs = self._build_context(question, retrieval.documents, use_compression)

        messages = self.prompt_assembler.assemble(question, context_text)
        llm, decision = self.router.get_llm(question)

        with self.tracker.timed() as timing:
            response = llm.invoke(messages)

        prompt_tokens, completion_tokens = _extract_token_usage(response, messages, context_text)
        usage = self.tracker.record(
            query=question,
            model=decision.model_name,
            complexity=decision.complexity.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=timing["latency_seconds"],
        )
        return QueryOutcome(answer=response.content, usage=usage)

    def _build_context(
        self, question: str, documents: list[Document], use_compression: bool
    ) -> tuple[str, list[Document]]:
        if not use_compression:
            text = "\n\n".join(doc.page_content for doc in documents)
            return text, documents

        result = self.compressor.compress(question, documents)
        return result.context_text, result.documents


def _extract_token_usage(response, messages, context_text: str) -> tuple[int, int]:
    """Best-effort extraction of prompt/completion token counts.

    Falls back to local ``tiktoken`` estimation when the model response does
    not include usage metadata (e.g. some OpenAI-compatible proxies).
    """

    usage_metadata = getattr(response, "usage_metadata", None) or {}
    prompt_tokens = usage_metadata.get("input_tokens")
    completion_tokens = usage_metadata.get("output_tokens")

    if prompt_tokens is None:
        prompt_text = "\n".join(str(m.content) for m in messages)
        prompt_tokens = count_tokens(prompt_text)
    if completion_tokens is None:
        completion_tokens = count_tokens(str(response.content))

    return prompt_tokens, completion_tokens
