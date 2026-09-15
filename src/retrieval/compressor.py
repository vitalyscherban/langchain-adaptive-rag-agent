"""Contextual compression with a hard token budget enforcer.

Strips irrelevant content from retrieved chunks using either
``LLMChainExtractor`` (LLM rewrites each doc to only the relevant excerpt) or
``EmbeddingsFilter`` (drops docs below a similarity threshold, no LLM call),
then truncates the combined, compressed context to a hard token budget so it
never exceeds ``settings.context_token_budget`` regardless of what the
compressor produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tiktoken
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from src.config import Settings, get_settings

CompressorKind = Literal["llm_extractor", "embeddings_filter"]

_ENCODING_NAME = "cl100k_base"


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` using the ``cl100k_base`` tokenizer."""

    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return len(encoding.encode(text))


@dataclass
class CompressionResult:
    """Output of compressing + budget-enforcing a set of retrieved documents."""

    documents: list[Document]
    context_text: str
    token_count: int
    truncated: bool
    original_token_count: int


class ContextCompressor:
    """Compresses retrieved chunks and enforces a hard token budget."""

    def __init__(
        self,
        llm: BaseLanguageModel | None = None,
        embeddings: Embeddings | None = None,
        kind: CompressorKind = "embeddings_filter",
        settings: Settings | None = None,
        similarity_threshold: float = 0.6,
    ) -> None:
        self.settings = settings or get_settings()
        self.kind = kind
        self._llm = llm
        self._embeddings = embeddings
        self._similarity_threshold = similarity_threshold

    def _build_compressor(self):
        if self.kind == "llm_extractor":
            if self._llm is None:
                raise ValueError("An `llm` is required for the llm_extractor compressor.")
            return LLMChainExtractor.from_llm(self._llm)
        if self.kind == "embeddings_filter":
            if self._embeddings is None:
                raise ValueError("`embeddings` is required for the embeddings_filter compressor.")
            return EmbeddingsFilter(
                embeddings=self._embeddings, similarity_threshold=self._similarity_threshold
            )
        raise ValueError(f"Unknown compressor kind: {self.kind}")

    def compress(self, query: str, documents: list[Document]) -> CompressionResult:
        """Compress ``documents`` for relevance to ``query`` and enforce token budget.

        Args:
            query: The user query used to judge relevance.
            documents: Retrieved documents to compress.

        Returns:
            A :class:`CompressionResult` describing the final, budget-safe
            context.
        """

        if not documents:
            return CompressionResult(
                documents=[], context_text="", token_count=0, truncated=False,
                original_token_count=0,
            )

        compressor = self._build_compressor()
        compressed_docs = list(compressor.compress_documents(documents, query))
        if not compressed_docs:
            # Compressor filtered everything out; fall back to the raw docs
            # rather than sending an empty context to the LLM.
            compressed_docs = documents

        original_text = "\n\n".join(doc.page_content for doc in compressed_docs)
        original_token_count = count_tokens(original_text)

        budget = self.settings.context_token_budget
        final_docs, final_text, truncated = _enforce_token_budget(compressed_docs, budget)
        final_token_count = count_tokens(final_text)

        return CompressionResult(
            documents=final_docs,
            context_text=final_text,
            token_count=final_token_count,
            truncated=truncated,
            original_token_count=original_token_count,
        )


def _enforce_token_budget(
    documents: list[Document], budget: int
) -> tuple[list[Document], str, bool]:
    """Greedily keep whole documents until adding the next would exceed budget.

    If even the first document exceeds the budget on its own, its text is
    truncated at the token level so the hard cap is never violated.
    """

    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    kept_docs: list[Document] = []
    kept_texts: list[str] = []
    used_tokens = 0
    truncated = False

    for doc in documents:
        doc_tokens = encoding.encode(doc.page_content)
        if used_tokens + len(doc_tokens) <= budget:
            kept_docs.append(doc)
            kept_texts.append(doc.page_content)
            used_tokens += len(doc_tokens)
            continue

        remaining = budget - used_tokens
        truncated = True
        if remaining > 0:
            truncated_text = encoding.decode(doc_tokens[:remaining])
            kept_docs.append(Document(page_content=truncated_text, metadata=doc.metadata))
            kept_texts.append(truncated_text)
            used_tokens += remaining
        break

    return kept_docs, "\n\n".join(kept_texts), truncated
