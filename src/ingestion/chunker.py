"""Semantic chunking with a robust fallback.

Uses ``langchain_experimental.text_splitter.SemanticChunker`` (embedding-based
boundary detection) when enabled and available, otherwise falls back to
``RecursiveCharacterTextSplitter`` so ingestion always works even without a
live embeddings API call or the experimental package installed.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document],
    settings: Settings | None = None,
    embeddings=None,
) -> list[Document]:
    """Split documents into chunks, preferring semantic chunking.

    Args:
        documents: Source documents to split.
        settings: Optional pre-loaded :class:`Settings`; defaults to
            :func:`src.config.get_settings`.
        embeddings: Optional embeddings model to use for semantic chunking.
            Required only when ``settings.use_semantic_chunker`` is true. If
            not provided in that case, one is created lazily via
            :func:`src.retrieval.vectorstore.get_embeddings`.

    Returns:
        The list of chunked :class:`~langchain_core.documents.Document`
        objects, in the original document order.
    """

    settings = settings or get_settings()

    if settings.use_semantic_chunker:
        chunks = _try_semantic_chunk(documents, settings, embeddings)
        if chunks is not None:
            return chunks
        logger.warning("Semantic chunker unavailable; falling back to recursive splitter.")

    return _recursive_chunk(documents, settings)


def _try_semantic_chunk(
    documents: list[Document], settings: Settings, embeddings
) -> list[Document] | None:
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        return None

    try:
        if embeddings is None:
            from src.retrieval.vectorstore import get_embeddings

            embeddings = get_embeddings(settings)
        splitter = SemanticChunker(embeddings)
        return splitter.split_documents(documents)
    except Exception:  # noqa: BLE001 - any runtime failure triggers fallback
        logger.exception("Semantic chunking failed; using recursive splitter fallback.")
        return None


def _recursive_chunk(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
