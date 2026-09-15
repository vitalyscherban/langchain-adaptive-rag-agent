"""Chroma-backed vector store setup (local, no external DB required)."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Build the embeddings client used for both indexing and queries."""

    settings = settings or get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
    )


def build_vectorstore(
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Open (or create) the local, on-disk Chroma collection.

    Args:
        settings: Optional pre-loaded settings.
        embeddings: Optional embeddings model; created via
            :func:`get_embeddings` if not supplied.

    Returns:
        A ``Chroma`` vector store persisted at ``settings.chroma_persist_dir``.
    """

    settings = settings or get_settings()
    embeddings = embeddings or get_embeddings(settings)
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def index_documents(
    documents: list[Document],
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Add ``documents`` to the persisted Chroma collection and return it."""

    settings = settings or get_settings()
    vectorstore = build_vectorstore(settings, embeddings)
    if documents:
        vectorstore.add_documents(documents)
    return vectorstore
