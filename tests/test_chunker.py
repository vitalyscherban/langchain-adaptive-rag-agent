"""Tests for the semantic/recursive chunker."""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import Settings
from src.ingestion.chunker import chunk_documents


def _settings(**overrides) -> Settings:
    base = dict(chunk_size=50, chunk_overlap=10, use_semantic_chunker=False)
    base.update(overrides)
    return Settings(**base)


def test_recursive_fallback_splits_long_document_into_multiple_chunks() -> None:
    long_text = "Sentence number {}. " * 40
    long_text = long_text.format(*range(40))
    documents = [Document(page_content=long_text, metadata={"source": "doc1"})]

    chunks = chunk_documents(documents, settings=_settings())

    assert len(chunks) > 1
    # Metadata should be preserved on every chunk.
    assert all(chunk.metadata.get("source") == "doc1" for chunk in chunks)
    # No chunk should exceed the configured size by more than a small margin.
    assert all(len(chunk.page_content) <= 60 for chunk in chunks)


def test_short_document_stays_as_single_chunk() -> None:
    documents = [Document(page_content="Short doc.", metadata={"source": "doc2"})]

    chunks = chunk_documents(documents, settings=_settings(chunk_size=800, chunk_overlap=100))

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short doc."


def test_semantic_chunker_falls_back_when_unavailable(monkeypatch) -> None:
    """If SemanticChunker construction fails, we must fall back gracefully."""

    documents = [Document(page_content="Some content. " * 20, metadata={"source": "doc3"})]

    # use_semantic_chunker=True but no embeddings provided and import may not
    # be installed / may fail without network access -> should not raise.
    chunks = chunk_documents(
        documents, settings=_settings(use_semantic_chunker=True), embeddings=None
    )

    assert len(chunks) >= 1
