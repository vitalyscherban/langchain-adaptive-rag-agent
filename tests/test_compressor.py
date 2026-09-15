"""Tests for the contextual compressor and hard token budget enforcer."""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import Settings
from src.retrieval.compressor import ContextCompressor, count_tokens


class _FakeEmbeddingsFilterCompressor:
    """Stand-in for EmbeddingsFilter that returns docs unchanged."""

    def compress_documents(self, documents, query):
        return documents


def _make_documents(n: int, words_per_doc: int = 300) -> list[Document]:
    return [
        Document(page_content=("word " * words_per_doc).strip(), metadata={"i": i})
        for i in range(n)
    ]


def test_compress_enforces_hard_token_budget(monkeypatch) -> None:
    settings = Settings(context_token_budget=50)
    compressor = ContextCompressor(embeddings=object(), kind="embeddings_filter", settings=settings)
    monkeypatch.setattr(compressor, "_build_compressor", lambda: _FakeEmbeddingsFilterCompressor())

    documents = _make_documents(5, words_per_doc=100)
    result = compressor.compress("some query", documents)

    assert result.token_count <= settings.context_token_budget
    assert result.truncated is True


def test_compress_keeps_all_docs_when_within_budget(monkeypatch) -> None:
    settings = Settings(context_token_budget=10_000)
    compressor = ContextCompressor(embeddings=object(), kind="embeddings_filter", settings=settings)
    monkeypatch.setattr(compressor, "_build_compressor", lambda: _FakeEmbeddingsFilterCompressor())

    documents = _make_documents(3, words_per_doc=20)
    result = compressor.compress("some query", documents)

    assert len(result.documents) == 3
    assert result.truncated is False
    assert result.token_count == count_tokens(result.context_text)


def test_compress_handles_empty_documents() -> None:
    settings = Settings(context_token_budget=100)
    compressor = ContextCompressor(embeddings=object(), kind="embeddings_filter", settings=settings)

    result = compressor.compress("some query", [])

    assert result.documents == []
    assert result.context_text == ""
    assert result.token_count == 0
    assert result.truncated is False


def test_compress_falls_back_to_raw_docs_when_compressor_filters_everything(monkeypatch) -> None:
    class _EmptyCompressor:
        def compress_documents(self, documents, query):
            return []

    settings = Settings(context_token_budget=10_000)
    compressor = ContextCompressor(embeddings=object(), kind="embeddings_filter", settings=settings)
    monkeypatch.setattr(compressor, "_build_compressor", lambda: _EmptyCompressor())

    documents = _make_documents(2, words_per_doc=10)
    result = compressor.compress("some query", documents)

    assert len(result.documents) == 2


def test_requires_llm_for_llm_extractor_kind() -> None:
    compressor = ContextCompressor(kind="llm_extractor")
    try:
        compressor._build_compressor()
        assert False, "expected ValueError"
    except ValueError:
        pass
