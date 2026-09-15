"""Centralized configuration loaded from environment variables.

All tunable parameters for chunking, retrieval, model routing, memory, and
cost tracking live here so the rest of the codebase never reads ``os.environ``
directly. Values are loaded once at import time via ``python-dotenv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all runtime configuration."""

    # --- API / connectivity ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    # --- Model routing ---
    cheap_model: str = field(default_factory=lambda: os.getenv("CHEAP_MODEL", "gpt-4o-mini"))
    strong_model: str = field(default_factory=lambda: os.getenv("STRONG_MODEL", "gpt-4o"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- Vector store ---
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "adaptive_rag_docs")
    )

    # --- Retrieval tuning ---
    simple_top_k_min: int = field(default_factory=lambda: _get_int("SIMPLE_TOP_K_MIN", 1))
    simple_top_k_max: int = field(default_factory=lambda: _get_int("SIMPLE_TOP_K_MAX", 3))
    complex_top_k_min: int = field(default_factory=lambda: _get_int("COMPLEX_TOP_K_MIN", 5))
    complex_top_k_max: int = field(default_factory=lambda: _get_int("COMPLEX_TOP_K_MAX", 8))
    mmr_fetch_k_multiplier: int = field(
        default_factory=lambda: _get_int("MMR_FETCH_K_MULTIPLIER", 4)
    )
    mmr_lambda: float = field(default_factory=lambda: _get_float("MMR_LAMBDA", 0.5))

    # --- Token budget enforcement ---
    context_token_budget: int = field(
        default_factory=lambda: _get_int("CONTEXT_TOKEN_BUDGET", 1500)
    )

    # --- Conversation memory ---
    memory_max_token_limit: int = field(
        default_factory=lambda: _get_int("MEMORY_MAX_TOKEN_LIMIT", 800)
    )

    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 100))
    use_semantic_chunker: bool = field(
        default_factory=lambda: _get_bool("USE_SEMANTIC_CHUNKER", False)
    )

    # --- Cost tracking ---
    tracking_log_path: str = field(
        default_factory=lambda: os.getenv("TRACKING_LOG_PATH", "./logs/usage.jsonl")
    )
    cheap_model_prompt_price_per_1k: float = field(
        default_factory=lambda: _get_float("CHEAP_MODEL_PROMPT_PRICE_PER_1K", 0.00015)
    )
    cheap_model_completion_price_per_1k: float = field(
        default_factory=lambda: _get_float("CHEAP_MODEL_COMPLETION_PRICE_PER_1K", 0.0006)
    )
    strong_model_prompt_price_per_1k: float = field(
        default_factory=lambda: _get_float("STRONG_MODEL_PROMPT_PRICE_PER_1K", 0.0025)
    )
    strong_model_completion_price_per_1k: float = field(
        default_factory=lambda: _get_float("STRONG_MODEL_COMPLETION_PRICE_PER_1K", 0.01)
    )


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` snapshot from the current environment."""

    return Settings()
