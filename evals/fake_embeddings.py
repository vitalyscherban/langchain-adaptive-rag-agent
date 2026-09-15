"""Deterministic, offline embeddings for tests and demos.

Real embedding models require a live API call. To let the eval script (and
tests) run fully offline/deterministically, this module implements a simple
hashing-based bag-of-words embedder: each word is hashed into one of
``n_dims`` buckets, and the resulting bucket-count vector is L2-normalized.
Documents that share more vocabulary end up with higher cosine similarity,
which is enough signal for MMR re-ranking and embeddings-based filtering to
behave sensibly in demos, without needing network access or an API key.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings

_WORD_RE = re.compile(r"[a-z0-9]+")


class DeterministicHashEmbeddings(Embeddings):
    """Offline, deterministic stand-in for a real embeddings model."""

    def __init__(self, n_dims: int = 256) -> None:
        self.n_dims = n_dims

    def _embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.n_dims
        for word in _WORD_RE.findall(text.lower()):
            bucket = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % self.n_dims
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)
