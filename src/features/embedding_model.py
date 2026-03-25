"""
Singleton wrapper around sentence-transformers for text encoding.

Model: all-MiniLM-L6-v2
  • 384-dimensional vectors
  • ~80 MB on disk
  • ~14,000 sentences/sec on CPU (M1 / modern Intel)
  • Embeddings are L2-normalised → cosine similarity == dot product

Usage:
    model = EmbeddingModel.get()
    vec   = model.encode_single("Python developer with 5 years experience")
    sims  = model.encode(["resume text 1", "resume text 2"])
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Optional

import numpy as np


# The model name can be overridden via env var RESUME_EMBED_MODEL
import os
MODEL_NAME: str = os.getenv("RESUME_EMBED_MODEL", "all-MiniLM-L6-v2")


class EmbeddingModel:
    """
    Singleton embedding model.

    Use EmbeddingModel.get() instead of __init__() to avoid
    re-loading the model on every call.
    """

    _instance: Optional["EmbeddingModel"] = None

    # Singleton factory

    @classmethod
    def get(cls, model_name: str = MODEL_NAME) -> "EmbeddingModel":
        """Return (or create) the singleton instance."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    # Init
    def __init__(self, model_name: str = MODEL_NAME):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: "
                "pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)   # ✅ ADD THIS LINE
        

    # Encoding

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised embedding vectors.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # unit vectors → cosine = dot product
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings

    @lru_cache(maxsize=2048)
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single string, with LRU caching.
        Identical texts (e.g. same JD queried multiple times) hit the cache.
        """
        return self.encode([text])[0]

    # Similarity utilities

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised vectors.
        Equivalent to dot product when vectors are unit-length.
        """
        return float(np.dot(a, b))

    def top_k_similar(
        self,
        query: str,
        candidates: list[str],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k most similar candidates to `query`.

        Returns
        -------
        list of (candidate_text, similarity_score), sorted descending
        """
        if not candidates:
            return []

        q_vec = self.encode_single(query)
        c_vecs = self.encode(candidates)
        scores = c_vecs @ q_vec  # batch dot product

        top_indices = np.argsort(scores)[::-1][:k]
        return [(candidates[i], float(scores[i])) for i in top_indices]


    # Utility

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        return f"EmbeddingModel(model={self.model_name}, dim={self.embedding_dim})"