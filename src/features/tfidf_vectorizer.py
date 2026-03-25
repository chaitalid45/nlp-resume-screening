"""
TF-IDF based similarity as a lightweight alternative (or supplement) to
sentence embeddings.  Useful for keyword-dense JDs where exact term overlap
matters as much as semantic proximity.

Usage:
    tv = TFIDFVectorizer()
    tv.fit(corpus_texts)
    score = tv.similarity(resume_text, jd_text)
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as _SKVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFIDFVectorizer:
    """
    Wrapper around sklearn TfidfVectorizer with convenience methods
    for resume-to-JD similarity scoring.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 15_000,
        min_df: int = 1,
    ):
        self._vectorizer = _SKVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            stop_words="english",
            sublinear_tf=True,  # use 1 + log(tf) to dampen frequency
        )
        self._fitted = False

    # Fit / transform

    def fit(self, texts: list[str]) -> "TFIDFVectorizer":
        """Fit the vocabulary on a corpus of texts."""
        self._vectorizer.fit(texts)
        self._fitted = True
        logger.info(
            f"TF-IDF fitted on {len(texts)} documents, "
            f"vocab size: {len(self._vectorizer.vocabulary_)}"
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts into TF-IDF vectors."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        return self._vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    # Similarity

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Return cosine similarity between two texts.
        Fits on both texts if the vectorizer hasn't been fitted yet.
        """
        if not self._fitted:
            self.fit([text_a, text_b])

        vecs = self._vectorizer.transform([text_a, text_b])
        score = cosine_similarity(vecs[0], vecs[1])[0][0]
        return round(float(score), 4)

    def batch_similarity(
        self, query: str, candidates: list[str]
    ) -> list[float]:
        """
        Return cosine similarities between `query` and each candidate.
        More efficient than calling similarity() N times.
        """
        if not self._fitted:
            self.fit([query] + candidates)

        all_vecs = self._vectorizer.transform([query] + candidates)
        q_vec = all_vecs[0]
        c_vecs = all_vecs[1:]
        scores = cosine_similarity(q_vec, c_vecs)[0]
        return [round(float(s), 4) for s in scores]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def top_terms(self, text: str, n: int = 20) -> list[tuple[str, float]]:
        """Return the top-n TF-IDF terms for a given text."""
        if not self._fitted:
            self.fit([text])
        vec = self._vectorizer.transform([text]).toarray()[0]
        names = self._vectorizer.get_feature_names_out()
        indices = np.argsort(vec)[::-1][:n]
        return [(names[i], round(float(vec[i]), 4)) for i in indices if vec[i] > 0]