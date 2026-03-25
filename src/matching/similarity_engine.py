"""
The core scoring engine that combines:
  - Semantic similarity  (sentence-transformer cosine, weight=0.55)
  - Skill coverage score (proportion of JD skills in resume, weight=0.45)

Produces a MatchResult dataclass for each resume-JD pair.

Usage:
    engine = SimilarityEngine()
    result = engine.score(resume_text, jd_text, candidate_id="alice.pdf")
    print(result.final_score, result.matched_skills)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from src.features.embedding_model import EmbeddingModel
from src.features.skill_extractor import SkillExtractor
from src.preprocessing.text_cleaner import TextCleaner


# Result dataclass

@dataclass
class MatchResult:
    """Scoring result for a single resume against a JD."""
    candidate_id: str

    # Component scores (all in range 0.0 – 1.0)
    semantic_score: float   # cosine similarity of sentence embeddings
    skill_score: float      # % of JD skills found in resume
    final_score: float      # weighted blend

    # Skill breakdown
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills:   list[str] = field(default_factory=list)

    # Metadata
    resume_word_count: int = 0
    processing_ms: float = 0.0

    @property
    def grade(self) -> str:
        """Letter grade based on final score."""
        if self.final_score >= 0.80:
            return "A"
        if self.final_score >= 0.65:
            return "B"
        if self.final_score >= 0.50:
            return "C"
        if self.final_score >= 0.35:
            return "D"
        return "F"

    def to_dict(self) -> dict:
        return {
            "candidate_id":    self.candidate_id,
            "final_score":     self.final_score,
            "grade":           self.grade,
            "semantic_score":  self.semantic_score,
            "skill_score":     self.skill_score,
            "matched_skills":  self.matched_skills,
            "missing_skills":  self.missing_skills,
            "extra_skills":    self.extra_skills,
            "resume_word_count": self.resume_word_count,
            "processing_ms":   round(self.processing_ms, 1),
        }


# Engine

# Scoring weights — must sum to 1.0
WEIGHTS: dict[str, float] = {
    "semantic": 0.55,
    "skill":    0.45,
}


class SimilarityEngine:
    """
    Scores a resume against a job description.

    Parameters
    ----------
    weights : dict | None
        Override default WEIGHTS dict, e.g. {"semantic": 0.7, "skill": 0.3}
    spacy_model : str
        Passed to SkillExtractor; set to "" to skip NER.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        spacy_model: str = "en_core_web_lg",
    ):
        self.weights = weights or WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

        self.embedder  = EmbeddingModel.get()
        self.skill_ext = SkillExtractor(spacy_model=spacy_model)
        self.cleaner   = TextCleaner()
        logger.info(f"SimilarityEngine ready. Weights: {self.weights}")

    # Public API

    def score(
        self,
        resume_text: str,
        jd_text: str,
        candidate_id: str = "",
    ) -> MatchResult:
        """
        Score a single resume against a job description.

        Parameters
        ----------
        resume_text : raw text from the resume
        jd_text     : raw text of the job description
        candidate_id: filename or unique identifier for logging

        Returns
        -------
        MatchResult
        """
        t0 = time.perf_counter()

        # --- Embedding-based semantic score ----------------------------
        r_clean = self.cleaner.clean_for_embedding(resume_text)
        j_clean = self.cleaner.clean_for_embedding(jd_text)

        r_vec = self.embedder.encode_single(r_clean)
        j_vec = self.embedder.encode_single(j_clean)
        semantic = float(np.dot(r_vec, j_vec))          # cosine (normalised)
        semantic = max(0.0, min(1.0, semantic))         # clamp to [0, 1]

        # --- Skill-based score ----------------------------------------
        r_skills = self.skill_ext.extract(resume_text)
        j_skills = self.skill_ext.extract(jd_text)
        skill_gap = self.skill_ext.match_jd_skills(r_skills, j_skills)

        # --- Weighted blend -------------------------------------------
        final = (
            self.weights["semantic"] * semantic
            + self.weights["skill"]  * skill_gap["score"]
        )
        final = round(final, 4)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = MatchResult(
            candidate_id=candidate_id,
            semantic_score=round(semantic, 4),
            skill_score=skill_gap["score"],
            final_score=final,
            matched_skills=skill_gap["matched"],
            missing_skills=skill_gap["missing"],
            extra_skills=skill_gap["extra"],
            resume_word_count=len(resume_text.split()),
            processing_ms=elapsed_ms,
        )

        logger.info(
            f"Scored '{candidate_id}': "
            f"final={final:.3f}  sem={semantic:.3f}  skill={skill_gap['score']:.3f}  "
            f"matched={len(skill_gap['matched'])}  missing={len(skill_gap['missing'])}  "
            f"({elapsed_ms:.0f}ms)"
        )
        return result

    def score_batch(
        self,
        resumes: list[dict],
        jd_text: str,
    ) -> list[MatchResult]:
        """
        Score multiple resumes against the same JD.

        Parameters
        ----------
        resumes : list of dicts with keys "text" (required) and "id" (optional)
        jd_text : job description text

        Returns
        -------
        list[MatchResult] — unsorted (use ranker.rank_candidates to sort)
        """
        if not resumes:
            return []

        # Pre-encode the JD once so encode_single's LRU cache hits every time
        jd_clean = self.cleaner.clean_for_embedding(jd_text)
        _ = self.embedder.encode_single(jd_clean)  # warm cache

        results = []
        for r in resumes:
            text = r.get("text", "")
            rid  = r.get("id", "")
            results.append(self.score(text, jd_text, candidate_id=rid))

        return results