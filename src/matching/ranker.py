"""
Sorts MatchResult objects and applies optional threshold / diversity filters.

Usage:
    results = engine.score_batch(resumes, jd_text)
    ranked  = rank_candidates(results, top_k=10, min_score=0.3)
"""

from __future__ import annotations

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from src.matching.similarity_engine import MatchResult, SimilarityEngine


def rank_candidates(
    resumes: list[dict],
    jd_text: str,
    top_k: int = 10,
    min_score: float = 0.0,
    engine: SimilarityEngine | None = None,
) -> list[MatchResult]:
    """
    Score and rank a list of resumes against a job description.

    Parameters
    ----------
    resumes   : list of dicts, each with "text" (str) and optional "id" (str)
    jd_text   : full text of the job description
    top_k     : maximum number of results to return
    min_score : minimum final_score to include in results (0.0 = no filter)
    engine    : SimilarityEngine instance; creates one if not supplied

    Returns
    -------
    list[MatchResult] sorted descending by final_score, capped to top_k
    """
    if not resumes:
        logger.warning("rank_candidates called with empty resumes list")
        return []

    if engine is None:
        engine = SimilarityEngine()

    results = engine.score_batch(resumes, jd_text)

    # Filter by minimum score
    if min_score > 0.0:
        before = len(results)
        results = [r for r in results if r.final_score >= min_score]
        logger.info(
            f"Score filter (>={min_score}): {before} → {len(results)} candidates"
        )

    # Sort descending
    results.sort(key=lambda r: r.final_score, reverse=True)

    # Cap to top_k
    ranked = results[:top_k]

    logger.info(
        f"Ranked {len(resumes)} resumes → returning top {len(ranked)}.  "
        f"Best: {ranked[0].candidate_id} ({ranked[0].final_score:.3f})" if ranked else
        f"Ranked {len(resumes)} resumes → no results above threshold."
    )
    return ranked


def format_ranking_report(results: list[MatchResult]) -> str:
    """
    Return a plain-text leaderboard report for quick CLI inspection.
    """
    if not results:
        return "No candidates to report."

    lines = [
        "=" * 70,
        f"{'Rank':<5} {'Candidate':<30} {'Score':>6} {'Grade':>5} "
        f"{'Semantic':>9} {'Skills':>7}",
        "-" * 70,
    ]
    for i, r in enumerate(results, 1):
        name = r.candidate_id[:28] + ".." if len(r.candidate_id) > 30 else r.candidate_id
        lines.append(
            f"{i:<5} {name:<30} {r.final_score:>6.3f} {r.grade:>5} "
            f"{r.semantic_score:>9.3f} {r.skill_score:>7.3f}"
        )
    lines.append("=" * 70)

    # Skill gap summary for top candidate
    if results:
        top = results[0]
        lines += [
            f"\nTop candidate: {top.candidate_id}",
            f"  Matched skills ({len(top.matched_skills)}): "
            + ", ".join(top.matched_skills[:10])
            + (" ..." if len(top.matched_skills) > 10 else ""),
            f"  Missing skills ({len(top.missing_skills)}): "
            + ", ".join(top.missing_skills[:10])
            + (" ..." if len(top.missing_skills) > 10 else ""),
        ]

    return "\n".join(lines)