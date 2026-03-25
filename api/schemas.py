"""
Pydantic request / response models for the FastAPI endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ScreenResponse(BaseModel):
    """Single candidate result returned by /screen."""
    rank: int
    candidate: str
    final_score: float = Field(ge=0.0, le=1.0)
    grade: str
    semantic_score: float = Field(ge=0.0, le=1.0)
    skill_score: float = Field(ge=0.0, le=1.0)
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    resume_word_count: int
    processing_ms: float


class ScreenSummary(BaseModel):
    """Summary wrapper returned by /screen."""
    total_candidates: int
    top_k: int
    min_score: float
    results: list[ScreenResponse]


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    embedding_dim: int
    skills_vocab_size: int