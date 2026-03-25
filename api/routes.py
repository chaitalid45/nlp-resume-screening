"""
Additional API routes beyond the core /screen endpoint.

Registered in app.py via:  app.include_router(router)

Routes:
    POST /classify      — classify a candidate as SHORTLIST / REJECT
    POST /extract-skills — extract skills from a block of text
    POST /parse-resume  — parse a single resume file (no scoring)
    GET  /jd-skills     — extract skills from a JD text (GET with body)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.features.skill_extractor import SkillExtractor
from src.models.classifier import ClassificationResult, RuleBasedClassifier
from src.preprocessing.resume_parser import ResumeParser

router = APIRouter()

# Shared instances (lightweight — ok to create per-import)
_skill_ext = SkillExtractor(spacy_model="")  # vocab-only for these routes
_parser    = ResumeParser()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ExtractSkillsRequest(BaseModel):
    text: str
    include_categories: bool = False


class ExtractSkillsResponse(BaseModel):
    skills: list[str]
    categories: dict[str, list[str]] = {}
    total: int


class ClassifyRequest(BaseModel):
    candidate_id: str
    final_score: float
    semantic_score: float
    skill_score: float
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    min_score: float = 0.50
    required_skills: list[str] = []


class ParseResponse(BaseModel):
    filename: str
    word_count: int
    email: str | None
    phone: str | None
    linkedin: str | None
    github: str | None
    sections: list[str]
    raw_text_preview: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/extract-skills", response_model=ExtractSkillsResponse, tags=["utilities"])
async def extract_skills(req: ExtractSkillsRequest):
    """
    Extract skills from any block of text (resume, JD, or freeform).
    Optionally group results by skill category.
    """
    skills = _skill_ext.extract(req.text)
    cats   = _skill_ext.get_categories(skills) if req.include_categories else {}
    return ExtractSkillsResponse(skills=skills, categories=cats, total=len(skills))


@router.post("/classify", tags=["screening"])
async def classify_candidate(req: ClassifyRequest) -> dict:
    """
    Apply rule-based shortlisting to a scored candidate.
    Useful for post-processing /screen results with custom thresholds.
    """
    from src.matching.similarity_engine import MatchResult

    result = MatchResult(
        candidate_id=req.candidate_id,
        semantic_score=req.semantic_score,
        skill_score=req.skill_score,
        final_score=req.final_score,
        matched_skills=req.matched_skills,
        missing_skills=req.missing_skills,
    )

    clf = RuleBasedClassifier(
        min_score=req.min_score,
        required_skills=req.required_skills,
    )
    decision: ClassificationResult = clf.predict(result)
    return {
        "candidate_id": req.candidate_id,
        **decision.to_dict(),
    }


@router.post("/parse-resume", response_model=ParseResponse, tags=["utilities"])
async def parse_resume(file: UploadFile = File(...)):
    """
    Parse a single resume file and return structured metadata.
    Does NOT score or rank — useful for previewing what the parser extracts.
    """
    suffix = Path(file.filename or "resume").suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc", ".txt"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        parsed = _parser.parse(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    preview = parsed.raw_text[:500].replace("\n", " ").strip()
    return ParseResponse(
        filename=file.filename or "",
        word_count=parsed.word_count,
        email=parsed.email,
        phone=parsed.phone,
        linkedin=parsed.linkedin,
        github=parsed.github,
        sections=list(parsed.sections.keys()),
        raw_text_preview=preview + ("…" if len(parsed.raw_text) > 500 else ""),
    )


@router.post("/jd-skills", response_model=ExtractSkillsResponse, tags=["utilities"])
async def extract_jd_skills(
    text: str = Body(..., embed=True, description="Job description text"),
    include_categories: bool = Body(False, embed=True),
):
    """Extract and categorise required skills from a job description."""
    skills = _skill_ext.extract(text)
    cats   = _skill_ext.get_categories(skills) if include_categories else {}
    return ExtractSkillsResponse(skills=skills, categories=cats, total=len(skills))