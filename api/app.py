"""
FastAPI application for the NLP Resume Screening service.

Endpoints
---------
POST /screen          Upload resumes + JD, get ranked candidates back
GET  /health          Liveness probe with model metadata
GET  /skills          List the skills vocabulary

Run with:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas import HealthResponse, ScreenResponse, ScreenSummary
from src.features.embedding_model import EmbeddingModel
from src.features.skill_extractor import SkillExtractor
from src.matching.ranker import rank_candidates
from src.matching.similarity_engine import SimilarityEngine
from src.preprocessing.resume_parser import ResumeParser


_parser:  ResumeParser | None     = None
_engine:  SimilarityEngine | None = None
_skill_ext: SkillExtractor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once on startup, clean up on shutdown."""
    global _parser, _engine, _skill_ext
    logger.info("Loading models …")
    _parser    = ResumeParser()
    _engine    = SimilarityEngine(spacy_model=os.getenv("SPACY_MODEL", "en_core_web_lg"))
    _skill_ext = _engine.skill_ext
    logger.info("All models ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="NLP Resume Screener",
    description=(
        "Rank resumes against a job description using semantic embeddings "
        "(sentence-transformers) and skill extraction (spaCy NER)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)



@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Liveness probe — returns model metadata."""
    model = EmbeddingModel.get()
    vocab = sum(
        len(v) for v in _skill_ext._skill_vocab.values()
    ) if _skill_ext else 0
    return HealthResponse(
        status="ok",
        model=model.model_name,
        embedding_dim=model.embedding_dim,
        skills_vocab_size=vocab,
    )


@app.get("/skills", tags=["meta"])
async def list_skills():
    """Return the full skills vocabulary grouped by category."""
    if _skill_ext is None:
        raise HTTPException(503, "Service not ready")
    return _skill_ext._skill_vocab


@app.post("/screen", response_model=ScreenSummary, tags=["screening"])
async def screen_resumes(
    job_description: str = Form(
        ...,
        description="Full text of the job description",
        min_length=50,
    ),
    top_k: int = Form(
        default=5,
        ge=1,
        le=50,
        description="Number of top candidates to return",
    ),
    min_score: float = Form(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum final score (0 = no filter)",
    ),
    files: list[UploadFile] = File(
        ...,
        description="Resume files (PDF, DOCX, or TXT)",
    ),
):
    """
    Upload one or more resume files and a job description.
    Returns ranked candidates with semantic + skill scores.
    """
    if _parser is None or _engine is None:
        raise HTTPException(503, "Models not loaded — try again shortly")

    if not files:
        raise HTTPException(400, "At least one resume file is required")

    resumes: list[dict] = []
    parse_errors: list[str] = []

    for upload in files:
        suffix = Path(upload.filename or "resume").suffix.lower()
        if suffix not in {".pdf", ".docx", ".doc", ".txt"}:
            parse_errors.append(
                f"{upload.filename}: unsupported format (use PDF, DOCX, TXT)"
            )
            continue

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                content = await upload.read()
                tmp.write(content)
                tmp_path = tmp.name

            parsed = _parser.parse(tmp_path)
            resumes.append({"id": upload.filename, "text": parsed.raw_text})

        except Exception as exc:
            logger.warning(f"Failed to parse {upload.filename}: {exc}")
            parse_errors.append(f"{upload.filename}: {exc}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not resumes:
        detail = "No resumes could be parsed."
        if parse_errors:
            detail += " Errors: " + "; ".join(parse_errors)
        raise HTTPException(422, detail)

    ranked = rank_candidates(
        resumes=resumes,
        jd_text=job_description,
        top_k=top_k,
        min_score=min_score,
        engine=_engine,
    )

    response_items = [
        ScreenResponse(
            rank=i + 1,
            candidate=r.candidate_id,
            final_score=r.final_score,
            grade=r.grade,
            semantic_score=r.semantic_score,
            skill_score=r.skill_score,
            matched_skills=r.matched_skills,
            missing_skills=r.missing_skills,
            extra_skills=r.extra_skills,
            resume_word_count=r.resume_word_count,
            processing_ms=r.processing_ms,
        )
        for i, r in enumerate(ranked)
    ]

    return ScreenSummary(
        total_candidates=len(resumes),
        top_k=top_k,
        min_score=min_score,
        results=response_items,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)