"""
Extracts skills from resume / JD text using two complementary strategies:
  1. spaCy NER  — catches proper-noun technology names (ORG, PRODUCT)
  2. Vocabulary matching — regex scan against a curated skills DB JSON

Usage:
    extractor = SkillExtractor()
    skills = extractor.extract("Experienced Python developer with AWS and Docker skills.")
    gap    = extractor.match_jd_skills(resume_skills, jd_skills)
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default path to skills vocabulary (resolved relative to this file)
_DEFAULT_SKILLS_DB = (
    Path(__file__).resolve().parents[2] / "data" / "skills_db.json"
)


class SkillExtractor:
    """
    Dual-strategy skill extractor.

    Parameters
    ----------
    skills_db_path : str | Path | None
        Path to skills_db.json.  Falls back to the bundled DB if not given.
    spacy_model : str
        spaCy model name.  Set to "" to skip NER and use vocab-only mode.
    """

    def __init__(
        self,
        skills_db_path: Optional[str | Path] = None,
        spacy_model: str = "en_core_web_lg",
    ):
        db_path = Path(skills_db_path) if skills_db_path else _DEFAULT_SKILLS_DB
        self._skill_vocab: dict[str, list[str]] = self._load_db(db_path)
        self._flat_vocab: set[str] = {
            s.lower()
            for cat in self._skill_vocab.values()
            for s in cat
        }

        # Attempt to load spaCy; gracefully degrade if unavailable
        self._nlp = None
        if spacy_model:
            try:
                import spacy
                self._nlp = spacy.load(spacy_model)
            except (ImportError, OSError) as exc:
                logger.warning(
                    f"spaCy model '{spacy_model}' unavailable ({exc}). "
                    "Falling back to vocab-only matching."
                )

    # Public API

    def extract(self, text: str) -> list[str]:
        """
        Return a sorted list of unique skills found in `text`.
        Combines NER entities (if spaCy is available) with vocab matching.
        """
        ner_skills  = self._extract_via_ner(text)  if self._nlp  else set()
        kw_skills   = self._extract_via_vocab(text)
        combined    = sorted(ner_skills | kw_skills)
        return combined

    def match_jd_skills(
        self,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> dict:
        """
        Compare resume skills against JD-required skills.

        Returns
        -------
        dict with keys:
            matched  – skills present in both
            missing  – skills in JD but not in resume
            extra    – skills in resume but not in JD
            score    – float 0-1, proportion of JD skills matched
        """
        rs = {s.lower() for s in resume_skills}
        js = {s.lower() for s in jd_skills}

        matched = sorted(rs & js)
        missing = sorted(js - rs)
        extra   = sorted(rs - js)
        score   = round(len(matched) / max(len(js), 1), 4)

        return {
            "matched": matched,
            "missing": missing,
            "extra":   extra,
            "score":   score,
        }

    def get_categories(self, skills: list[str]) -> dict[str, list[str]]:
        """
        Group a list of skills by their category from the skills DB.

        Returns a dict like {"ml_frameworks": ["pytorch", "keras"], ...}
        """
        cats: dict[str, list[str]] = {}
        for skill in skills:
            cat = self._get_category(skill.lower())
            if cat:
                cats.setdefault(cat, []).append(skill)
        return cats

    # Private helpers

    def _extract_via_ner(self, text: str) -> set[str]:
        """Use spaCy NER to find named entities that are in the skill vocab."""
        doc = self._nlp(text[:10000])  # cap to avoid memory issues
        return {
            ent.text.lower()
            for ent in doc.ents
            if ent.label_ in ("ORG", "PRODUCT", "GPE")
            and ent.text.lower() in self._flat_vocab
        }

    def _extract_via_vocab(self, text: str) -> set[str]:
        """
        Scan text for every skill in the vocabulary using whole-word regex.
        Multi-word phrases (e.g. 'machine learning') are checked as-is.
        """
        text_lower = text.lower()
        found: set[str] = set()
        for skill in self._flat_vocab:
            pattern = rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])"
            if re.search(pattern, text_lower):
                found.add(skill)
        return found

    def _get_category(self, skill: str) -> Optional[str]:
        for cat, items in self._skill_vocab.items():
            if skill in {s.lower() for s in items}:
                return cat
        return None

    @staticmethod
    def _load_db(path: Path) -> dict[str, list[str]]:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        
        return data