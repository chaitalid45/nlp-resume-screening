"""
Binary classifier that decides SHORTLIST / REJECT for a candidate
based on their MatchResult features.

Two modes:
  1. Rule-based  — fast, no training required, good for cold start
  2. ML-based    — logistic regression trained on labelled screening data

Usage (rule-based, zero setup):
    clf = RuleBasedClassifier(min_score=0.55, required_skills=["python"])
    decision = clf.predict(match_result)
    # → {"label": "SHORTLIST", "confidence": 0.82, "reasons": [...]}

Usage (ML-based, requires labelled data):
    clf = MLClassifier()
    clf.fit(match_results, labels)       # labels: list of 0/1
    decision = clf.predict(match_result)
    clf.save("models/classifier.joblib")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from src.matching.similarity_engine import MatchResult


# Shared result type


@dataclass
class ClassificationResult:
    label: Literal["SHORTLIST", "REJECT"]
    confidence: float           # 0.0 – 1.0
    reasons: list[str] = field(default_factory=list)

    @property
    def is_shortlisted(self) -> bool:
        return self.label == "SHORTLIST"

    def to_dict(self) -> dict:
        return {
            "label":      self.label,
            "confidence": round(self.confidence, 3),
            "reasons":    self.reasons,
        }


# Rule-based classifier (no training needed)

class RuleBasedClassifier:
    """
    Shortlists candidates that pass a set of configurable thresholds.

    Parameters
    ----------
    min_score       : minimum final_score to shortlist
    min_skill_score : minimum skill_score to shortlist
    required_skills : skills that MUST be present (all required)
    preferred_skills: skills where having any boosts confidence
    """

    def __init__(
        self,
        min_score: float = 0.50,
        min_skill_score: float = 0.30,
        required_skills: Optional[list[str]] = None,
        preferred_skills: Optional[list[str]] = None,
    ):
        self.min_score        = min_score
        self.min_skill_score  = min_skill_score
        self.required_skills  = [s.lower() for s in (required_skills or [])]
        self.preferred_skills = [s.lower() for s in (preferred_skills or [])]

    def predict(self, result: MatchResult) -> ClassificationResult:
        reasons: list[str] = []
        reject_reasons: list[str] = []

        # --- Hard gates -----------------------------------------------
        if result.final_score < self.min_score:
            reject_reasons.append(
                f"Overall score {result.final_score:.2f} below threshold {self.min_score}"
            )

        if result.skill_score < self.min_skill_score:
            reject_reasons.append(
                f"Skill coverage {result.skill_score:.0%} below threshold {self.min_skill_score:.0%}"
            )

        missing_required = [
            s for s in self.required_skills
            if s not in result.matched_skills
        ]
        if missing_required:
            reject_reasons.append(
                f"Missing required skills: {', '.join(missing_required)}"
            )

        # --- Positive signals -----------------------------------------
        if result.final_score >= 0.75:
            reasons.append(f"Strong overall match ({result.final_score:.2f})")
        if result.skill_score >= 0.70:
            reasons.append(f"High skill coverage ({result.skill_score:.0%})")

        preferred_found = [
            s for s in self.preferred_skills
            if s in result.matched_skills
        ]
        if preferred_found:
            reasons.append(f"Has preferred skills: {', '.join(preferred_found)}")

        # --- Decision -------------------------------------------------
        if reject_reasons:
            return ClassificationResult(
                label="REJECT",
                confidence=min(0.95, 0.5 + 0.15 * len(reject_reasons)),
                reasons=reject_reasons,
            )

        confidence = min(0.95, 0.5 + 0.1 * len(reasons) + result.final_score * 0.3)
        return ClassificationResult(
            label="SHORTLIST",
            confidence=round(confidence, 3),
            reasons=reasons or ["Passed all thresholds"],
        )


# ML-based classifier (logistic regression on MatchResult features)

class MLClassifier:
    """
    Logistic regression trained on (MatchResult → SHORTLIST/REJECT) labels.

    Features used:
        [final_score, semantic_score, skill_score,
         num_matched_skills, num_missing_skills, resume_word_count]
    """

    def __init__(self):
        self._clf = None
        self._threshold = 0.5


    def fit(
        self,
        results: list[MatchResult],
        labels: list[int],          # 1 = SHORTLIST, 0 = REJECT
        threshold: float = 0.5,
    ) -> "MLClassifier":
        """Train the classifier on labelled MatchResult objects."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            raise ImportError("scikit-learn is required: pip install scikit-learn")

        X = self._featurise(results)
        y = np.array(labels)

        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
        ])
        self._clf.fit(X, y)
        self._threshold = threshold

        train_acc = self._clf.score(X, y)
        logger.info(
            f"MLClassifier trained on {len(results)} samples. "
            f"Train accuracy: {train_acc:.2%}"
        )
        return self

    def predict(self, result: MatchResult) -> ClassificationResult:
        if self._clf is None:
            raise RuntimeError("Call fit() or load() before predict()")

        X = self._featurise([result])
        prob = self._clf.predict_proba(X)[0][1]  # P(SHORTLIST)
        label = "SHORTLIST" if prob >= self._threshold else "REJECT"

        return ClassificationResult(
            label=label,
            confidence=round(float(prob if label == "SHORTLIST" else 1 - prob), 3),
            reasons=[f"ML model probability: {prob:.2%}"],
        )

    def save(self, path: str) -> None:
        """Persist the trained model to disk."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required: pip install joblib")
        joblib.dump({"clf": self._clf, "threshold": self._threshold}, path)
        logger.info(f"Classifier saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MLClassifier":
        """Load a previously saved model."""
        import joblib
        data = joblib.load(path)
        instance = cls()
        instance._clf       = data["clf"]
        instance._threshold = data["threshold"]
        logger.info(f"Classifier loaded from {path}")
        return instance


    @staticmethod
    def _featurise(results: list[MatchResult]) -> np.ndarray:
        return np.array([
            [
                r.final_score,
                r.semantic_score,
                r.skill_score,
                len(r.matched_skills),
                len(r.missing_skills),
                min(r.resume_word_count / 1000.0, 3.0),  # normalise
            ]
            for r in results
        ])


# Make Optional available at module level
from typing import Optional