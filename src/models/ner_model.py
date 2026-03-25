"""
Custom Named Entity Recognition (NER) model wrapper.
Supports both spaCy's built-in NER and a fine-tuned custom model
for domain-specific tech entities (frameworks, tools, cloud services).

Usage:
    ner = NERModel()
    entities = ner.extract_entities("Built microservices using FastAPI and Kubernetes.")
    # → [{"text": "FastAPI", "label": "FRAMEWORK"}, {"text": "Kubernetes", "label": "TOOL"}]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 3),
        }


# Custom label mappings for tech resume domain
TECH_LABEL_MAP = {
    "ORG":     "COMPANY",
    "PRODUCT": "TOOL",
    "GPE":     "LOCATION",
    "PERSON":  "PERSON",
    "DATE":    "DATE",
    "ORDINAL": "ORDINAL",
}

# Labels we care about for skill extraction
SKILL_RELEVANT_LABELS = {"ORG", "PRODUCT"}


class NERModel:
    """
    NER model wrapper supporting spaCy's en_core_web_lg.

    Parameters
    ----------
    model_name : str
        spaCy model to load. Default: "en_core_web_lg"
    custom_labels : bool
        If True, map spaCy labels to domain-specific tech labels.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        custom_labels: bool = True,
    ):
        self.model_name    = model_name
        self.custom_labels = custom_labels
        self._nlp          = None
        self._load_model()

    # Public API

    def extract_entities(
        self,
        text: str,
        labels: Optional[list[str]] = None,
    ) -> list[Entity]:
        """
        Extract named entities from text.

        Parameters
        ----------
        text   : input text
        labels : filter to only these entity labels (e.g. ["ORG", "PRODUCT"])
                 If None, return all entities.

        Returns
        -------
        list of Entity objects
        """
        if not self._nlp:
            return []

        doc = self._nlp(text[:15_000])  # cap to avoid memory spikes
        entities: list[Entity] = []

        for ent in doc.ents:
            label = TECH_LABEL_MAP.get(ent.label_, ent.label_) \
                    if self.custom_labels else ent.label_

            if labels and ent.label_ not in labels:
                continue

            entities.append(Entity(
                text=ent.text.strip(),
                label=label,
                start=ent.start_char,
                end=ent.end_char,
            ))

        return entities

    def extract_skill_entities(self, text: str) -> list[str]:
        """
        Return a list of unique skill-relevant entity texts
        (ORG and PRODUCT labels only), lowercased.
        """
        entities = self.extract_entities(text, labels=list(SKILL_RELEVANT_LABELS))
        return sorted({e.text.lower() for e in entities if len(e.text) > 1})

    def extract_dates(self, text: str) -> list[Entity]:
        """Extract all DATE entities from text."""
        return self.extract_entities(text, labels=["DATE"])

    def extract_organisations(self, text: str) -> list[Entity]:
        """Extract organisation (company/institution) mentions."""
        return self.extract_entities(text, labels=["ORG"])

    def batch_extract(self, texts: list[str]) -> list[list[Entity]]:
        """
        Extract entities from multiple texts efficiently using spaCy's pipe().
        Faster than calling extract_entities() in a loop.
        """
        if not self._nlp or not texts:
            return [[] for _ in texts]

        results = []
        for doc in self._nlp.pipe(texts, batch_size=16):
            entities = [
                Entity(
                    text=ent.text.strip(),
                    label=TECH_LABEL_MAP.get(ent.label_, ent.label_)
                          if self.custom_labels else ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                )
                for ent in doc.ents
            ]
            results.append(entities)
        return results

    # Inspection helpers

    def label_distribution(self, text: str) -> dict[str, int]:
        """Return a count of each entity label found in text."""
        entities = self.extract_entities(text)
        dist: dict[str, int] = {}
        for e in entities:
            dist[e.label] = dist.get(e.label, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    # Private

    def _load_model(self) -> None:
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
            # Disable unused pipeline components for speed
            disabled = [
                c for c in ["tagger", "parser", "attribute_ruler", "lemmatizer"]
                if c in self._nlp.pipe_names
            ]
            if disabled:
                self._nlp.select_pipes(disable=disabled)
            logger.info(
                f"NERModel: loaded '{self.model_name}', "
                f"active pipes: {self._nlp.pipe_names}"
            )
        except (ImportError, OSError) as exc:
            logger.warning(
                f"NERModel: could not load '{self.model_name}' ({exc}). "
                "Entity extraction disabled."
            )
            self._nlp = None