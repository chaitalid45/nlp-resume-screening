# NLP Resume Screening

An end-to-end NLP pipeline that ranks resumes against a job description using
semantic embeddings (sentence-transformers) and skill extraction (spaCy NER).

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 2. Run the API
uvicorn api.app:app --reload

# 3. Screen resumes (curl example)
curl -X POST http://localhost:8000/screen \
  -F "job_description=We need a Python ML engineer with PyTorch and AWS experience." \
  -F "top_k=5" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx"
```

## Architecture

```
Resume PDFs/DOCXs
       │
       ▼
 ResumeParser         ← PyMuPDF + python-docx
       │
       ▼
 TextCleaner          ← noise removal, normalisation
       │
       ├─────────────────────────┐
       ▼                         ▼
 SkillExtractor           EmbeddingModel
 (spaCy NER + regex)      (all-MiniLM-L6-v2)
       │                         │
       └──────────┬──────────────┘
                  ▼
          SimilarityEngine
          score = 0.55×semantic + 0.45×skills
                  │
                  ▼
              Ranker  →  top-K MatchResults
                  │
                  ▼
             FastAPI /screen
```

## Scoring

| Signal          | Weight | Method                              |
|-----------------|--------|-------------------------------------|
| Semantic        | 55%    | Cosine similarity of embeddings     |
| Skill coverage  | 45%    | % of JD skills found in resume      |

Adjust `WEIGHTS` in `src/matching/similarity_engine.py` to tune.

## Project structure

nlp-resume-screening/
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   ├── job_descriptions/
│   ├── sample_resumes/
│   └── skills_db.json          ← curated skill vocabulary
├── src/
│   ├── preprocessing/
│   │   ├── resume_parser.py    ← PDF/DOCX → ParsedResume
│   │   ├── text_cleaner.py     ← normalise text
│   │   └── section_extractor.py
│   ├── features/
│   │   ├── skill_extractor.py  ← NER + keyword matching
│   │   ├── tfidf_vectorizer.py
│   │   └── embedding_model.py  ← sentence-transformers
│   ├── matching/
│   │   ├── similarity_engine.py← weighted scoring
│   │   └── ranker.py
│   └── models/
│       ├── classifier.py
│       └── ner_model.py
├── api/
│   ├── app.py                  ← FastAPI entrypoint
│   ├── routes.py
│   └── schemas.py
└── tests/
    ├── test_parser.py
    └── test_matching.py
