"""
streamlit_app.py  —  NLP Resume Screener
-----------------------------------------
Simple 3-step workflow:
  1. Upload resumes (PDF / DOCX / TXT)
  2. Paste a job description
  3. Click Screen → ranked results with scores & skill tags

Run:
    streamlit run streamlit_app.py
"""

import os, sys, time, tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ · NLP Resume Screener",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ───────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, p, div, span {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── App background ──────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(160deg, #f0f4ff 0%, #faf5ff 50%, #f0fdf4 100%);
    min-height: 100vh;
}
.block-container { padding-top: 2.5rem !important; max-width: 760px !important; }

/* ── Hero header ─────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0891b2 100%);
    border-radius: 20px;
    padding: 2rem 2.2rem 1.8rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 20px 60px rgba(79,70,229,0.25);
}
.hero h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px !important;
    color: white !important;
    letter-spacing: -0.02em;
}
.hero p { margin: 0; opacity: 0.88; font-size: 0.95rem; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 12px;
    color: white;
    letter-spacing: 0.02em;
}

/* ── Step labels ─────────────────────────────────────────────────────── */
.step-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.6rem 0 0.6rem;
}
.step-num {
    width: 30px; height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(79,70,229,0.35);
}
.step-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1e1b4b;
}

/* ── Streamlit inputs ────────────────────────────────────────────────── */
.stTextArea textarea {
    border: 2px solid #e0e7ff !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    background: white !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.08) !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #c7d2fe !important;
    border-radius: 14px !important;
    background: rgba(238,242,255,0.6) !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366f1 !important;
}

/* ── Run button ──────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 8px 24px rgba(79,70,229,0.35) !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(79,70,229,0.45) !important;
}

/* ── Metric cards (summary row) ──────────────────────────────────────── */
[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 14px !important;
    padding: 1rem 1.1rem !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.07) !important;
}
[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: #1e1b4b !important;
}

/* ── Expander ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #4f46e5 !important;
    font-size: 0.88rem !important;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid #e0e7ff !important; margin: 1.5rem 0 !important; }

/* ── Result card ─────────────────────────────────────────────────────── */
.rcard {
    background: white;
    border: 1.5px solid #e0e7ff;
    border-radius: 16px;
    padding: 1.1rem 1.3rem 1rem;
    margin-bottom: 12px;
    box-shadow: 0 4px 20px rgba(99,102,241,0.06);
    transition: box-shadow 0.2s;
}
.rcard:hover { box-shadow: 0 8px 32px rgba(99,102,241,0.14); }

/* ── Score bar ───────────────────────────────────────────────────────── */
.sbar-bg {
    background: #f3f4f6;
    border-radius: 99px;
    height: 7px;
    overflow: hidden;
    flex: 1;
}
.sbar-fill { height: 100%; border-radius: 99px; }

/* ── Skill tags ──────────────────────────────────────────────────────── */
.stag {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px 3px 2px 0;
    letter-spacing: 0.01em;
}
.stag-green  { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
.stag-red    { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }

/* ── Grade pill ──────────────────────────────────────────────────────── */
.gpill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 800;
    padding: 2px 11px;
    border-radius: 20px;
    letter-spacing: 0.03em;
}
.gA { background: #dcfce7; color: #15803d; }
.gB { background: #dbeafe; color: #1d4ed8; }
.gC { background: #fef9c3; color: #92400e; }
.gD { background: #ffedd5; color: #9a3412; }
.gF { background: #fee2e2; color: #b91c1c; }

/* ── Download button ─────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: white !important;
    border: 2px solid #e0e7ff !important;
    border-radius: 12px !important;
    color: #4f46e5 !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.18s !important;
}
.stDownloadButton > button:hover {
    background: #eef2ff !important;
    border-color: #6366f1 !important;
}

/* ── Misc ────────────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading NLP models — takes ~10 sec first time…")
def get_engine():
    from src.matching.similarity_engine import SimilarityEngine
    return SimilarityEngine(spacy_model="")   # vocab-only; no spaCy install needed


@st.cache_resource(show_spinner=False)
def get_parser():
    from src.preprocessing.resume_parser import ResumeParser
    return ResumeParser()


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_files(files):
    parser, resumes, errors = get_parser(), [], []
    for f in files:
        suffix = Path(f.name).suffix.lower()
        if suffix not in {".pdf", ".docx", ".doc", ".txt"}:
            errors.append(f"⚠ {f.name}: unsupported format"); continue
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read()); tmp_path = tmp.name
            parsed = parser.parse(tmp_path)
            resumes.append({"id": f.name, "text": parsed.raw_text})
        except Exception as e:
            errors.append(f"⚠ {f.name}: {e}")
        finally:
            try: os.unlink(tmp_path)
            except: pass
    return resumes, errors


def score_color(s):
    if s >= 0.70: return "#16a34a"
    if s >= 0.50: return "#d97706"
    return "#dc2626"

def grade_cls(g):
    return {"A":"gA","B":"gB","C":"gC","D":"gD","F":"gF"}.get(g,"gF")


# ═══════════════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🎯 ResumeIQ</h1>
  <p>AI-powered resume screening using NLP embeddings &amp; skill extraction.</p>
  <div style="margin-top:12px;">
    <span class="hero-badge">⚡ Instant ranking</span>
    <span class="hero-badge">📄 PDF · DOCX · TXT</span>
    <span class="hero-badge">🧠 Semantic + Skill scoring</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Upload
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-label">
  <div class="step-num">1</div>
  <div class="step-title">Upload Resumes</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop PDF, DOCX, or TXT resume files here",
    accept_multiple_files=True,
    type=["pdf", "docx", "doc", "txt"],
    label_visibility="collapsed",
)

if uploaded:
    names = "  ·  ".join(f.name for f in uploaded)
    st.markdown(
        f"<div style='font-size:0.8rem;color:#6366f1;font-weight:600;"
        f"margin-top:6px;'>✓ {len(uploaded)} file(s): {names}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Job Description
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-label">
  <div class="step-num">2</div>
  <div class="step-title">Paste Job Description</div>
</div>
""", unsafe_allow_html=True)

jd_text = st.text_area(
    "job description",
    height=155,
    placeholder=(
        "e.g.  Senior Python ML Engineer — PyTorch, TensorFlow, AWS SageMaker, "
        "Docker, Kubernetes, FastAPI, PostgreSQL, MLflow. 5+ years experience. "
        "NLP or computer vision background preferred."
    ),
    label_visibility="collapsed",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Options (collapsed)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

with st.expander("⚙️  Options — Top K  ·  Scoring weights"):
    oc1, oc2, oc3 = st.columns([1, 1.4, 1])
    with oc1:
        top_k = st.number_input(
            "Top K results", min_value=1, max_value=50, value=10, step=1
        )
    with oc2:
        w_sem = st.slider(
            "Semantic weight", 0.1, 0.9, 0.55, 0.05,
            help="Share given to embedding-based semantic similarity (skill weight = 1 − this)"
        )
    with oc3:
        w_sk = round(1.0 - w_sem, 2)
        st.metric("Skill weight", f"{w_sk:.2f}")
    st.markdown(
        f"<div style='font-size:0.78rem;color:#6b7280;margin-top:4px;'>"
        f"final&nbsp;score&nbsp;=&nbsp;<b style='color:#6366f1'>{w_sem:.0%}</b>&nbsp;×&nbsp;semantic"
        f"&nbsp;+&nbsp;<b style='color:#10b981'>{w_sk:.0%}</b>&nbsp;×&nbsp;skill&nbsp;coverage"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Run
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
run_clicked = st.button("🚀  Run Screening", type="primary", use_container_width=True)

if not run_clicked:
    st.stop()

# ── Validation ────────────────────────────────────────────────────────────────
if not uploaded:
    st.error("⚠️  Please upload at least one resume file.")
    st.stop()
if not jd_text or len(jd_text.split()) < 8:
    st.error("⚠️  Job description is too short — enter at least 8 words.")
    st.stop()

# ── Parse ─────────────────────────────────────────────────────────────────────
with st.spinner("📄  Parsing resume files…"):
    resumes, errors = parse_files(uploaded)

for msg in errors:
    st.warning(msg)

if not resumes:
    st.error("No resumes could be parsed. Check file formats.")
    st.stop()

# ── Score ─────────────────────────────────────────────────────────────────────
with st.spinner(f"🧠  Scoring {len(resumes)} resume(s)…"):
    engine = get_engine()
    engine.weights = {"semantic": float(w_sem), "skill": float(w_sk)}
    from src.matching.ranker import rank_candidates
    t0     = time.perf_counter()
    ranked = rank_candidates(resumes, jd_text, top_k=int(top_k), engine=engine)
    ms     = (time.perf_counter() - t0) * 1000

if not ranked:
    st.warning("No results returned. Try uploading more resumes or lowering min_score in Options.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("""<hr>""", unsafe_allow_html=True)

st.markdown(
    f"<h3 style='color:#1e1b4b;font-weight:800;margin-bottom:2px;'>"
    f"Results</h3>"
    f"<p style='color:#6b7280;font-size:0.84rem;margin-top:0;'>"
    f"Showing top {len(ranked)} of {len(resumes)} · "
    f"screened in <b style='color:#4f46e5'>{ms:.0f} ms</b> · "
    f"scoring: {w_sem:.0%} semantic + {w_sk:.0%} skills</p>",
    unsafe_allow_html=True,
)

# ── Summary metrics ───────────────────────────────────────────────────────────
avg_score = sum(r.final_score for r in ranked) / len(ranked)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Uploaded",   len(resumes))
m2.metric("Showing",    len(ranked))
m3.metric("Best score", f"{ranked[0].final_score:.3f}")
m4.metric("Avg score",  f"{avg_score:.3f}")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# ── Bar chart ─────────────────────────────────────────────────────────────────
names  = [r.candidate_id.split(".")[0][:24] for r in ranked]
sem_c  = np.array([r.semantic_score * w_sem for r in ranked])
sk_c   = np.array([r.skill_score    * w_sk  for r in ranked])
finals = sem_c + sk_c

fig, ax = plt.subplots(figsize=(8, max(3.5, len(ranked) * 0.48 + 1.2)))
fig.patch.set_facecolor("white")
ax.set_facecolor("#f8faff")

y = np.arange(len(ranked))
b1 = ax.barh(y, sem_c, height=0.52, color="#6366f1",
             label=f"Semantic ×{w_sem:.2f}", zorder=3)
b2 = ax.barh(y, sk_c, left=sem_c, height=0.52, color="#10b981",
             label=f"Skill ×{w_sk:.2f}", zorder=3)

# Score labels on bars
for i, (s, f) in enumerate(zip(finals, finals)):
    ax.text(f + 0.01, i, f"{f:.3f}", va="center", ha="left",
            fontsize=8.5, fontweight="600", color="#374151")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9.5, color="#374151")
ax.set_xlabel("Score contribution", fontsize=9, color="#6b7280")
ax.set_xlim(0, 1.18)
ax.tick_params(colors="#9ca3af", length=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#e0e7ff")
ax.spines["bottom"].set_color("#e0e7ff")
ax.grid(axis="x", color="#e0e7ff", linewidth=0.8, zorder=0)

legend = ax.legend(loc="lower right", fontsize=9, framealpha=1,
                   edgecolor="#e0e7ff", facecolor="white")
ax.set_title("Candidate ranking", fontsize=11, fontweight="700",
             color="#1e1b4b", pad=12, loc="left")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ── Result cards ──────────────────────────────────────────────────────────────
for i, r in enumerate(ranked):
    sc  = score_color(r.final_score)
    gc  = grade_cls(r.grade)
    pct_sem  = r.semantic_score * 100
    pct_skill= r.skill_score    * 100

    matched_tags = "".join(
        f"<span class='stag stag-green'>{s}</span>"
        for s in r.matched_skills[:14]
    )
    missing_tags = "".join(
        f"<span class='stag stag-red'>{s}</span>"
        for s in r.missing_skills[:10]
    )

    skills_html = ""
    if r.matched_skills:
        skills_html += (
            f"<div style='font-size:0.7rem;font-weight:600;color:#6b7280;"
            f"text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 5px;'>"
            f"✅ Matched skills ({len(r.matched_skills)})</div>"
            + matched_tags
        )
    if r.missing_skills:
        skills_html += (
            f"<div style='font-size:0.7rem;font-weight:600;color:#6b7280;"
            f"text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 5px;'>"
            f"❌ Missing skills ({len(r.missing_skills)})</div>"
            + missing_tags
        )

    st.markdown(f"""
<div class="rcard">

  <!-- ── Header row ── -->
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
    <span style="font-size:1.25rem;font-weight:800;color:#c7d2fe;min-width:32px;
                 font-variant-numeric:tabular-nums;">#{i+1}</span>
    <span style="flex:1;font-size:0.97rem;font-weight:700;color:#1e1b4b;
                 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
      {r.candidate_id}
    </span>
    <span class="gpill {gc}">{r.grade}</span>
    <span style="font-size:1.5rem;font-weight:800;color:{sc};
                 font-variant-numeric:tabular-nums;letter-spacing:-0.02em;">
      {r.final_score:.3f}
    </span>
  </div>

  <!-- ── Semantic bar ── -->
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
    <span style="font-size:0.73rem;font-weight:600;color:#6b7280;min-width:108px;">
      🔵 Semantic
    </span>
    <div class="sbar-bg">
      <div class="sbar-fill" style="width:{pct_sem:.1f}%;
           background:linear-gradient(90deg,#6366f1,#818cf8);"></div>
    </div>
    <span style="font-size:0.78rem;font-weight:700;color:#6366f1;
                 min-width:40px;text-align:right;">{r.semantic_score:.3f}</span>
  </div>

  <!-- ── Skill bar ── -->
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
    <span style="font-size:0.73rem;font-weight:600;color:#6b7280;min-width:108px;">
      🟢 Skill coverage
    </span>
    <div class="sbar-bg">
      <div class="sbar-fill" style="width:{pct_skill:.1f}%;
           background:linear-gradient(90deg,#10b981,#34d399);"></div>
    </div>
    <span style="font-size:0.78rem;font-weight:700;color:#10b981;
                 min-width:40px;text-align:right;">{r.skill_score:.3f}</span>
  </div>

  {skills_html}

  <!-- ── Footer meta ── -->
  <div style="font-size:0.7rem;color:#9ca3af;margin-top:10px;padding-top:8px;
              border-top:1px solid #f3f4f6;">
    {r.resume_word_count} words · processed in {r.processing_ms:.0f} ms
  </div>

</div>
""", unsafe_allow_html=True)


# ── Download CSV ──────────────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

csv_rows = [{
    "rank":           i + 1,
    "candidate":      r.candidate_id,
    "final_score":    r.final_score,
    "grade":          r.grade,
    "semantic_score": r.semantic_score,
    "skill_score":    r.skill_score,
    "matched_skills": ", ".join(r.matched_skills),
    "missing_skills": ", ".join(r.missing_skills),
    "word_count":     r.resume_word_count,
} for i, r in enumerate(ranked)]

csv_bytes = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇  Download results as CSV",
    data=csv_bytes,
    file_name="resume_screening_results.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown(
    "<div style='text-align:center;font-size:0.72rem;color:#9ca3af;margin-top:24px;'>"
    "ResumeIQ · sentence-transformers + skill extraction · built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)