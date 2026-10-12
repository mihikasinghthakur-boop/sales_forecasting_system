"""
app.py
======
Sales Forecast AI — Premium SaaS Dashboard
Streamlit frontend for the Data-Driven Sales Forecasting & Analysis System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    DATA_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_dataset,
    get_category_maps,
    load_raw_data,
    merge_tables,
    preprocess,
)
from src.eda import (
    get_summary_stats,
    plot_monthly_heatmap,
    plot_promotion_impact,
    plot_sales_by_store_type,
    plot_sales_trend,
    plot_top_families,
)
from src.model_trainer import BEST_MODEL_PATH, train_and_evaluate
from src.predictor import TRAINING_END_DATE, predict

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sales Forecast AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
/* ── Fonts ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Tokens ──────────────────────────────────────────────────────────────  */
:root {
  --bg-base:       #080e1c;
  --bg-surface:    #0f172a;
  --bg-elevated:   #1e293b;
  --bg-card:       rgba(30,41,59,0.7);
  --border:        rgba(99,102,241,0.18);
  --border-subtle: rgba(255,255,255,0.07);
  --indigo:        #6366f1;
  --violet:        #8b5cf6;
  --teal:          #14b8a6;
  --emerald:       #10b981;
  --amber:         #f59e0b;
  --coral:         #f43f5e;
  --lavender:      #a5b4fc;
  --sky:           #38bdf8;
  --text-primary:  #f1f5f9;
  --text-secondary:#94a3b8;
  --text-muted:    #475569;
}

/* ── Base ────────────────────────────────────────────────────────────────  */
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}
.stApp {
  background-color: var(--bg-base);
  background-image:
    radial-gradient(ellipse 80% 50% at 10% 0%, rgba(99,102,241,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 90% 100%, rgba(20,184,166,0.08) 0%, transparent 60%);
  background-attachment: fixed;
}

/* ── Main container ──────────────────────────────────────────────────────  */
.block-container {
  padding: 1.5rem 2.5rem 4rem !important;
  max-width: 1440px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────  */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0c1220 0%, #080e1c 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap: 4px; display: flex; flex-direction: column; }
[data-testid="stSidebar"] .stRadio [role="radio"] {
  display: none;
}
[data-testid="stSidebar"] .stRadio label {
  display: block;
  cursor: pointer;
  border-radius: 8px;
  padding: 0.55rem 0.9rem;
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--text-secondary) !important;
  transition: background 0.18s, color 0.18s;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(99,102,241,0.12);
  color: var(--lavender) !important;
}

/* ── Streamlit native metric cards ───────────────────────────────────────  */
[data-testid="stMetric"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.1rem 1.3rem !important;
  backdrop-filter: blur(12px);
  transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: rgba(99,102,241,0.4); }
[data-testid="stMetricLabel"]  { color: var(--text-secondary) !important; font-size: 0.76rem !important; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
[data-testid="stMetricValue"]  { color: var(--text-primary) !important; font-size: 1.7rem !important; font-weight: 700; letter-spacing: -0.02em; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────  */
.stButton > button {
  background: linear-gradient(135deg, var(--indigo) 0%, var(--violet) 100%);
  color: #fff;
  border: none;
  border-radius: 10px;
  padding: 0.65rem 1.8rem;
  font-weight: 600;
  font-size: 0.9rem;
  letter-spacing: 0.01em;
  box-shadow: 0 4px 20px rgba(99,102,241,0.35);
  transition: transform 0.18s, box-shadow 0.18s;
  width: 100%;
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(99,102,241,0.55);
}
.stButton > button:active { transform: translateY(0); }

/* ── Download button ─────────────────────────────────────────────────────  */
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  border: 1px solid rgba(99,102,241,0.4) !important;
  color: var(--lavender) !important;
  box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background: rgba(99,102,241,0.1) !important;
  border-color: var(--indigo) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────────  */
.stAlert { border-radius: 10px; }

/* ── Expander ────────────────────────────────────────────────────────────  */
[data-testid="stExpander"] {
  border: 1px solid var(--border-subtle) !important;
  border-radius: 12px !important;
  background: rgba(30,41,59,0.4) !important;
}
.streamlit-expanderHeader {
  color: var(--lavender) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
}

/* ── DataFrames ──────────────────────────────────────────────────────────  */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Sliders ─────────────────────────────────────────────────────────────  */
[data-testid="stSlider"] [role="slider"] { background: var(--indigo) !important; }

/* ── Toggle ──────────────────────────────────────────────────────────────  */
[data-testid="stToggle"] [data-checked="true"] { background: var(--indigo) !important; }

/* ── Input labels ────────────────────────────────────────────────────────  */
label { color: var(--text-secondary) !important; font-weight: 500; font-size: 0.85rem !important; }

/* ── HR ──────────────────────────────────────────────────────────────────  */
hr { border: none; border-top: 1px solid var(--border-subtle); margin: 1.5rem 0; }

/* ── Custom components ──────────────────────────────────────────────────── */

/* Brand header */
.brand-header {
  display: flex; align-items: center; gap: 0.7rem;
  padding: 1.4rem 1rem 0.8rem;
}
.brand-icon {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--indigo), var(--violet));
  border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
}
.brand-name {
  font-size: 1.05rem; font-weight: 700; color: var(--text-primary);
  letter-spacing: -0.01em;
}
.brand-tag {
  font-size: 0.68rem; color: var(--text-muted); margin-top: 1px;
}

/* Nav item */
.nav-section { font-size: 0.68rem; font-weight: 700; color: var(--text-muted);
  letter-spacing: 0.09em; text-transform: uppercase; padding: 0.9rem 1rem 0.3rem; }

/* Status chips */
.status-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 0.28rem 0.7rem; border-radius: 20px;
  font-size: 0.75rem; font-weight: 600;
}
.chip-ok   { background: rgba(16,185,129,0.12); color: var(--emerald); border: 1px solid rgba(16,185,129,0.25); }
.chip-info { background: rgba(99,102,241,0.12); color: var(--lavender); border: 1px solid rgba(99,102,241,0.25); }
.chip-warn { background: rgba(245,158,11,0.12); color: var(--amber); border: 1px solid rgba(245,158,11,0.25); }
.chip-err  { background: rgba(244,63,94,0.12);  color: var(--coral);  border: 1px solid rgba(244,63,94,0.25); }

/* Page hero */
.page-hero {
  padding: 0.2rem 0 1.5rem;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 1.8rem;
}
.page-hero h1 {
  font-size: 1.9rem; font-weight: 800;
  color: var(--text-primary);
  letter-spacing: -0.03em; margin: 0 0 0.3rem;
}
.page-hero .subtitle { color: var(--text-secondary); font-size: 0.95rem; }

/* Forecaster hero (larger, full-width) */
.fc-hero {
  background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.08) 50%, rgba(20,184,166,0.06) 100%);
  border: 1px solid rgba(99,102,241,0.22);
  border-radius: 20px;
  padding: 2.5rem 3rem 2rem;
  margin-bottom: 2rem;
  position: relative; overflow: hidden;
}
.fc-hero::before {
  content: '';
  position: absolute; top: -80px; right: -60px;
  width: 280px; height: 280px;
  background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
  pointer-events: none;
}
.fc-hero-eyebrow {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--indigo); margin-bottom: 0.5rem;
}
.fc-hero h1 {
  font-size: 2.4rem; font-weight: 800; letter-spacing: -0.04em;
  color: var(--text-primary); margin: 0 0 0.5rem;
  line-height: 1.15;
}
.fc-hero h1 span {
  background: linear-gradient(135deg, var(--indigo) 0%, var(--lavender) 50%, var(--sky) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.fc-hero p { color: var(--text-secondary); font-size: 1rem; margin: 0; max-width: 520px; }

/* Pill badges */
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.72rem; font-weight: 600; padding: 0.22rem 0.65rem;
  border-radius: 20px; border: 1px solid;
}
.badge-indigo { color: var(--lavender); border-color: rgba(99,102,241,0.35); background: rgba(99,102,241,0.1); }
.badge-teal   { color: #5eead4;         border-color: rgba(20,184,166,0.35); background: rgba(20,184,166,0.1); }
.badge-amber  { color: #fcd34d;         border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.1); }

/* Section label */
.section-label {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--indigo); margin-bottom: 0.6rem;
}

/* Parameters panel */
.params-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem 1.6rem;
  backdrop-filter: blur(10px);
}

/* Insight card */
.insight-card {
  background: rgba(20,184,166,0.07);
  border: 1px solid rgba(20,184,166,0.2);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  font-size: 0.88rem;
  color: var(--text-secondary);
  line-height: 1.55;
}
.insight-card strong { color: var(--teal); }

/* Info callout */
.callout {
  border-left: 3px solid;
  border-radius: 0 10px 10px 0;
  padding: 0.8rem 1rem;
  font-size: 0.86rem;
  color: var(--text-secondary);
  line-height: 1.55;
}
.callout-indigo { border-color: var(--indigo); background: rgba(99,102,241,0.07); }
.callout-amber  { border-color: var(--amber);  background: rgba(245,158,11,0.07); }
.callout-teal   { border-color: var(--teal);   background: rgba(20,184,166,0.07); }

/* Step label */
.step-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px; border-radius: 8px;
  background: rgba(99,102,241,0.2); color: var(--lavender);
  font-size: 0.78rem; font-weight: 700; margin-right: 8px;
}

/* Model metric card */
.model-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.2rem 1.5rem;
  text-align: center;
}
.model-card.best { border-color: rgba(99,102,241,0.5); box-shadow: 0 0 20px rgba(99,102,241,0.15); }
.model-card .mc-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 4px; }
.model-card .mc-name  { font-size: 0.92rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.6rem; }
.model-card .mc-value { font-size: 1.9rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.03em; }
.model-card .mc-unit  { font-size: 0.72rem; color: var(--text-muted); margin-top: 2px; }
.model-card .mc-best  { font-size: 0.7rem; font-weight: 700; color: var(--indigo); margin-top: 6px; }

/* Forecast result summary bar */
.fc-summary-bar {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.1rem 1.5rem;
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 1rem;
  margin-bottom: 1.5rem;
}
.fc-summary-item { text-align: center; }
.fc-summary-item .fcs-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); font-weight: 700; }
.fc-summary-item .fcs-value { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); }

/* Empty state */
.empty-state {
  text-align: center; padding: 3rem 1.5rem;
  color: var(--text-muted);
}
.empty-state svg { opacity: 0.3; margin-bottom: 1rem; }
.empty-state h3 { color: var(--text-secondary); font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem; }
.empty-state p  { font-size: 0.85rem; }

/* Dataset info pill (sidebar) */
.dataset-pill {
  margin: 0.8rem 0.8rem 0;
  padding: 0.75rem 1rem;
  border-radius: 10px;
  background: rgba(99,102,241,0.06);
  border: 1px solid rgba(99,102,241,0.12);
  font-size: 0.72rem;
}
.dataset-pill .dp-title { font-weight: 700; color: var(--lavender); margin-bottom: 0.3rem; font-size: 0.73rem; }
.dataset-pill .dp-row   { color: var(--text-muted); display: flex; align-items: center; gap: 6px; margin-top: 3px; }

/* Footer */
.app-footer {
  margin-top: 3rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--border-subtle);
  text-align: center;
  font-size: 0.72rem;
  color: var(--text-muted);
}
.app-footer a { color: var(--indigo); text-decoration: none; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

for _k, _v in [
    ("df", None), ("train_results", None), ("data_loaded", False),
    ("forecast", None), ("forecast_meta", {}),
    ("uploaded_df", None), ("upload_source", "builtin"),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-header">
      <div class="brand-icon">📈</div>
      <div>
        <div class="brand-name">Sales Forecast AI</div>
        <div class="brand-tag">AI-Powered Analytics</div>
      </div>
    </div>
    <hr style="margin:0.6rem 0.8rem;">
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        ["🔮  Forecaster", "📊  Dashboard & EDA", "🤖  Model Training"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:0.8rem 0.8rem;'>", unsafe_allow_html=True)

    # Status
    st.markdown('<div class="nav-section">System Status</div>', unsafe_allow_html=True)
    data_ok = DATA_DIR.exists() and (DATA_DIR / "train.csv").exists()
    json_ok = DATA_DIR.exists() and (DATA_DIR / "category_maps.json").exists()
    model_ok = BEST_MODEL_PATH.exists()

    if data_ok:
        st.markdown('<div class="status-chip chip-ok" style="margin:0.3rem 0.8rem;">✓ Full dataset ready</div>', unsafe_allow_html=True)
    elif json_ok:
        st.markdown('<div class="status-chip chip-info" style="margin:0.3rem 0.8rem;">● Lite mode — Forecaster on</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-chip chip-err" style="margin:0.3rem 0.8rem;">✕ Data folder missing</div>', unsafe_allow_html=True)

    if model_ok:
        st.markdown('<div class="status-chip chip-ok" style="margin:0.3rem 0.8rem;">✓ XGBoost model loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-chip chip-warn" style="margin:0.3rem 0.8rem;">⚠ No model — train first</div>', unsafe_allow_html=True)

    # Dataset info
    st.markdown("""
    <div class="dataset-pill">
      <div class="dp-title">Favorita Store Sales</div>
      <div class="dp-row">🗓️ Ecuador · 2013 – 2017</div>
      <div class="dp-row">🏪 54 stores · 33 families</div>
      <div class="dp-row">🤖 XGBoost · 14 features</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _plotly_layout(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply consistent Plotly dark theme to any figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        height=height,
        hoverlabel=dict(
            bgcolor="rgba(15,23,42,0.95)",
            bordercolor="#6366f1",
            font=dict(size=13, color="#f1f5f9"),
        ),
        legend=dict(
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="rgba(99,102,241,0.2)",
            borderwidth=1,
            font=dict(size=12),
        ),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                     tickfont=dict(size=11, color="#64748b"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                     tickfont=dict(size=11, color="#64748b"))
    return fig


def _check_data_dir() -> bool:
    if not DATA_DIR.exists():
        st.error(
            f"**Data directory not found:** `{DATA_DIR}`\n\n"
            "Place `train.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, "
            "and `transactions.csv` inside a `data/` folder at the project root.",
            icon="❌",
        )
        return False
    if not (DATA_DIR / "train.csv").exists():
        st.markdown("""
        <div class="callout callout-indigo">
          <b>train.csv is not available in this deployment.</b><br>
          The <b>Dashboard & EDA</b> and <b>Model Training</b> pages require the full
          Kaggle <em>Store Sales</em> dataset (~116 MB).<br><br>
          Download from
          <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data"
             target="_blank" style="color:#a5b4fc;">Kaggle — Store Sales Time Series Forecasting</a>
          and place <code>train.csv</code> in the <code>data/</code> folder.<br><br>
          💡 <b>The Forecaster page works right now — no data needed.</b>
        </div>
        """, unsafe_allow_html=True)
        return False
    return True


def _footer():
    st.markdown("""
    <div class="app-footer">
      Sales Forecast AI &nbsp;·&nbsp;
      Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> &amp; XGBoost &nbsp;·&nbsp;
      Dataset: <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting" target="_blank">Favorita Store Sales (Kaggle)</a>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — FORECASTER
# ═══════════════════════════════════════════════════════════════════════════

def page_forecaster():
    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="fc-hero">
      <div class="fc-hero-eyebrow">AI-Powered · XGBoost · Ecuador Retail</div>
      <h1>See the <span>Future</span> of Your Sales</h1>
      <p>Select a store, product family, and date window — get a data-driven
         daily sales forecast in seconds.</p>
      <div style="margin-top:1rem;display:flex;gap:8px;flex-wrap:wrap;">
        <span class="badge badge-indigo">📅 Any date range</span>
        <span class="badge badge-teal">🏪 54 stores</span>
        <span class="badge badge-teal">🛒 33 product families</span>
        <span class="badge badge-amber">⚡ Sub-second inference</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model guard ──────────────────────────────────────────────────────────
    if not BEST_MODEL_PATH.exists():
        st.markdown("""
        <div class="callout callout-amber">
          <b>No trained model found.</b><br>
          Go to <b>Model Training</b> and train the XGBoost pipeline first.
          You will need the full dataset for that step.
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Category maps ────────────────────────────────────────────────────────
    try:
        cats = get_category_maps()
    except Exception as exc:
        st.error(f"Could not load category metadata: {exc}")
        return

    # ── Parameters ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Configure Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="params-panel">', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Date Range**")
        dc1, dc2 = st.columns(2)
        with dc1:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp("2017-08-16").date(),
                min_value=pd.Timestamp("2013-01-01").date(),
                max_value=pd.Timestamp("2025-12-31").date(),
                key="fc_start",
            )
        with dc2:
            end_date = st.date_input(
                "End Date",
                value=pd.Timestamp("2017-09-15").date(),
                min_value=pd.Timestamp("2013-01-01").date(),
                max_value=pd.Timestamp("2025-12-31").date(),
                key="fc_end",
            )

        st.markdown("**Macro Assumption**")
        oil_price = st.number_input(
            "WTI Oil Price (USD/bbl)",
            min_value=20.0, max_value=200.0, value=47.26, step=0.5,
            key="fc_oil",
            help="Training-period median ≈ $47.26. Adjust for your scenario.",
        )

    with col_r:
        st.markdown("**Store & Product**")
        store_nbr = st.selectbox(
            "Store Number", options=cats["store_numbers"], index=0, key="fc_store"
        )
        families = cats["families"]
        fam_idx = families.index("BEVERAGES") if "BEVERAGES" in families else 0
        family = st.selectbox(
            "Product Family", options=families, index=fam_idx, key="fc_family"
        )

        st.markdown("**Promotion**")
        onpromotion = st.toggle(
            "Items currently on promotion",
            value=False, key="fc_promo",
            help="Toggle on if the selected family runs an active promotion.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Validation messages ──────────────────────────────────────────────────
    if end_date <= start_date:
        st.warning("End date must be after start date.", icon="⚠️")
        return

    n_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    forecast_start_ts = pd.Timestamp(start_date)

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        if forecast_start_ts > TRAINING_END_DATE:
            days_beyond = (forecast_start_ts - TRAINING_END_DATE).days
            st.markdown(
                f'<div class="callout callout-amber">'
                f'<b>Extrapolation:</b> forecast starts <b>{days_beyond} days</b> after training data ends '
                f'({TRAINING_END_DATE.date()}). Accuracy decreases with distance from the training window.'
                f'</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="callout callout-teal">'
                f'Forecasting <b>{n_days} days</b> within the training period '
                f'(data ends {TRAINING_END_DATE.date()}). '
                f'You can compare predictions against known actuals from train.csv.'
                f'</div>', unsafe_allow_html=True
            )

    with col_btn:
        run_forecast = st.button("🔮 Generate Forecast", key="btn_forecast")

    if run_forecast:
        with st.spinner(f"Generating {n_days}-day forecast for {family} · Store {store_nbr}…"):
            try:
                forecast_df = predict(
                    start_date=start_date, end_date=end_date,
                    store_nbr=store_nbr, family=family,
                    onpromotion=int(onpromotion), oil_price=oil_price,
                )
                st.session_state["forecast"] = forecast_df
                st.session_state["forecast_meta"] = {
                    "store_nbr": store_nbr, "family": family,
                    "onpromotion": onpromotion, "oil_price": oil_price,
                    "n_days": n_days,
                }
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
                return

    # ── Results ──────────────────────────────────────────────────────────────
    forecast_df = st.session_state.get("forecast")
    if forecast_df is None or forecast_df.empty:
        st.markdown("""
        <div class="empty-state">
          <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="1.5">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
          <h3>No forecast generated yet</h3>
          <p>Configure your parameters above and click <b>Generate Forecast</b></p>
        </div>
        """, unsafe_allow_html=True)
        _footer()
        return

    # KPI row
    meta = st.session_state.get("forecast_meta", {})
    _store = meta.get("store_nbr", store_nbr)
    _family = meta.get("family", family)
    _promo = meta.get("onpromotion", onpromotion)

    total_pred = forecast_df["predicted_sales"].sum()
    avg_pred   = forecast_df["predicted_sales"].mean()
    peak_idx   = forecast_df["predicted_sales"].idxmax()
    trough_idx = forecast_df["predicted_sales"].idxmin()
    peak       = forecast_df.loc[peak_idx]
    trough     = forecast_df.loc[trough_idx]
    peak_date  = peak["date"]
    trough_date = trough["date"]
    if hasattr(peak_date, "date"):
        peak_date = peak_date.date()
    if hasattr(trough_date, "date"):
        trough_date = trough_date.date()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Forecast Summary</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Predicted Sales",  f"{total_pred:,.0f}",   help="Sum of daily predicted units over the forecast window")
    k2.metric("Daily Average",          f"{avg_pred:,.1f}",     help="Mean predicted units per day")
    k3.metric("Peak Day",               str(peak_date),         f"{peak['predicted_sales']:,.0f} units")
    k4.metric("Lowest Day",             str(trough_date),       f"{trough['predicted_sales']:,.0f} units")

    # Insight
    weekend_mask = pd.to_datetime(forecast_df["date"]).dt.dayofweek >= 5
    wknd_avg = forecast_df.loc[weekend_mask, "predicted_sales"].mean() if weekend_mask.any() else 0
    wkdy_avg = forecast_df.loc[~weekend_mask, "predicted_sales"].mean() if (~weekend_mask).any() else 0
    if wknd_avg > 0 and wkdy_avg > 0:
        diff_pct = (wknd_avg - wkdy_avg) / wkdy_avg * 100
        direction = "higher" if diff_pct > 0 else "lower"
        st.markdown(
            f'<div class="insight-card" style="margin-top:0.8rem;">'
            f'<strong>Pattern insight:</strong> Weekend sales average '
            f'<strong>{abs(diff_pct):.1f}% {direction}</strong> than weekdays for '
            f'{_family} at Store {_store} in this forecast window. '
            f'Peak day is {peak_date} with <strong>{peak["predicted_sales"]:,.0f} units</strong>.'
            f'</div>', unsafe_allow_html=True
        )

    # Main forecast chart
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Forecast Chart</div>', unsafe_allow_html=True)
    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["predicted_sales"],
        mode="none", fill="tozeroy",
        fillcolor="rgba(99,102,241,0.08)", showlegend=False, hoverinfo="skip",
    ))
    # Main line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["predicted_sales"],
        mode="lines+markers",
        name="Predicted Sales",
        line=dict(color="#6366f1", width=2.5, shape="spline", smoothing=0.6),
        marker=dict(size=4, color="#818cf8", symbol="circle",
                    line=dict(color="#6366f1", width=1.5)),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: <b>%{y:,.1f} units</b><extra></extra>",
    ))
    # Training boundary marker
    if pd.Timestamp(start_date) <= TRAINING_END_DATE <= pd.Timestamp(end_date):
        fig.add_vline(
            x=TRAINING_END_DATE, line_dash="dash",
            line_color="rgba(245,158,11,0.5)", line_width=1.5,
            annotation_text="Training ends",
            annotation_font=dict(color="#f59e0b", size=11),
            annotation_position="top right",
        )

    # Peak & trough markers
    fig.add_trace(go.Scatter(
        x=[peak["date"]], y=[peak["predicted_sales"]],
        mode="markers", name="Peak",
        marker=dict(size=10, color="#10b981", symbol="triangle-up",
                    line=dict(color="#fff", width=1.5)),
        hovertemplate=f"<b>Peak Day</b><br>%{{x|%b %d}}: %{{y:,.1f}} units<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[trough["date"]], y=[trough["predicted_sales"]],
        mode="markers", name="Trough",
        marker=dict(size=10, color="#f43f5e", symbol="triangle-down",
                    line=dict(color="#fff", width=1.5)),
        hovertemplate=f"<b>Lowest Day</b><br>%{{x|%b %d}}: %{{y:,.1f}} units<extra></extra>",
    ))

    fig = _plotly_layout(fig, height=400)
    fig.update_layout(
        title=dict(
            text=f"<b>Sales Forecast</b>  <span style='font-size:13px;font-weight:400;color:#64748b;'>"
                 f"Store {_store} · {_family} · {'Promo ON' if _promo else 'No Promo'}</span>",
            font=dict(size=17, color="#e2e8f0"), x=0, xanchor="left",
        ),
        hovermode="x unified",
        showlegend=True,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Predicted Daily Sales (units)")
    st.plotly_chart(fig, use_container_width=True)

    # Day-of-week bar
    forecast_df["day_name"] = pd.to_datetime(forecast_df["date"]).dt.day_name()
    by_day = forecast_df.groupby("day_name")["predicted_sales"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    ).dropna()

    if not by_day.empty:
        col_dow, col_tbl = st.columns([1, 1], gap="large")
        with col_dow:
            weekend_days = ["Saturday", "Sunday"]
            bar_colors = ["#6366f1" if d not in weekend_days else "#14b8a6" for d in by_day.index]
            fig_dow = go.Figure(go.Bar(
                x=by_day.index, y=by_day.values,
                marker_color=bar_colors,
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Avg: %{y:,.1f} units<extra></extra>",
            ))
            fig_dow = _plotly_layout(fig_dow, height=280)
            fig_dow.update_layout(
                title=dict(text="<b>Sales by Day of Week</b>", font=dict(size=14, color="#e2e8f0"),
                           x=0, xanchor="left"),
                showlegend=False,
            )
            fig_dow.update_yaxes(title_text="Avg Predicted Sales")
            st.plotly_chart(fig_dow, use_container_width=True)

        with col_tbl:
            st.markdown('<div class="section-label" style="margin-top:0.2rem;">Daily Breakdown</div>', unsafe_allow_html=True)
            with st.expander("View full forecast table", expanded=True):
                display_df = forecast_df[["date","store_nbr","family","onpromotion","predicted_sales"]].copy()
                display_df["date"] = pd.to_datetime(display_df["date"]).dt.date
                display_df["onpromotion"] = display_df["onpromotion"].map({0:"No", 1:"Yes"})
                display_df.columns = ["Date","Store","Family","Promotion","Predicted Sales"]
                st.dataframe(
                    display_df,
                    use_container_width=True, hide_index=True, height=240,
                    column_config={
                        "Predicted Sales": st.column_config.NumberColumn(
                            "Predicted Sales",
                            format="%.1f",
                            help="Daily predicted unit sales from XGBoost model",
                        )
                    },
                )

    # Download
    csv_out = forecast_df[["date","store_nbr","family","onpromotion","predicted_sales"]].copy()
    csv_out["date"] = pd.to_datetime(csv_out["date"]).dt.date
    st.download_button(
        label="⬇️  Download Forecast CSV",
        data=csv_out.to_csv(index=False).encode(),
        file_name=f"forecast_store{_store}_{_family}_{start_date}_{end_date}.csv",
        mime="text/csv", key="dl_forecast",
    )

    # About expander
    with st.expander("ℹ️ Model details & limitations"):
        st.markdown(f"""
        **Model**: XGBoost Regressor — 500 estimators, learning rate 0.05, max depth 6  
        **Training window**: 2013-01-01 → **{TRAINING_END_DATE.date()}** (Favorita Ecuador)  
        **Features ({len(FEATURE_COLUMNS)})**: {", ".join(f"`{f}`" for f in FEATURE_COLUMNS)}  
        
        **Key assumptions for future dates**:
        - Transactions = store historical median (not actual future volume)
        - Holidays = none assumed (is_holiday = 0)
        - Oil price = your input (default: last known ≈ $47.26/bbl)
        
        **Limitations**: Point forecast only — no confidence intervals.  
        Extrapolation accuracy degrades beyond the training window.  
        Results reflect Ecuador 2013–2017 patterns and are not production forecasts.
        """)

    _footer()


# ───────────────────────────────────────────────────────────────────────────
# UPLOAD HELPERS
# ───────────────────────────────────────────────────────────────────────────

_REQUIRED_UPLOAD_COLS = {"date", "store_nbr", "family", "sales", "onpromotion"}
_SAMPLE_CSV = """date,store_nbr,family,sales,onpromotion
2017-01-01,1,BEVERAGES,1423.5,0
2017-01-02,1,BEVERAGES,1389.0,0
2017-01-03,1,BEVERAGES,1601.2,1
2017-01-04,1,GROCERY I,3214.8,0
2017-01-05,1,GROCERY I,2987.3,1
2017-01-06,2,BEVERAGES,1100.0,0
2017-01-07,2,GROCERY I,2650.0,0
"""


def _validate_upload(df_raw: pd.DataFrame) -> tuple[bool, str]:
    """Check uploaded CSV has required columns and valid types. Returns (ok, msg)."""
    missing = _REQUIRED_UPLOAD_COLS - set(df_raw.columns)
    if missing:
        return False, f"Missing required columns: {sorted(missing)}"
    if df_raw.empty:
        return False, "Uploaded file is empty."
    try:
        df_copy = df_raw.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
        bad_dates = int(df_copy["date"].isna().sum())
        if bad_dates > 0:
            return False, f"{bad_dates} rows have unparseable date values."
    except Exception as exc:
        return False, f"Could not parse 'date' column: {exc}"
    df_copy["sales"] = pd.to_numeric(df_copy["sales"], errors="coerce")
    bad_sales = int(df_copy["sales"].isna().sum())
    if bad_sales > 0:
        return False, f"{bad_sales} rows have non-numeric 'sales' values."
    return True, ""


def _preprocess_upload(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Lightweight preprocess for user-uploaded CSVs (no external CSVs available)."""
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
    df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0).astype(int)
    df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").fillna(0).astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["family_enc"] = df["family"].astype("category").cat.codes
    for col in ["type_enc", "cluster", "dcoilwtico", "transactions", "is_holiday", "type"]:
        if col not in df.columns:
            df[col] = 0
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD & EDA
# ═══════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown("""
    <div class="page-hero">
      <h1>Dashboard &amp; Exploratory Analysis</h1>
      <div class="subtitle">Historical patterns, data quality, and interactive visualisations.
      Analyse the built-in Favorita dataset or upload your own compatible CSV.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset source selector ──────────────────────────────────────────
    st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
    src_cols = st.columns([2, 3])
    with src_cols[0]:
        data_source = st.radio(
            "Choose dataset",
            ["Built-in Favorita dataset", "Upload my own CSV"],
            key="dash_source",
            label_visibility="collapsed",
        )

    df = None
    with src_cols[1]:
        if data_source == "Upload my own CSV":
            st.markdown(
                '<div class="callout callout-indigo">'
                '<b>Required columns:</b> <code>date</code>, <code>store_nbr</code>, '
                '<code>family</code>, <code>sales</code>, <code>onpromotion</code><br>'
                'Date format: YYYY-MM-DD &nbsp;|&nbsp; Sales: numeric &nbsp;|&nbsp; '
                'On Promotion: 0 or 1'
                '</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download sample template CSV",
                data=_SAMPLE_CSV.encode(),
                file_name="sales_template.csv",
                mime="text/csv",
                key="dl_template",
            )
            uploaded = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                key="dash_upload",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                try:
                    raw = pd.read_csv(uploaded, low_memory=False)
                    ok, msg = _validate_upload(raw)
                    if not ok:
                        st.error(f"Upload validation failed: {msg}")
                    else:
                        df = _preprocess_upload(raw)
                        st.session_state["uploaded_df"] = df
                        st.success(
                            f"✅ Uploaded {len(df):,} rows · "
                            f"{df['date'].min().date()} → {df['date'].max().date()}"
                        )
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")
            else:
                df = st.session_state.get("uploaded_df")
                if df is None:
                    st.info("Upload a CSV file above to begin analysis.")
                    _footer()
                    return

    # ── Load built-in data if needed ───────────────────────────────────
    if data_source == "Built-in Favorita dataset":
        if not _check_data_dir():
            _footer()
            return
        if not st.session_state["data_loaded"] or st.session_state["df"] is None:
            with st.spinner("Loading and merging CSV files — this takes ~15 seconds…"):
                try:
                    df = build_dataset()
                    st.session_state["df"] = df
                    st.session_state["data_loaded"] = True
                except Exception as exc:
                    st.error(f"Failed to load data: {exc}")
                    return
        else:
            df = st.session_state["df"]

    if df is None:
        _footer()
        return

    # ── KPI row ───────────────────────────────────────────────────────
    total_sales  = df["sales"].sum()
    avg_daily    = df.groupby("date")["sales"].sum().mean()
    n_stores     = df["store_nbr"].nunique()
    n_families   = df["family"].nunique() if "family" in df.columns else "N/A"
    n_rows       = len(df)
    date_min     = df["date"].min()
    date_max     = df["date"].max()
    date_span    = (date_max - date_min).days

    # YoY growth (last year vs. penultimate year)
    df_yr = df.copy()
    df_yr["year"] = pd.to_datetime(df_yr["date"]).dt.year
    yr_sales = df_yr.groupby("year")["sales"].sum()
    years = sorted(yr_sales.index)
    if len(years) >= 2:
        yoy = (yr_sales[years[-1]] - yr_sales[years[-2]]) / yr_sales[years[-2]] * 100
        yoy_str = f"{yoy:+.1f}% YoY"
    else:
        yoy_str = "N/A"

    # ── KPI cards ───────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Sales", f"{total_sales/1e6:.2f}M",
              help="Aggregate unit sales across all rows")
    m2.metric("Daily Average", f"{avg_daily:,.0f}",
              help="Mean total daily sales (all stores/families combined)")
    m3.metric("Stores", n_stores)
    m4.metric("Product Families", n_families)
    m5.metric("YoY Growth", yoy_str,
              help=f"{years[-1]} vs {years[-2]}" if len(years) >= 2 else "")

    # Insight banner
    top_family = df.groupby("family")["sales"].sum().idxmax() if "family" in df.columns else "N/A"
    top_store  = df.groupby("store_nbr")["sales"].sum().idxmax() if "store_nbr" in df.columns else "N/A"
    st.markdown(
        f'<div class="insight-card" style="margin:0.8rem 0 1.5rem;">'
        f'<strong>Dataset snapshot:</strong> {n_rows:,} records spanning '
        f'<strong>{date_span} days</strong> ({date_min.date()} → {date_max.date()}). '
        f'Top-selling family: <strong>{top_family}</strong>. '
        f'Highest-volume store: <strong>Store {top_store}</strong>. '
        f'{"Last full year saw " + yoy_str + " vs. prior year." if yoy_str != "N/A" else ""}'
        f'</div>', unsafe_allow_html=True
    )

    # ── Sales Trend ────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Historical Trends</div>', unsafe_allow_html=True)
    with st.spinner("Rendering sales trend…"):
        try:
            st.plotly_chart(plot_sales_trend(df), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render sales trend: {exc}")

    with st.spinner("Rendering seasonality heatmap…"):
        try:
            st.plotly_chart(plot_monthly_heatmap(df), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render heatmap: {exc}")

    # ── 2-column charts ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:0.5rem;">Store &amp; Product Breakdown</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2, gap="large")
    with ca:
        with st.spinner("Rendering store-type chart…"):
            try:
                st.plotly_chart(plot_sales_by_store_type(df), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not render store chart: {exc}")
    with cb:
        with st.spinner("Rendering promotion impact…"):
            try:
                st.plotly_chart(plot_promotion_impact(df), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not render promotion chart: {exc}")

    # ── Top families ─────────────────────────────────────────────────────────────────────────
    if "family" in df.columns:
        st.markdown('<div class="section-label" style="margin-top:0.5rem;">Product Family Rankings</div>', unsafe_allow_html=True)
        _max_fam = df["family"].nunique()
        top_n = st.slider("Show top N families", 3, min(33, _max_fam), min(15, _max_fam), key="top_n_slider")
        with st.spinner("Rendering family rankings…"):
            try:
                st.plotly_chart(plot_top_families(df, top_n=top_n), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not render family chart: {exc}")

    # ── Data quality ───────────────────────────────────────────────────────────────────
    with st.expander("🔍 Data Quality Report"):
        total_rows = len(df)
        available_feat = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing_counts = df[available_feat + ["sales"]].isna().sum()
        missing_df2 = missing_counts[missing_counts > 0].reset_index()
        missing_df2.columns = ["Column", "Missing Count"]
        missing_df2["Missing %"] = (missing_df2["Missing Count"] / total_rows * 100).round(2)

        dq1, dq2, dq3, dq4 = st.columns(4)
        dq1.metric("Total Rows", f"{total_rows:,}")
        dq2.metric("Zero-Sales Rows", f"{(df['sales'] == 0).sum():,}",
                   help="Rows where sales = 0 (may be legitimate: closed stores, no demand)")
        dq3.metric("Negative-Sales Rows", f"{(df['sales'] < 0).sum():,}",
                   help="May represent returns or adjustments — preserved, not removed")
        dq4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

        if len(missing_df2) > 0:
            st.warning("Some columns have missing values (see below).")
            st.dataframe(missing_df2, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No missing values in key columns after preprocessing.")

    with st.expander("📐 Summary Statistics"):
        try:
            stats = get_summary_stats(df)
            st.dataframe(stats, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not compute summary statistics: {exc}")

    with st.expander("🗂️ Data Schema & Preview"):
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Non-Null": [int(df[c].notna().sum()) for c in df.columns],
            "Unique Values": [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        st.caption(f"Showing first 8 of {len(df):,} rows")
        st.dataframe(df.head(8), use_container_width=True)

    # ── Cleaned data download ────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1rem;">Export</div>', unsafe_allow_html=True)
    export_cols = [c for c in ["date", "store_nbr", "family", "sales", "onpromotion"] + FEATURE_COLUMNS
                   if c in df.columns]
    export_df = df[export_cols].copy()
    if "date" in export_df.columns:
        export_df["date"] = pd.to_datetime(export_df["date"]).dt.date
    st.download_button(
        label="⬇️ Download Cleaned Dataset (CSV)",
        data=export_df.to_csv(index=False).encode(),
        file_name="sales_cleaned.csv",
        mime="text/csv",
        key="dl_cleaned",
    )

    _footer()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def page_training():
    st.markdown("""
    <div class="page-hero">
      <h1>Model Training</h1>
      <div class="subtitle">Preprocess data, train the XGBoost pipeline, and evaluate performance against the linear baseline.</div>
    </div>
    """, unsafe_allow_html=True)

    if not _check_data_dir():
        _footer()
        return

    # Pipeline info
    with st.expander("ℹ️ About the ML pipeline", expanded=False):
        st.markdown(f"""
        | | Baseline | Primary |
        |---|---|---|
        | **Model** | LinearRegression | **XGBRegressor** |
        | **Preprocessing** | StandardScaler | None needed |
        | **Trees** | — | 500 |
        | **Learning rate** | — | 0.05 |
        | **Max depth** | — | 6 |
        | **Subsample** | — | 0.8 |
        | **colsample_bytree** | — | 0.8 |
        | **random_state** | — | 42 |

        **Split**: Chronological — the last `test_fraction` of unique dates forms the test set.
        This prevents data leakage that random shuffling would cause for time-series data.

        **Features ({len(FEATURE_COLUMNS)})**: {', '.join(f'`{f}`' for f in FEATURE_COLUMNS)}

        **Target**: `sales` — unit sales per day, per store, per product family.
        """)

    # Step 1
    st.markdown(
        '<span class="step-badge">1</span> <b style="color:#e2e8f0;">Load & Preprocess Data</b>',
        unsafe_allow_html=True,
    )
    st.caption("Merges train.csv ← stores ← oil (ffill) ← holidays (deduped) ← transactions.")

    if st.button("🔄 Load & Preprocess Data", key="btn_load"):
        with st.spinner("Loading and merging all tables — this may take 15–30 s…"):
            try:
                df = build_dataset()
                st.session_state["df"] = df
                st.session_state["data_loaded"] = True
                st.success(
                    f"✅  {len(df):,} rows · {len(df.columns)} columns · "
                    f"{df['date'].min().date()} → {df['date'].max().date()}"
                )
                st.dataframe(df[FEATURE_COLUMNS].describe().round(3), use_container_width=True)
            except Exception as exc:
                st.error(f"Failed to load data: {exc}")
                return

    st.markdown("<hr>", unsafe_allow_html=True)

    # Step 2
    st.markdown(
        '<span class="step-badge">2</span> <b style="color:#e2e8f0;">Train the Pipeline</b>',
        unsafe_allow_html=True,
    )

    if st.session_state["df"] is None:
        st.markdown(
            '<div class="callout callout-amber">Load the data first (Step 1) before training.</div>',
            unsafe_allow_html=True,
        )
        _footer()
        return

    df_loaded = st.session_state["df"]
    test_frac = st.slider(
        "Test-set fraction (chronological holdout)",
        min_value=0.05, max_value=0.30, value=0.15, step=0.01,
        help="Last X% of dates reserved as the unseen test set.",
    )
    sorted_dates = sorted(df_loaded["date"].unique())
    n_test = max(1, int(len(sorted_dates) * test_frac))
    cutoff  = sorted_dates[-n_test]
    st.markdown(
        f'<div class="callout callout-indigo">'
        f'Train on all dates before <b>{pd.Timestamp(cutoff).date()}</b> · '
        f'Test on last <b>{n_test}</b> unique dates · '
        f'Evaluation: RMSE & MAE on the held-out period.'
        f'</div>', unsafe_allow_html=True
    )

    if st.button("🚀 Train Models", key="btn_train"):
        log_msgs: list[str] = []
        log_box = st.empty()

        def _log(msg: str):
            log_msgs.append(msg)
            log_box.code("\n".join(log_msgs))

        with st.spinner("Training — LinearRegression then XGBoost (1–3 min)…"):
            try:
                results = train_and_evaluate(
                    df_loaded, test_fraction=test_frac, progress_callback=_log
                )
                st.session_state["train_results"] = results
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return

        st.success("✅  Training complete! Best model saved to `models/best_model.joblib`.")
        log_box.empty()

    # Results
    results = st.session_state.get("train_results")
    if not results:
        _footer()
        return

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Evaluation Results</div>', unsafe_allow_html=True)

    metrics   = results["metrics"]
    best_name = results["best_model"]

    # ── Four metric cards per model: RMSE · MAE · R² · MAPE ─────────────────
    for mname, m in metrics.items():
        is_best = mname == best_name
        trophy  = "  🏆" if is_best else ""
        border  = "border:1px solid rgba(99,102,241,0.5);box-shadow:0 0 16px rgba(99,102,241,0.12);" if is_best else ""
        st.markdown(
            f'<div class="model-card" style="margin-bottom:0.7rem;{border}">'
            f'<div class="mc-label" style="text-align:left;margin-bottom:0.5rem;">{mname}{trophy}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE",  f"{m['rmse']:.2f}",  help="Root Mean Squared Error (sales units) — lower is better")
        c2.metric("MAE",   f"{m['mae']:.2f}",   help="Mean Absolute Error (sales units) — lower is better")
        r2_val = m.get("r2", float("nan"))
        c3.metric("R²",    f"{r2_val:.4f}" if not (r2_val != r2_val) else "—",
                  help="Coefficient of determination — 1.0 is perfect, higher is better")
        mape_val = m.get("mape", float("nan"))
        c4.metric("MAPE",  f"{mape_val:.1f}%" if not (mape_val != mape_val) else "—",
                  help="Mean Absolute Percentage Error — lower is better; zero-sales rows excluded")

    # Interpretation callout
    st.markdown(
        '<div class="callout callout-indigo" style="margin:1rem 0 0.5rem;">'
        '<b>How to read these metrics</b> — All measured on the <em>chronological hold-out test set</em> '
        '(no data leakage). '
        '<b>RMSE</b>: average error in sales units, penalises large misses more. '
        '<b>MAE</b>: average absolute error, robust to outliers. '
        '<b>R²</b>: fraction of variance explained (1.0 = perfect). '
        '<b>MAPE</b>: percentage error (zero-sales rows excluded to avoid division by zero). '
        f'Evaluation period: dates after <b>{pd.Timestamp(results["cutoff_date"]).date()}</b> '
        f'({results["test_size"]:,} test rows).'
        '</div>', unsafe_allow_html=True
    )

    # Split info row
    s1, s2, s3 = st.columns(3)
    s1.metric("Train Rows", f"{results['train_size']:,}")
    s2.metric("Test Rows",  f"{results['test_size']:,}")
    s3.metric("Test Cutoff", str(results["cutoff_date"].date()))

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Metric bar chart (RMSE + MAE side-by-side) ────────────────────────────
    st.markdown('<div class="section-label">Error Metric Comparison</div>', unsafe_allow_html=True)
    model_names = list(metrics.keys())
    fig_bar = go.Figure()
    for metric_name, color in [("rmse", "#6366f1"), ("mae", "#14b8a6")]:
        vals = [metrics[m][metric_name] for m in model_names]
        fig_bar.add_trace(go.Bar(
            x=model_names, y=vals,
            name=metric_name.upper(),
            marker_color=[color if m == best_name else f"{color}66" for m in model_names],
            marker_line_width=0,
            hovertemplate=f"<b>%{{x}}</b><br>{metric_name.upper()}: %{{y:.4f}}<extra></extra>",
        ))
    fig_bar = _plotly_layout(fig_bar, height=260)
    fig_bar.update_layout(
        barmode="group", showlegend=True,
        title=dict(text="<b>RMSE vs MAE — Model Comparison</b>",
                   font=dict(size=14, color="#e2e8f0"), x=0, xanchor="left"),
        yaxis_title="Error (sales units)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Actual vs Predicted chart ─────────────────────────────────────────────
    if "test_dates" in results and "y_test" in results and "y_pred_best" in results:
        st.markdown('<div class="section-label">Actual vs. Predicted — Test Set</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="callout callout-teal" style="margin-bottom:0.8rem;">'
            f'Showing the <b>{best_name}</b> model predictions against actual sales '
            f'on the <b>hold-out test set</b> ({results["test_size"]:,} rows). '
            'Each point is one store/family/day observation. '
            'Perfect predictions would lie exactly on the diagonal line.'
            '</div>', unsafe_allow_html=True
        )
        y_actual = results["y_test"]
        y_pred   = results["y_pred_best"]
        # Sample for performance (scatter with >1M pts is slow)
        max_pts = 20_000
        if len(y_actual) > max_pts:
            import numpy as _np_local
            idx = _np_local.random.default_rng(42).choice(len(y_actual), max_pts, replace=False)
            y_actual_plot = y_actual[idx]
            y_pred_plot   = y_pred[idx]
            sample_note   = f" (random sample of {max_pts:,}/{len(y_actual):,} points)"
        else:
            y_actual_plot = y_actual
            y_pred_plot   = y_pred
            sample_note   = ""

        avp_fig = go.Figure()
        avp_fig.add_trace(go.Scatter(
            x=y_actual_plot, y=y_pred_plot,
            mode="markers",
            name=f"Predictions{sample_note}",
            marker=dict(color="#6366f1", size=3, opacity=0.4),
            hovertemplate="Actual: %{x:,.1f}<br>Predicted: %{y:,.1f}<extra></extra>",
        ))
        # Perfect prediction diagonal
        diag_max = float(max(y_actual_plot.max(), y_pred_plot.max())) * 1.05
        avp_fig.add_trace(go.Scatter(
            x=[0, diag_max], y=[0, diag_max],
            mode="lines", name="Perfect prediction",
            line=dict(color="#14b8a6", width=1.5, dash="dot"),
        ))
        avp_fig = _plotly_layout(avp_fig, height=380)
        avp_fig.update_layout(
            title=dict(text=f"<b>Actual vs. Predicted Sales — {best_name}</b>",
                       font=dict(size=14, color="#e2e8f0"), x=0, xanchor="left"),
            xaxis_title="Actual Sales (units)",
            yaxis_title="Predicted Sales (units)",
        )
        st.plotly_chart(avp_fig, use_container_width=True)

    # ── Download model ────────────────────────────────────────────────────────
    if BEST_MODEL_PATH.exists():
        with open(BEST_MODEL_PATH, "rb") as fh:
            st.download_button(
                label="⬇️  Download best_model.joblib",
                data=fh, file_name="best_model.joblib",
                mime="application/octet-stream", key="dl_model",
            )

    _footer()


# ═══════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════

if page == "🔮  Forecaster":
    page_forecaster()
elif page == "📊  Dashboard & EDA":
    page_dashboard()
elif page == "🤖  Model Training":
    page_training()
