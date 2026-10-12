"""
app.py
======
Streamlit frontend for the Data-Driven Sales Forecasting & Analysis System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure the project root is on the path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    DATA_DIR,
    FEATURE_COLUMNS,
    build_dataset,
    get_category_maps,
    load_raw_data,
)
from src.eda import (
    get_summary_stats,
    plot_promotion_impact,
    plot_sales_by_store_type,
    plot_sales_trend,
    plot_top_families,
)
from src.model_trainer import BEST_MODEL_PATH, train_and_evaluate
from src.predictor import predict

# ─────────────────────────────────────────────────────────────────────────────
# Page Config & Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root overrides ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-attachment: fixed;
    }

    /* ── Main container ── */
    .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem;
        max-width: 1400px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1535 0%, #0f0c29 100%);
        border-right: 1px solid rgba(167,139,250,0.15);
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #c4b5fd !important;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(167,139,250,0.08);
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        backdrop-filter: blur(8px);
    }
    [data-testid="stMetricLabel"] { color: #a78bfa !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.6rem; font-weight: 700; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #a78bfa);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(108,99,255,0.4);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108,99,255,0.6);
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .section-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }

    /* ── Info / warning banners ── */
    .stAlert {
        border-radius: 10px;
    }

    /* ── DataFrame ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        color: #c4b5fd !important;
        font-weight: 600;
    }

    /* ── Divider ── */
    hr { border-color: rgba(167,139,250,0.15); }

    /* ── Selectbox / DateInput labels ── */
    label { color: #c4b5fd !important; font-weight: 500; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(167,139,250,0.1));
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #6C63FF, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* ── Metric comparison card ── */
    .metric-card {
        background: rgba(15,12,41,0.6);
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .metric-card h3 { color: #a78bfa; font-size: 1rem; margin: 0; }
    .metric-card .val { color: #f8fafc; font-size: 1.8rem; font-weight: 700; }
    .metric-card .label { color: #64748b; font-size: 0.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Defaults
# ─────────────────────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state["df"] = None
if "train_results" not in st.session_state:
    st.session_state["train_results"] = None
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1.2rem 0 0.5rem;'>
            <div style='font-size:2.5rem;'>📊</div>
            <div style='font-size:1.1rem; font-weight:700; color:#a78bfa;'>
                Sales Forecasting
            </div>
            <div style='font-size:0.75rem; color:#64748b; margin-top:2px;'>
                AI-Powered Analytics
            </div>
        </div>
        <hr style='margin:1rem 0;'>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["📈  Dashboard & EDA", "🤖  Model Training", "🔮  Sales Forecaster"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Data status indicator
    data_ok = DATA_DIR.exists() and (DATA_DIR / "train.csv").exists()
    json_ok = DATA_DIR.exists() and (DATA_DIR / "category_maps.json").exists()
    if data_ok:
        st.success("✅ Data folder detected", icon="✅")
    elif json_ok:
        st.info("ℹ️ Lite mode (no train.csv)", icon="ℹ️")
    else:
        st.error("❌ data/ folder missing", icon="❌")

    model_ok = BEST_MODEL_PATH.exists()
    if model_ok:
        st.success("✅ Trained model found", icon="✅")
    else:
        st.warning("⚠️ No trained model yet", icon="⚠️")

    st.markdown(
        """
        <div style='position:absolute; bottom:1.5rem; left:0; right:0;
                    text-align:center; color:#475569; font-size:0.72rem;'>
            Favorita Sales Dataset · 2013–2017
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Utility: data guard
# ─────────────────────────────────────────────────────────────────────────────

def _check_data_dir() -> bool:
    """Return True if data directory and train.csv exist; else show info banner."""
    if not DATA_DIR.exists():
        st.error(
            f"❌ **Data directory not found**: `{DATA_DIR}`\n\n"
            "Please place `train.csv`, `stores.csv`, `oil.csv`, "
            "`holidays_events.csv`, and `transactions.csv` "
            "inside a `data/` folder in the project root.",
            icon="❌",
        )
        return False
    if not (DATA_DIR / "train.csv").exists():
        st.info(
            "📦 **`train.csv` is not available in this deployment.**\n\n"
            "The **Dashboard & EDA** and **Model Training** pages require the full "
            "Kaggle *Store Sales* dataset (~116 MB).\n\n"
            "👉 Download it from "
            "[Kaggle — Store Sales Time Series Forecasting]("
            "https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) "
            "and place `train.csv` in the `data/` folder to enable these features.\n\n"
            "💡 The **🔮 Sales Forecaster** page works without the dataset — try it!",
            icon="ℹ️",
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Dashboard & EDA
# ─────────────────────────────────────────────────────────────────────────────

def page_dashboard():
    # Hero
    st.markdown(
        """
        <div class='hero-banner'>
            <p class='hero-title'>📊 Dashboard & Exploratory Analysis</p>
            <p class='hero-sub'>
                Understand your data at a glance — schema, statistics, and rich interactive visualisations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not _check_data_dir():
        return

    # ── Load / cache data ──────────────────────────────────────────────────
    if not st.session_state["data_loaded"] or st.session_state["df"] is None:
        with st.spinner("Loading and merging all CSV files…"):
            try:
                df = build_dataset()
                st.session_state["df"] = df
                st.session_state["data_loaded"] = True
            except Exception as exc:
                st.error(f"❌ Failed to load data: {exc}")
                return
    else:
        df = st.session_state["df"]

    # ── Top KPI cards ──────────────────────────────────────────────────────
    total_sales = df["sales"].sum()
    n_stores = df["store_nbr"].nunique()
    n_families = df["family"].nunique() if "family" in df.columns else "—"
    date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"{total_sales:,.0f}")
    c2.metric("Stores", n_stores)
    c3.metric("Product Families", n_families)
    c4.metric("Date Range", date_range)

    st.markdown("---")

    # ── Data Schema & Stats ────────────────────────────────────────────────
    with st.expander("🗂️ Data Schema & Sample Rows", expanded=False):
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Non-Null": [df[c].notna().sum() for c in df.columns],
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        st.caption(f"Total rows: {len(df):,}  |  Total columns: {len(df.columns)}")
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📐 Summary Statistics", expanded=False):
        stats = get_summary_stats(df)
        st.dataframe(stats.style.background_gradient(cmap="Purples", axis=None),
                     use_container_width=True)

    st.markdown("---")

    # ── Interactive Charts ────────────────────────────────────────────────
    st.markdown("### 📉 Historical Sales Trends")
    with st.spinner("Rendering sales trend chart…"):
        fig1 = plot_sales_trend(df)
        st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🏪 Sales by Store Type & Cluster")
        with st.spinner("Rendering store-type chart…"):
            fig2 = plot_sales_by_store_type(df)
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("### 🏷️ Promotion Impact")
        with st.spinner("Rendering promotion chart…"):
            fig3 = plot_promotion_impact(df)
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🛒 Top Product Families")
    top_n = st.slider("Number of families to show", 5, 33, 15)
    with st.spinner("Rendering top families chart…"):
        fig4 = plot_top_families(df, top_n=top_n)
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Model Training
# ─────────────────────────────────────────────────────────────────────────────

def page_training():
    st.markdown(
        """
        <div class='hero-banner'>
            <p class='hero-title'>🤖 Model Training Control Panel</p>
            <p class='hero-sub'>
                Preprocess data, train the hybrid ML pipeline, and compare model performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not _check_data_dir():
        return

    st.markdown("#### Step 1 — Load & Preprocess Data")
    st.caption(
        "Merges train.csv ← stores ← oil (ffill) ← holidays (deduplicated) ← transactions. "
        "Extracts temporal features and encodes categoricals."
    )

    if st.button("🔄 Load & Preprocess Data", key="btn_load"):
        with st.spinner("Loading and preprocessing all data tables…"):
            try:
                df = build_dataset()
                st.session_state["df"] = df
                st.session_state["data_loaded"] = True
                st.success(
                    f"✅ Data ready — {len(df):,} rows · {len(df.columns)} columns "
                    f"· date range {df['date'].min().date()} → {df['date'].max().date()}"
                )
                # Quick sanity table
                st.dataframe(df[FEATURE_COLUMNS].describe().round(3),
                             use_container_width=True)
            except Exception as exc:
                st.error(f"❌ {exc}")
                return

    st.markdown("---")
    st.markdown("#### Step 2 — Train the Hybrid ML Pipeline")

    if st.session_state["df"] is None:
        st.warning("⚠️ Load the data first (Step 1) before training.", icon="⚠️")
        return

    test_frac = st.slider(
        "Test set fraction (time-based split)",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        help="The last X% of dates will be reserved as the test set.",
    )

    st.info(
        "**Models**: LinearRegression (baseline) vs XGBRegressor (500 trees)\n\n"
        "**Split**: Chronological — no data leakage\n\n"
        "**Metrics**: RMSE & MAE on held-out test dates",
        icon="ℹ️",
    )

    if st.button("🚀 Train Models", key="btn_train"):
        progress_msgs = []
        log_container = st.empty()

        def log(msg: str):
            progress_msgs.append(msg)
            log_container.code("\n".join(progress_msgs), language="")

        with st.spinner("Training in progress — this may take 1–3 minutes…"):
            try:
                results = train_and_evaluate(
                    st.session_state["df"],
                    test_fraction=test_frac,
                    progress_callback=log,
                )
                st.session_state["train_results"] = results
            except Exception as exc:
                st.error(f"❌ Training failed: {exc}")
                return

        st.success("✅ Training complete! Best model saved.", icon="✅")
        log_container.empty()

    # ── Show results ───────────────────────────────────────────────────────
    results = st.session_state.get("train_results")
    if results:
        st.markdown("---")
        st.markdown("#### 📊 Model Evaluation Results")

        metrics = results["metrics"]
        best = results["best_model"]

        # Metric cards
        cols = st.columns(len(metrics) * 2)
        for i, (model_name, m) in enumerate(metrics.items()):
            is_best = model_name == best
            border_color = "#a78bfa" if is_best else "rgba(167,139,250,0.2)"
            badge = " 🏆 Best" if is_best else ""
            with cols[i * 2]:
                st.markdown(
                    f"""<div class='metric-card' style='border-color:{border_color};'>
                        <h3>{model_name}{badge}</h3>
                        <div class='val'>{m['rmse']:.2f}</div>
                        <div class='label'>RMSE</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with cols[i * 2 + 1]:
                st.markdown(
                    f"""<div class='metric-card' style='border-color:{border_color};'>
                        <h3>{model_name}</h3>
                        <div class='val'>{m['mae']:.2f}</div>
                        <div class='label'>MAE</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparison table
        metrics_df = pd.DataFrame(metrics).T.rename_axis("Model").reset_index()
        metrics_df = metrics_df.sort_values("rmse")
        st.dataframe(
            metrics_df.style
            .highlight_min(subset=["rmse", "mae"], color="#2d1b69")
            .format({"rmse": "{:.4f}", "mae": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        # Split info
        c1, c2, c3 = st.columns(3)
        c1.metric("Train Rows", f"{results['train_size']:,}")
        c2.metric("Test Rows", f"{results['test_size']:,}")
        c3.metric("Test Cutoff", str(results["cutoff_date"].date()))

        # Download button for model
        if BEST_MODEL_PATH.exists():
            with open(BEST_MODEL_PATH, "rb") as f:
                st.download_button(
                    label="⬇️ Download Trained Model (.joblib)",
                    data=f,
                    file_name="best_model.joblib",
                    mime="application/octet-stream",
                    key="download_model",
                )

        # Visualise metrics comparison
        st.markdown("#### 📉 RMSE Comparison")
        fig_metrics = go.Figure()
        model_names = list(metrics.keys())
        rmse_vals = [metrics[m]["rmse"] for m in model_names]
        mae_vals = [metrics[m]["mae"] for m in model_names]
        fig_metrics.add_trace(go.Bar(
            x=model_names, y=rmse_vals, name="RMSE",
            marker_color=["#a78bfa" if m == best else "#475569" for m in model_names],
        ))
        fig_metrics.add_trace(go.Bar(
            x=model_names, y=mae_vals, name="MAE",
            marker_color=["#6C63FF" if m == best else "#334155" for m in model_names],
        ))
        fig_metrics.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            title="Error Metrics Comparison",
        )
        st.plotly_chart(fig_metrics, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Sales Forecaster
# ─────────────────────────────────────────────────────────────────────────────

def page_forecaster():
    st.markdown(
        """
        <div class='hero-banner'>
            <p class='hero-title'>🔮 Sales Forecaster</p>
            <p class='hero-sub'>
                Select a future date range, store, and product family to generate AI-powered sales predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not _check_data_dir():
        return

    if not BEST_MODEL_PATH.exists():
        st.warning(
            "⚠️ No trained model found. Please go to **🤖 Model Training** and train the pipeline first.",
            icon="⚠️",
        )
        return

    # ── Load category options ──────────────────────────────────────────────
    try:
        cats = get_category_maps()
    except Exception as exc:
        st.error(f"❌ Could not load category metadata: {exc}")
        return

    # ── Input controls ─────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Forecast Parameters")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Forecast Start Date",
            value=pd.Timestamp("2017-08-16").date(),
            min_value=pd.Timestamp("2017-01-01").date(),
            max_value=pd.Timestamp("2025-12-31").date(),
            key="fc_start",
        )
        end_date = st.date_input(
            "Forecast End Date",
            value=pd.Timestamp("2017-09-15").date(),
            min_value=pd.Timestamp("2017-01-01").date(),
            max_value=pd.Timestamp("2025-12-31").date(),
            key="fc_end",
        )

    with col2:
        store_nbr = st.selectbox(
            "Store Number",
            options=cats["store_numbers"],
            index=0,
            key="fc_store",
        )
        family = st.selectbox(
            "Product Family",
            options=cats["families"],
            index=cats["families"].index("BEVERAGES") if "BEVERAGES" in cats["families"] else 0,
            key="fc_family",
        )

    col3, col4 = st.columns(2)
    with col3:
        onpromotion = st.toggle(
            "Items on Promotion",
            value=False,
            key="fc_promo",
            help="Whether the product family is running a promotion during this period.",
        )
    with col4:
        oil_price = st.number_input(
            "Assumed WTI Oil Price (USD)",
            min_value=20.0,
            max_value=200.0,
            value=50.0,
            step=0.5,
            key="fc_oil",
            help="WTI crude oil price assumption for the forecast period.",
        )

    st.markdown("---")

    # ── Validate date range ────────────────────────────────────────────────
    if end_date <= start_date:
        st.warning("⚠️ End date must be after start date.", icon="⚠️")
        return

    n_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    st.caption(f"Forecast window: **{n_days} days** ({start_date} → {end_date})")

    # ── Run forecast ───────────────────────────────────────────────────────
    if st.button("🔮 Generate Forecast", key="btn_forecast"):
        with st.spinner(f"Generating {n_days}-day sales forecast…"):
            try:
                forecast_df = predict(
                    start_date=start_date,
                    end_date=end_date,
                    store_nbr=store_nbr,
                    family=family,
                    onpromotion=int(onpromotion),
                    oil_price=oil_price,
                )
                st.session_state["forecast"] = forecast_df
            except Exception as exc:
                st.error(f"❌ Forecast failed: {exc}")
                return

    # ── Show results ───────────────────────────────────────────────────────
    forecast_df = st.session_state.get("forecast")
    if forecast_df is not None and not forecast_df.empty:
        st.markdown("---")
        st.markdown("#### 📈 Forecast Results")

        # KPIs
        total_pred = forecast_df["predicted_sales"].sum()
        avg_pred = forecast_df["predicted_sales"].mean()
        peak = forecast_df.loc[forecast_df["predicted_sales"].idxmax()]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predicted Sales", f"{total_pred:,.2f}")
        c2.metric("Daily Average", f"{avg_pred:,.2f}")
        c3.metric("Peak Day", f"{peak['date'].date()} ({peak['predicted_sales']:,.2f})")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["predicted_sales"],
            mode="lines+markers",
            name="Predicted Sales",
            line=dict(color="#a78bfa", width=2.5),
            marker=dict(size=6, color="#6C63FF",
                        line=dict(color="#a78bfa", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(108,99,255,0.12)",
        ))
        fig.update_layout(
            title=dict(
                text=f"Forecast · Store {store_nbr} · {family} · "
                     f"{'Promotion ON' if onpromotion else 'No Promotion'}",
                font=dict(color="#a78bfa", size=16),
            ),
            xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Predicted Sales", gridcolor="rgba(255,255,255,0.08)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0", family="Inter, sans-serif"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("📋 View Raw Forecast Table", expanded=False):
            display_df = forecast_df.copy()
            display_df["date"] = display_df["date"].dt.date
            display_df["onpromotion"] = display_df["onpromotion"].map(
                {0: "No", 1: "Yes"}
            )
            display_df.columns = ["Date", "Store", "Family", "On Promotion", "Predicted Sales"]
            st.dataframe(
                display_df.style.background_gradient(
                    subset=["Predicted Sales"], cmap="Purples"
                ).format({"Predicted Sales": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        # CSV download
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecast as CSV",
            data=csv,
            file_name=f"forecast_store{store_nbr}_{family}_{start_date}_{end_date}.csv",
            mime="text/csv",
            key="dl_forecast",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

if page == "📈  Dashboard & EDA":
    page_dashboard()
elif page == "🤖  Model Training":
    page_training()
elif page == "🔮  Sales Forecaster":
    page_forecaster()
