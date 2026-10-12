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
from src.predictor import TRAINING_END_DATE, predict

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

    /* ── Info box ── */
    .info-box {
        background: rgba(56,189,248,0.08);
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #94a3b8;
    }
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
if "forecast" not in st.session_state:
    st.session_state["forecast"] = None

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
        ["🔮  Sales Forecaster", "📈  Dashboard & EDA", "🤖  Model Training"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Data status indicator
    data_ok = DATA_DIR.exists() and (DATA_DIR / "train.csv").exists()
    json_ok = DATA_DIR.exists() and (DATA_DIR / "category_maps.json").exists()
    if data_ok:
        st.success("✅ Full dataset available", icon="✅")
    elif json_ok:
        st.info("ℹ️ Lite mode — Forecaster active", icon="ℹ️")
    else:
        st.error("❌ data/ folder missing", icon="❌")

    model_ok = BEST_MODEL_PATH.exists()
    if model_ok:
        st.success("✅ Trained model loaded", icon="✅")
    else:
        st.warning("⚠️ No trained model yet", icon="⚠️")

    st.markdown(
        """
        <div style='margin-top:1rem; padding: 0.8rem; background: rgba(167,139,250,0.06);
                    border: 1px solid rgba(167,139,250,0.15); border-radius: 8px;
                    font-size:0.72rem; color:#64748b;'>
            <b style='color:#a78bfa;'>Dataset</b><br>
            Favorita Store Sales<br>
            Ecuador · 2013–2017<br>
            54 stores · 33 product families
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Utility: data guard (only for pages that need train.csv)
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
# PAGE 1 — Sales Forecaster  (works WITHOUT train.csv)
# ─────────────────────────────────────────────────────────────────────────────

def page_forecaster():
    st.markdown(
        """
        <div class='hero-banner'>
            <p class='hero-title'>🔮 Sales Forecaster</p>
            <p class='hero-sub'>
                Select a store, product family, and date range to generate
                AI-powered sales predictions using a trained XGBoost model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model guard ────────────────────────────────────────────────────────
    if not BEST_MODEL_PATH.exists():
        st.warning(
            "⚠️ **No trained model found.**\n\n"
            "Please go to **🤖 Model Training** and train the pipeline first. "
            "You will need the full dataset for that step.",
            icon="⚠️",
        )
        return

    # ── Load category options (works with JSON fallback) ───────────────────
    try:
        cats = get_category_maps()
    except Exception as exc:
        st.error(f"❌ Could not load category metadata: {exc}")
        return

    # ── Model info banner ──────────────────────────────────────────────────
    with st.expander("ℹ️ About this forecast — important notes", expanded=False):
        st.markdown(
            f"""
            **Model**: XGBoost Regressor (500 trees, chronological train/test split)

            **Training data**: Favorita Store Sales — Ecuador, 2013-01-01 → **{TRAINING_END_DATE.date()}**

            **Features used** ({len(FEATURE_COLUMNS)}):
            `store_nbr`, `family_enc`, `onpromotion`, `type_enc` (store type),
            `cluster`, `day_of_week`, `month`, `is_weekend`, `quarter`, `year`,
            `day_of_year`, `dcoilwtico` (oil price), `transactions` (store median),
            `is_holiday`

            **Limitations**:
            - Forecasts beyond {TRAINING_END_DATE.date()} are **extrapolations** —
              the model has never seen those dates. Uncertainty increases with
              forecast horizon.
            - Future transactions and holidays are unknown; the model uses
              historical store medians and assumes no holidays.
            - This is a point forecast — no confidence intervals are provided.
            - Predictions reflect patterns learned from Ecuador 2013–2017 and
              should not be used for actual business decisions without validation.
            """
        )

    # ── Input controls ─────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Forecast Parameters")

    col1, col2 = st.columns(2)

    with col1:
        default_start = pd.Timestamp("2017-08-16").date()
        default_end = pd.Timestamp("2017-09-15").date()

        start_date = st.date_input(
            "Forecast Start Date",
            value=default_start,
            min_value=pd.Timestamp("2013-01-01").date(),
            max_value=pd.Timestamp("2025-12-31").date(),
            key="fc_start",
        )
        end_date = st.date_input(
            "Forecast End Date",
            value=default_end,
            min_value=pd.Timestamp("2013-01-01").date(),
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
        families = cats["families"]
        default_family_idx = families.index("BEVERAGES") if "BEVERAGES" in families else 0
        family = st.selectbox(
            "Product Family",
            options=families,
            index=default_family_idx,
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
            value=47.26,
            step=0.5,
            key="fc_oil",
            help="WTI crude oil price assumption. Training-period median ≈ $47/bbl.",
        )

    # ── Date validation ────────────────────────────────────────────────────
    if end_date <= start_date:
        st.warning("⚠️ End date must be after start date.", icon="⚠️")
        return

    n_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1

    # Warn when forecasting far beyond training window
    forecast_start_ts = pd.Timestamp(start_date)
    if forecast_start_ts > TRAINING_END_DATE:
        days_beyond = (forecast_start_ts - TRAINING_END_DATE).days
        st.warning(
            f"⚠️ **Extrapolation warning**: Your forecast starts "
            f"**{days_beyond} days after** the training data ends "
            f"({TRAINING_END_DATE.date()}). Accuracy decreases for dates "
            "further from the training window.",
            icon="⚠️",
        )
    elif forecast_start_ts < TRAINING_END_DATE:
        st.info(
            "ℹ️ You are forecasting within the **training period** "
            f"(data ends {TRAINING_END_DATE.date()}). "
            "These predictions can be compared against known actuals if you "
            "have the train.csv file.",
            icon="ℹ️",
        )

    st.caption(f"Forecast window: **{n_days} days** ({start_date} → {end_date})")

    st.markdown("---")

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
                st.session_state["forecast_meta"] = {
                    "store_nbr": store_nbr,
                    "family": family,
                    "onpromotion": onpromotion,
                }
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
        peak_idx = forecast_df["predicted_sales"].idxmax()
        peak = forecast_df.loc[peak_idx]
        trough_idx = forecast_df["predicted_sales"].idxmin()
        trough = forecast_df.loc[trough_idx]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predicted Sales", f"{total_pred:,.1f}")
        c2.metric("Daily Average", f"{avg_pred:,.2f}")
        peak_date = peak["date"]
        if hasattr(peak_date, "date"):
            peak_date = peak_date.date()
        c3.metric("Peak Day", f"{peak_date}", f"{peak['predicted_sales']:,.2f} units")
        trough_date = trough["date"]
        if hasattr(trough_date, "date"):
            trough_date = trough_date.date()
        c4.metric("Lowest Day", f"{trough_date}", f"{trough['predicted_sales']:,.2f} units")

        # Chart
        meta = st.session_state.get("forecast_meta", {})
        _store = meta.get("store_nbr", store_nbr)
        _family = meta.get("family", family)
        _promo = meta.get("onpromotion", onpromotion)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["predicted_sales"],
            mode="lines+markers",
            name="Predicted Sales",
            line=dict(color="#a78bfa", width=2.5),
            marker=dict(size=5, color="#6C63FF",
                        line=dict(color="#a78bfa", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(108,99,255,0.12)",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Sales: %{y:,.2f}<extra></extra>",
        ))
        # Add a dashed line at training end if it falls in range
        if pd.Timestamp(start_date) <= TRAINING_END_DATE <= pd.Timestamp(end_date):
            fig.add_vline(
                x=TRAINING_END_DATE,
                line_dash="dash",
                line_color="rgba(251,191,36,0.6)",
                annotation_text="Training data ends",
                annotation_font_color="#fbbf24",
            )
        fig.update_layout(
            title=dict(
                text=f"Forecast · Store {_store} · {_family} · "
                     f"{'Promotion ON' if _promo else 'No Promotion'}",
                font=dict(color="#a78bfa", size=16),
            ),
            xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Predicted Sales (units)", gridcolor="rgba(255,255,255,0.08)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0", family="Inter, sans-serif"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Weekday pattern insight
        forecast_df["day_name"] = pd.to_datetime(forecast_df["date"]).dt.day_name()
        by_day = forecast_df.groupby("day_name")["predicted_sales"].mean().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        if not by_day.isna().all():
            fig_day = px.bar(
                x=by_day.index,
                y=by_day.values,
                labels={"x": "Day of Week", "y": "Avg Predicted Sales"},
                color=by_day.values,
                color_continuous_scale="Purples",
                title="Average Predicted Sales by Day of Week",
            )
            fig_day.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0", family="Inter, sans-serif"),
                showlegend=False,
                coloraxis_showscale=False,
                title_font_color="#a78bfa",
            )
            fig_day.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
            fig_day.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig_day, use_container_width=True)

        # Data table
        with st.expander("📋 View Raw Forecast Table", expanded=False):
            display_df = forecast_df[["date", "store_nbr", "family", "onpromotion", "predicted_sales"]].copy()
            display_df["date"] = pd.to_datetime(display_df["date"]).dt.date
            display_df["onpromotion"] = display_df["onpromotion"].map({0: "No", 1: "Yes"})
            display_df.columns = ["Date", "Store", "Family", "On Promotion", "Predicted Sales"]
            st.dataframe(
                display_df.style.background_gradient(
                    subset=["Predicted Sales"], cmap="Purples"
                ).format({"Predicted Sales": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        # CSV download
        csv_data = forecast_df[["date", "store_nbr", "family", "onpromotion", "predicted_sales"]].copy()
        csv_data["date"] = pd.to_datetime(csv_data["date"]).dt.date
        csv = csv_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecast as CSV",
            data=csv,
            file_name=f"forecast_store{_store}_{_family}_{start_date}_{end_date}.csv",
            mime="text/csv",
            key="dl_forecast",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Dashboard & EDA
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
    date_min = df["date"].min()
    date_max = df["date"].max()
    date_range = f"{date_min.date()} → {date_max.date()}"
    avg_daily = df.groupby("date")["sales"].sum().mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", f"{total_sales:,.0f}")
    c2.metric("Stores", n_stores)
    c3.metric("Product Families", n_families)
    c4.metric("Avg Daily Sales", f"{avg_daily:,.0f}")
    c5.metric("Date Range", date_range)

    st.markdown("---")

    # ── Data quality summary ───────────────────────────────────────────────
    with st.expander("🔍 Data Quality Report", expanded=False):
        total_rows = len(df)
        missing_counts = df[FEATURE_COLUMNS + ["sales"]].isna().sum()
        missing_df = missing_counts[missing_counts > 0].reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        missing_df["Missing %"] = (missing_df["Missing Count"] / total_rows * 100).round(2)

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.metric("Total Rows", f"{total_rows:,}")
            st.metric("Total Columns", len(df.columns))
            st.metric("Zero-sales rows", f"{(df['sales'] == 0).sum():,}")
        with col_q2:
            st.metric("Negative sales rows", f"{(df['sales'] < 0).sum():,}")
            st.metric("Duplicate rows", f"{df.duplicated().sum():,}")
            if len(missing_df) > 0:
                st.metric("Columns with missing values", len(missing_df))
            else:
                st.metric("Missing values", "None ✅")

        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No missing values in feature columns after preprocessing.")

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
        try:
            fig1 = plot_sales_trend(df)
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as exc:
            st.error(f"❌ Could not render sales trend: {exc}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🏪 Sales by Store Type & Cluster")
        with st.spinner("Rendering store-type chart…"):
            try:
                fig2 = plot_sales_by_store_type(df)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as exc:
                st.error(f"❌ Could not render store-type chart: {exc}")

    with col_b:
        st.markdown("### 🏷️ Promotion Impact")
        with st.spinner("Rendering promotion chart…"):
            try:
                fig3 = plot_promotion_impact(df)
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as exc:
                st.error(f"❌ Could not render promotion chart: {exc}")

    st.markdown("### 🛒 Top Product Families")
    top_n = st.slider("Number of families to show", 5, 33, 15)
    with st.spinner("Rendering top families chart…"):
        try:
            fig4 = plot_top_families(df, top_n=top_n)
            st.plotly_chart(fig4, use_container_width=True)
        except Exception as exc:
            st.error(f"❌ Could not render families chart: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Model Training
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

    # ── Model info ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ About the ML pipeline", expanded=False):
        st.markdown(
            f"""
            **Baseline**: `LinearRegression` (StandardScaler → LinearRegression)

            **Primary**: `XGBRegressor` (500 trees, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3)

            **Split strategy**: Chronological time-based split — the last `test_fraction`
            of unique dates forms the test set. This prevents data leakage that random
            splitting would cause in time-series problems.

            **Features** ({len(FEATURE_COLUMNS)}): {', '.join(f'`{f}`' for f in FEATURE_COLUMNS)}

            **Target**: `sales` (unit sales per day, per store, per product family)

            **Evaluation**: RMSE & MAE on the held-out chronological test set.
            """
        )

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

    # Compute cutoff preview
    df_preview = st.session_state["df"]
    sorted_dates = sorted(df_preview["date"].unique())
    n_test_preview = max(1, int(len(sorted_dates) * test_frac))
    cutoff_preview = sorted_dates[-n_test_preview]
    st.info(
        f"**Models**: LinearRegression (baseline) vs XGBRegressor (500 trees)\n\n"
        f"**Split**: Chronological — train on dates before **{pd.Timestamp(cutoff_preview).date()}**, "
        f"test on {n_test_preview} dates after\n\n"
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

        # Interpretation note
        st.markdown(
            f"""
            <div class='info-box'>
            <b>How to interpret these metrics:</b><br>
            • <b>RMSE</b> (Root Mean Squared Error): Average prediction error in <em>sales units</em>.
              Lower is better. Penalises large errors more than MAE.<br>
            • <b>MAE</b> (Mean Absolute Error): Average absolute error in <em>sales units</em>.
              Lower is better. More robust to outliers than RMSE.<br>
            • Evaluated on the chronological holdout — dates after {pd.Timestamp(results['cutoff_date']).date()}.<br>
            • No future information was used during training.
            </div>
            """,
            unsafe_allow_html=True,
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
        st.markdown("#### 📉 RMSE vs MAE Comparison")
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
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Error (sales units)"),
            xaxis=dict(title="Model"),
            title=dict(text="Error Metrics Comparison (lower = better)",
                       font=dict(color="#a78bfa")),
        )
        st.plotly_chart(fig_metrics, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

if page == "🔮  Sales Forecaster":
    page_forecaster()
elif page == "📈  Dashboard & EDA":
    page_dashboard()
elif page == "🤖  Model Training":
    page_training()
