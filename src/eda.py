"""
eda.py
======
Exploratory Data Analysis chart factories for the Sales Forecasting System.
All functions return Plotly Figure objects ready to be rendered in Streamlit.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette — rich dark theme compatible
# ---------------------------------------------------------------------------
PALETTE = px.colors.qualitative.Vivid
ACCENT = "#6C63FF"
BG_TRANSPARENT = "rgba(0,0,0,0)"

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BG_TRANSPARENT,
    plot_bgcolor=BG_TRANSPARENT,
    font=dict(family="Inter, sans-serif", color="#E2E8F0"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _apply_defaults(fig: go.Figure, title: str) -> go.Figure:
    """Apply shared dark-theme layout defaults to a figure."""
    fig.update_layout(title=dict(text=title, font=dict(size=18, color="#A78BFA")),
                      **_LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


# ---------------------------------------------------------------------------
# Chart 1 – Overall Sales Trend
# ---------------------------------------------------------------------------

def plot_sales_trend(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of total daily sales across all stores and families.

    Parameters
    ----------
    df : preprocessed DataFrame from data_loader.build_dataset()
    """
    daily = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )
    # Rolling 7-day average for a smoother trend line
    daily["sales_7d_avg"] = daily["sales"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["sales"],
        mode="lines",
        name="Daily Sales",
        line=dict(color="rgba(108,99,255,0.35)", width=1),
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["sales_7d_avg"],
        mode="lines",
        name="7-Day Average",
        line=dict(color=ACCENT, width=2.5),
    ))

    fig = _apply_defaults(fig, "📈 Overall Historical Sales Trend")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Total Sales")
    return fig


# ---------------------------------------------------------------------------
# Chart 2 – Sales Distribution by Store Type & Cluster
# ---------------------------------------------------------------------------

def plot_sales_by_store_type(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart showing total sales broken down by store type and cluster.
    """
    grouped = (
        df.groupby(["type", "cluster"], as_index=False)["sales"]
        .sum()
        .sort_values(["type", "cluster"])
    )
    grouped["cluster"] = grouped["cluster"].astype(str)

    fig = px.bar(
        grouped,
        x="type",
        y="sales",
        color="cluster",
        barmode="group",
        color_discrete_sequence=PALETTE,
        labels={"type": "Store Type", "sales": "Total Sales", "cluster": "Cluster"},
    )
    fig = _apply_defaults(fig, "🏪 Sales Distribution by Store Type & Cluster")
    fig.update_traces(marker_line_width=0)
    return fig


# ---------------------------------------------------------------------------
# Chart 3 – Promotion Impact on Sales
# ---------------------------------------------------------------------------

def plot_promotion_impact(df: pd.DataFrame) -> go.Figure:
    """
    Side-by-side box plot of sales for items with vs without active promotions.
    Only uses a random sample of rows to keep rendering fast.
    """
    sample = df.sample(min(150_000, len(df)), random_state=42) if len(df) > 150_000 else df.copy()
    sample = sample[sample["sales"] > 0]   # exclude zero-sales rows for clarity
    sample["Promotion"] = sample["onpromotion"].map({0: "No Promotion", 1: "On Promotion"})

    fig = px.box(
        sample,
        x="Promotion",
        y="sales",
        color="Promotion",
        color_discrete_map={"No Promotion": "#64748B", "On Promotion": "#A78BFA"},
        log_y=True,
        labels={"sales": "Sales (log scale)"},
        points=False,
    )
    fig = _apply_defaults(fig, "🏷️ Impact of Promotions on Sales")
    fig.update_traces(line_width=1.5)
    return fig


# ---------------------------------------------------------------------------
# Chart 4 – Top Families by Sales
# ---------------------------------------------------------------------------

def plot_top_families(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar of the top N product families by total sales."""
    family_sales = (
        df.groupby("family", as_index=False)["sales"]
        .sum()
        .nlargest(top_n, "sales")
        .sort_values("sales")
    )
    fig = px.bar(
        family_sales,
        x="sales",
        y="family",
        orientation="h",
        color="sales",
        color_continuous_scale="Purples",
        labels={"sales": "Total Sales", "family": "Product Family"},
    )
    fig = _apply_defaults(fig, f"🛒 Top {top_n} Product Families by Sales")
    fig.update_coloraxes(showscale=False)
    return fig


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a styled summary statistics table for the key numerical columns.
    """
    cols = ["sales", "onpromotion", "dcoilwtico", "transactions"]
    available = [c for c in cols if c in df.columns]
    stats = df[available].describe().T.round(2)
    return stats
