"""
eda.py
======
Exploratory Data Analysis chart factories — upgraded for the Sales Forecast AI redesign.
All functions return Plotly Figure objects ready for Streamlit.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Design system tokens
# ---------------------------------------------------------------------------

# Primary palette
C_INDIGO    = "#6366f1"   # electric indigo  — AI / predictions
C_VIOLET    = "#8b5cf6"   # violet           — accent
C_TEAL      = "#14b8a6"   # teal             — positive / historical
C_EMERALD   = "#10b981"   # emerald          — growth indicators
C_AMBER     = "#f59e0b"   # amber            — warnings / highlights
C_CORAL     = "#f43f5e"   # coral            — alerts
C_LAVENDER  = "#a5b4fc"   # lavender         — secondary elements
C_SKY       = "#38bdf8"   # sky blue         — supporting data
C_SLATE     = "#94a3b8"   # slate            — muted text / grid

# Sequential scales
SEQ_INDIGO  = [[0, "#1e1b4b"], [0.5, "#4338ca"], [1, "#818cf8"]]
SEQ_TEAL    = [[0, "#042f2e"], [0.5, "#0d9488"], [1, "#5eead4"]]

BG_TRANSPARENT = "rgba(0,0,0,0)"
GRID_COLOR     = "rgba(255,255,255,0.06)"

# Qualitative palette for multi-series charts
QUAL_PALETTE = [
    C_INDIGO, C_TEAL, C_AMBER, C_CORAL, C_LAVENDER,
    "#06b6d4", "#a3e635", "#fb923c", "#c084fc", "#34d399",
    "#f472b6", "#60a5fa",
]

_LAYOUT_BASE = dict(
    paper_bgcolor=BG_TRANSPARENT,
    plot_bgcolor=BG_TRANSPARENT,
    font=dict(family="Inter, system-ui, sans-serif", color="#cbd5e1", size=13),
    legend=dict(
        bgcolor="rgba(15,23,42,0.7)",
        bordercolor="rgba(99,102,241,0.2)",
        borderwidth=1,
        font=dict(size=12),
    ),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(
        bgcolor="rgba(15,23,42,0.95)",
        bordercolor=C_INDIGO,
        font=dict(family="Inter, sans-serif", size=13, color="#f1f5f9"),
    ),
)


def _base_axes() -> dict:
    return dict(
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=11, color=C_SLATE),
        title_font=dict(size=12, color=C_SLATE),
    )


def _apply_defaults(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    """Apply shared premium layout defaults."""
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:12px;color:{C_SLATE};font-weight:normal'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=17, color="#e2e8f0"), x=0, xanchor="left"),
        **_LAYOUT_BASE,
    )
    fig.update_xaxes(**_base_axes(), showgrid=True)
    fig.update_yaxes(**_base_axes(), showgrid=True)
    return fig


# ---------------------------------------------------------------------------
# Chart 1 – Overall Sales Trend (area + 7-day MA)
# ---------------------------------------------------------------------------

def plot_sales_trend(df: pd.DataFrame) -> go.Figure:
    """
    Area chart of total daily sales with a 7-day rolling average overlay.
    Uses teal for historical area and indigo for the moving average line.
    """
    daily = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )
    daily["ma7"]  = daily["sales"].rolling(7, min_periods=1).mean()
    daily["ma30"] = daily["sales"].rolling(30, min_periods=1).mean()

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["sales"],
        mode="lines",
        name="Daily Sales",
        line=dict(color=f"rgba(20,184,166,0.4)", width=1),
        fill="tozeroy",
        fillcolor="rgba(20,184,166,0.07)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sales: <b>%{y:,.0f}</b><extra></extra>",
    ))
    # 7-day MA
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ma7"],
        mode="lines",
        name="7-Day Average",
        line=dict(color=C_TEAL, width=2.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>7D Avg: <b>%{y:,.0f}</b><extra></extra>",
    ))
    # 30-day MA
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ma30"],
        mode="lines",
        name="30-Day Average",
        line=dict(color=C_INDIGO, width=2, dash="dot"),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>30D Avg: <b>%{y:,.0f}</b><extra></extra>",
    ))

    fig = _apply_defaults(
        fig,
        "Historical Sales Trend",
        "Daily aggregated sales across all stores and product families",
    )
    fig.update_xaxes(title_text="Date", rangeslider=dict(visible=False))
    fig.update_yaxes(title_text="Total Sales (units)")
    fig.update_layout(hovermode="x unified", height=380)
    return fig


# ---------------------------------------------------------------------------
# Chart 2 – Sales by Store Type & Cluster
# ---------------------------------------------------------------------------

def plot_sales_by_store_type(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: total sales broken down by store type and cluster.
    """
    if "type" not in df.columns or "cluster" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Store type / cluster data unavailable",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=C_SLATE, size=14))
        return _apply_defaults(fig, "Sales by Store Type")

    grouped = (
        df.groupby(["type", "cluster"], as_index=False)["sales"]
        .sum()
        .sort_values(["type", "cluster"])
    )
    grouped["cluster"] = "Cluster " + grouped["cluster"].astype(str)

    fig = px.bar(
        grouped,
        x="type",
        y="sales",
        color="cluster",
        barmode="group",
        color_discrete_sequence=QUAL_PALETTE,
        labels={"type": "Store Type", "sales": "Total Sales", "cluster": ""},
        custom_data=["cluster"],
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{x} — %{customdata[0]}</b><br>Sales: %{y:,.0f}<extra></extra>",
    )
    fig = _apply_defaults(fig, "Sales by Store Type & Cluster")
    fig.update_layout(height=340, bargap=0.2, bargroupgap=0.05)
    return fig


# ---------------------------------------------------------------------------
# Chart 3 – Promotion Impact
# ---------------------------------------------------------------------------

def plot_promotion_impact(df: pd.DataFrame) -> go.Figure:
    """
    Violin + box hybrid showing sales distribution for promoted vs non-promoted items.
    """
    sample = df.sample(min(100_000, len(df)), random_state=42) if len(df) > 100_000 else df.copy()
    sample = sample[sample["sales"] > 0]
    sample["Promotion"] = sample["onpromotion"].map({0: "No Promotion", 1: "On Promotion"})
    # Handle non-binary onpromotion (it's sometimes a count)
    if sample["Promotion"].isna().any():
        sample["Promotion"] = sample["onpromotion"].apply(
            lambda x: "On Promotion" if x > 0 else "No Promotion"
        )

    fig = go.Figure()
    for label, color in [("No Promotion", C_SLATE), ("On Promotion", C_INDIGO)]:
        sub = sample[sample["Promotion"] == label]["sales"]
        fig.add_trace(go.Violin(
            y=sub,
            name=label,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in color else f"{color}26",
            line_color=color,
            points=False,
            hovertemplate=f"<b>{label}</b><br>Sales: %{{y:,.1f}}<extra></extra>",
        ))

    fig = _apply_defaults(fig, "Promotion Impact on Sales", "Distribution of unit sales: promoted vs. non-promoted")
    fig.update_layout(violingap=0.3, height=340, yaxis_type="log",
                      yaxis_title="Sales — log scale")
    return fig


# ---------------------------------------------------------------------------
# Chart 4 – Top Families
# ---------------------------------------------------------------------------

def plot_top_families(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar ranking of top N product families by total sales."""
    family_sales = (
        df.groupby("family", as_index=False)["sales"]
        .sum()
        .nlargest(top_n, "sales")
        .sort_values("sales")
    )
    pct = family_sales["sales"] / family_sales["sales"].sum() * 100

    fig = go.Figure(go.Bar(
        x=family_sales["sales"],
        y=family_sales["family"],
        orientation="h",
        marker=dict(
            color=family_sales["sales"],
            colorscale=[[0, "#1e1b4b"], [0.5, C_INDIGO], [1, C_LAVENDER]],
            showscale=False,
            line=dict(width=0),
        ),
        customdata=pct.round(1),
        hovertemplate="<b>%{y}</b><br>Sales: %{x:,.0f}<br>Share: %{customdata}%<extra></extra>",
    ))
    fig = _apply_defaults(fig, f"Top {top_n} Product Families", "Ranked by total units sold")
    fig.update_layout(height=max(300, top_n * 28), yaxis_tickfont=dict(size=12))
    fig.update_xaxes(title_text="Total Sales (units)")
    return fig


# ---------------------------------------------------------------------------
# Chart 5 – Monthly seasonality heatmap
# ---------------------------------------------------------------------------

def plot_monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of average daily sales by year × month.
    """
    df = df.copy()
    df["year"]  = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

    pivot = (
        df.groupby(["year", "month"])["sales"]
        .mean()
        .unstack(fill_value=0)
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        colorscale=[[0, "#0f172a"], [0.5, C_INDIGO], [1, C_LAVENDER]],
        hovertemplate="<b>%{y} %{x}</b><br>Avg Daily Sales: %{z:,.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=C_SLATE, size=11),
            outlinecolor="transparent",
            len=0.7,
        ),
    ))
    fig = _apply_defaults(fig, "Seasonal Sales Heatmap", "Average daily sales by year and month")
    fig.update_layout(height=280)
    fig.update_xaxes(side="bottom", title_text="")
    fig.update_yaxes(title_text="Year")
    return fig


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary statistics table for key numerical columns."""
    cols = ["sales", "onpromotion", "dcoilwtico", "transactions"]
    available = [c for c in cols if c in df.columns]
    stats = df[available].describe().T.round(2)
    return stats
