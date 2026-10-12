"""
data_loader.py
==============
Robust data ingestion and preprocessing pipeline for the Sales Forecasting System.

Loads train.csv, stores.csv, oil.csv, holidays_events.csv, and transactions.csv
from the data/ directory, merges them safely, handles missing values, and
engineers temporal features ready for machine learning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Canonical feature columns produced by preprocess()
FEATURE_COLUMNS = [
    "store_nbr",
    "family_enc",
    "onpromotion",
    "type_enc",
    "cluster",
    "day_of_week",
    "month",
    "is_weekend",
    "quarter",
    "year",
    "day_of_year",
    "dcoilwtico",
    "transactions",
    "is_holiday",
]

TARGET_COLUMN = "sales"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature_columns() -> list[str]:
    """Return the canonical list of feature columns used by the ML models."""
    return list(FEATURE_COLUMNS)


def load_raw_data(data_dir: str | Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    Load all five source CSV files from *data_dir*.

    Returns
    -------
    dict with keys: 'train', 'stores', 'oil', 'holidays', 'transactions'

    Raises
    ------
    FileNotFoundError  – if data_dir or any required CSV is missing.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Please place train.csv, stores.csv, oil.csv, "
            "holidays_events.csv, and transactions.csv inside a data/ folder."
        )

    required = {
        "train": "train.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
    }

    missing = [name for name, fname in required.items() if not (data_dir / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required CSV files: {[required[k] for k in missing]}"
        )

    dfs: dict[str, pd.DataFrame] = {}
    for key, fname in required.items():
        dfs[key] = pd.read_csv(data_dir / fname, low_memory=False)

    return dfs


def merge_tables(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Safely merge the five source DataFrames into a single analytical table.

    Merge order (all left-joins on `train`):
        train ← stores  (on store_nbr)
             ← oil      (on date)
             ← holidays (on date)  — duplicate dates dropped first!
             ← transactions (on date + store_nbr)

    Parameters
    ----------
    dfs : dict returned by load_raw_data()

    Returns
    -------
    Merged DataFrame (same row count as train).
    """
    train = dfs["train"].copy()
    stores = dfs["stores"].copy()
    oil = dfs["oil"].copy()
    holidays = dfs["holidays"].copy()
    transactions = dfs["transactions"].copy()

    # --- Date normalisation ---
    for df in [train, oil, holidays, transactions]:
        df["date"] = pd.to_datetime(df["date"])

    # --- Stores ---
    merged = train.merge(stores, on="store_nbr", how="left")

    # --- Oil ---
    merged = merged.merge(oil.rename(columns={"dcoilwtico": "dcoilwtico"}),
                          on="date", how="left")

    # --- Holidays: keep only National-scope events; drop duplicate dates ---
    # We flag any date that appears in holidays as is_holiday = 1.
    # Deduplication prevents row explosion from multiple locale entries per day.
    holidays_dedup = (
        holidays
        .sort_values("type")              # prefer 'Holiday' over 'Event'
        .drop_duplicates(subset=["date"], keep="first")
        [["date", "type", "transferred"]]
        .rename(columns={"type": "holiday_type"})
    )
    merged = merged.merge(holidays_dedup, on="date", how="left")
    merged["is_holiday"] = merged["holiday_type"].notna().astype(int)
    merged.drop(columns=["holiday_type", "transferred"], inplace=True, errors="ignore")

    # --- Transactions ---
    transactions_agg = (
        transactions
        .groupby(["date", "store_nbr"], as_index=False)["transactions"]
        .sum()
    )
    merged = merged.merge(transactions_agg, on=["date", "store_nbr"], how="left")

    return merged


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and feature-engineer the merged DataFrame.

    Steps
    -----
    1. Forward-fill missing oil prices (then back-fill any leading NaNs).
    2. Fill missing transaction counts with the store-level median.
    3. Extract temporal features.
    4. Label-encode categorical columns (family, store type).

    Returns
    -------
    Cleaned DataFrame with all FEATURE_COLUMNS present.
    """
    df = df.copy()

    # 1. Oil prices — forward fill then back fill
    df["dcoilwtico"] = df.sort_values("date")["dcoilwtico"].ffill().bfill()

    # 2. Transactions — fill with per-store median
    store_med = df.groupby("store_nbr")["transactions"].transform("median")
    df["transactions"] = df["transactions"].fillna(store_med).fillna(0)

    # 3. Temporal features
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear

    # 4. Label-encode categoricals  ─ store them as simple integer codes
    df["family_enc"] = df["family"].astype("category").cat.codes
    df["type_enc"] = df["type"].astype("category").cat.codes

    # 5. Ensure is_holiday exists (in case merge was skipped)
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    # 6. Fill any remaining NaNs in feature columns
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    return df


def build_dataset(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """
    Convenience function: load → merge → preprocess in one call.

    Returns the fully prepared analytical DataFrame.
    """
    dfs = load_raw_data(data_dir)
    merged = merge_tables(dfs)
    processed = preprocess(merged)
    return processed


def get_category_maps(data_dir: str | Path = DATA_DIR) -> dict[str, list]:
    """
    Return mappings needed by the prediction UI.
    Returns dict with 'families', 'store_numbers', 'store_types', 'clusters'.

    Falls back to data/category_maps.json when train.csv is absent
    (e.g. on Streamlit Community Cloud without the full Kaggle dataset).
    """
    data_dir = Path(data_dir)
    train_csv = data_dir / "train.csv"
    json_path = data_dir / "category_maps.json"

    if train_csv.exists():
        dfs = load_raw_data(data_dir)
        train = dfs["train"]
        stores = dfs["stores"]
        return {
            "families": sorted(train["family"].unique().tolist()),
            "store_numbers": sorted(train["store_nbr"].unique().tolist()),
            "store_types": sorted(stores["type"].unique().tolist()),
            "clusters": sorted(stores["cluster"].unique().tolist()),
        }

    # Fallback: use pre-baked JSON (works on Streamlit Cloud)
    if json_path.exists():
        with open(json_path, "r") as fh:
            maps = json.load(fh)
        return {
            "families": maps["families"],
            "store_numbers": maps["store_numbers"],
            "store_types": maps["store_types"],
            "clusters": maps["clusters"],
        }

    raise FileNotFoundError(
        f"Neither train.csv nor category_maps.json found in {data_dir}. "
        "Cannot load category maps."
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading and merging data...")
    df = build_dataset()
    print(f"[OK] Merged shape       : {df.shape}")
    print(f"[OK] Date range         : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"[OK] Missing in features: {df[FEATURE_COLUMNS].isna().sum().sum()}")
    print(f"[OK] Columns            : {list(df.columns)}")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe())
