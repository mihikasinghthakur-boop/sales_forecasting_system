"""
predictor.py
============
Prediction module for the Sales Forecasting System.

Provides utilities to:
  - Load the serialised model artifact
  - Build a feature-complete future DataFrame from user inputs
  - Run inference and return a structured forecast table

Fallback mode: when train.csv is absent (e.g. on Streamlit Cloud), the
module reads pre-baked category mappings from data/category_maps.json so
the Forecaster page still works without the full Kaggle dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when running this module directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader import DATA_DIR, FEATURE_COLUMNS, load_raw_data
from src.model_trainer import BEST_MODEL_PATH, load_model

# ---------------------------------------------------------------------------
# JSON fallback helpers
# ---------------------------------------------------------------------------

_CATEGORY_MAPS_PATH = DATA_DIR / "category_maps.json"


def _load_category_maps_json() -> dict:
    """Load the pre-baked category_maps.json (used when train.csv is absent)."""
    if not _CATEGORY_MAPS_PATH.exists():
        raise FileNotFoundError(
            f"Neither train.csv nor category_maps.json found in {DATA_DIR}. "
            "Cannot run predictions."
        )
    with open(_CATEGORY_MAPS_PATH, "r") as fh:
        return json.load(fh)


def _train_csv_available(data_dir: Path = DATA_DIR) -> bool:
    """Return True if train.csv is present in data_dir."""
    return (data_dir / "train.csv").exists()

# ---------------------------------------------------------------------------
# Category encoding helpers (must mirror data_loader.preprocess)
# ---------------------------------------------------------------------------


def _get_family_codes(data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Return {family_name: category_code} consistent with training encoding."""
    if _train_csv_available(data_dir):
        dfs = load_raw_data(data_dir)
        families = sorted(dfs["train"]["family"].unique())
        return {f: i for i, f in enumerate(families)}
    return _load_category_maps_json()["family_codes"]


def _get_type_codes(data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Return {store_type: category_code} consistent with training encoding."""
    if _train_csv_available(data_dir):
        dfs = load_raw_data(data_dir)
        types = sorted(dfs["stores"]["type"].unique())
        return {t: i for i, t in enumerate(types)}
    return _load_category_maps_json()["type_codes"]


def _get_store_meta(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return stores DataFrame (store_nbr, type, cluster)."""
    if _train_csv_available(data_dir):
        dfs = load_raw_data(data_dir)
        return dfs["stores"][["store_nbr", "type", "cluster"]].copy()
    maps = _load_category_maps_json()["store_meta"]
    rows = [
        {"store_nbr": int(k), "type": v["type"], "cluster": v["cluster"]}
        for k, v in maps.items()
    ]
    return pd.DataFrame(rows)


def _get_recent_oil_price(data_dir: Path = DATA_DIR) -> float:
    """Return the most recent non-null oil price from the dataset."""
    if _train_csv_available(data_dir):
        dfs = load_raw_data(data_dir)
        oil = dfs["oil"].copy()
        oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
        return float(oil["dcoilwtico"].iloc[-1])
    return _load_category_maps_json()["recent_oil_price"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_future_frame(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    store_nbr: int,
    family: str,
    onpromotion: int = 0,
    oil_price: float | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Assemble a feature-complete DataFrame for future inference.

    Parameters
    ----------
    start_date   : first forecast date (inclusive)
    end_date     : last forecast date (inclusive)
    store_nbr    : store number (1-54)
    family       : product family string (e.g. 'BEVERAGES')
    onpromotion  : 0 or 1
    oil_price    : WTI crude price to assume; defaults to most recent known value
    data_dir     : path to raw data (for encoding lookup)

    Returns
    -------
    DataFrame with all FEATURE_COLUMNS columns, one row per date.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(dates) == 0:
        raise ValueError("end_date must be after start_date.")

    # --- Store metadata ---
    store_meta = _get_store_meta(data_dir)
    row = store_meta[store_meta["store_nbr"] == store_nbr]
    if row.empty:
        raise ValueError(f"store_nbr {store_nbr} not found in stores.csv.")
    store_type = row["type"].values[0]
    cluster = int(row["cluster"].values[0])

    # --- Encoding ---
    family_codes = _get_family_codes(data_dir)
    type_codes = _get_type_codes(data_dir)
    family_enc = family_codes.get(family, 0)
    type_enc = type_codes.get(store_type, 0)

    # --- Oil price ---
    if oil_price is None:
        oil_price = _get_recent_oil_price(data_dir)

    # --- Build frame ---
    df = pd.DataFrame({"date": dates})
    df["store_nbr"] = store_nbr
    df["family_enc"] = family_enc
    df["onpromotion"] = onpromotion
    df["type_enc"] = type_enc
    df["cluster"] = cluster
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["dcoilwtico"] = oil_price

    # Transactions: use 0 for future (unknown)
    df["transactions"] = 0

    # is_holiday: 0 for future (simplification)
    df["is_holiday"] = 0

    return df


def predict(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    store_nbr: int,
    family: str,
    onpromotion: int = 0,
    oil_price: float | None = None,
    model_path: Path = BEST_MODEL_PATH,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Generate a sales forecast for a future date range.

    Parameters
    ----------
    start_date, end_date : forecast window (inclusive)
    store_nbr            : store to forecast
    family               : product family
    onpromotion          : promotion flag (0/1)
    oil_price            : assumed WTI price (optional)
    model_path           : path to saved .joblib model
    data_dir             : path to raw data folder

    Returns
    -------
    DataFrame with columns: date, store_nbr, family, onpromotion,
                             predicted_sales
    """
    model = load_model(model_path)
    future_df = build_future_frame(
        start_date=start_date,
        end_date=end_date,
        store_nbr=store_nbr,
        family=family,
        onpromotion=onpromotion,
        oil_price=oil_price,
        data_dir=data_dir,
    )

    X = future_df[FEATURE_COLUMNS].values
    raw_preds = model.predict(X)
    preds = np.clip(raw_preds, 0, None)  # sales ≥ 0

    result = future_df[["date"]].copy()
    result["store_nbr"] = store_nbr
    result["family"] = family
    result["onpromotion"] = onpromotion
    result["predicted_sales"] = preds.round(2)

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running predictor smoke test…")
    forecast = predict(
        start_date="2017-08-16",
        end_date="2017-09-15",
        store_nbr=1,
        family="BEVERAGES",
        onpromotion=1,
    )
    print(forecast.head(10).to_string(index=False))
    print(f"\n✓ Forecast shape: {forecast.shape}")
