"""
model_trainer.py
================
Hybrid ML pipeline for the Sales Forecasting System.

Implements:
  - Time-based train/test split (no data leakage)
  - Baseline: LinearRegression
  - Primary: XGBRegressor
  - Evaluation: RMSE & MAE
  - Serialisation: saves best model to models/best_model.joblib
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when running this module directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def time_based_split(
    df: pd.DataFrame,
    test_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train/test sets using chronological ordering.

    The last `test_fraction` of unique dates becomes the test set.
    This prevents data leakage that random splitting would cause in
    time-series problems.

    Parameters
    ----------
    df            : preprocessed DataFrame (must have a 'date' column)
    test_fraction : fraction of dates to reserve for testing

    Returns
    -------
    (train_df, test_df)
    """
    sorted_dates = sorted(df["date"].unique())
    n_test = max(1, int(len(sorted_dates) * test_fraction))
    cutoff_date = sorted_dates[-n_test]

    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()
    return train_df, test_df


def _build_linear_pipeline() -> Pipeline:
    """Baseline: StandardScaler → LinearRegression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def _build_xgb_pipeline() -> Pipeline:
    """Primary: XGBRegressor (tree-based, no scaling needed but kept for API consistency)."""
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="rmse",
    )
    return Pipeline([("model", xgb)])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_evaluate(
    df: pd.DataFrame,
    test_fraction: float = 0.15,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Train both models, evaluate on the time-based test split, and save
    the best-performing model.

    Parameters
    ----------
    df                : preprocessed DataFrame from data_loader
    test_fraction     : fraction of dates reserved for the test set
    progress_callback : optional callable(msg: str) for progress reporting

    Returns
    -------
    dict with keys:
        'metrics'    : {model_name: {'rmse': float, 'mae': float}}
        'best_model' : name of the winning model ('LinearRegression' | 'XGBRegressor')
        'model_path' : Path to saved .joblib file
        'cutoff_date': datetime — boundary between train and test
        'train_size' : int
        'test_size'  : int
    """

    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # ── 1. Split ────────────────────────────────────────────────────────────
    _log("Splitting data (time-based)…")
    train_df, test_df = time_based_split(df, test_fraction=test_fraction)
    cutoff_date = test_df["date"].min()

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    _log(f"  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
    _log(f"  Cutoff date: {cutoff_date.date()}")

    # ── 2. Train models ─────────────────────────────────────────────────────
    models = {
        "LinearRegression": _build_linear_pipeline(),
        "XGBRegressor": _build_xgb_pipeline(),
    }

    metrics: dict[str, dict[str, float]] = {}

    for name, pipe in models.items():
        _log(f"Training {name}…")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        preds = np.clip(preds, 0, None)   # sales can't be negative

        rmse = _rmse(y_test, preds)
        mae = _mae(y_test, preds)
        metrics[name] = {"rmse": round(rmse, 4), "mae": round(mae, 4)}
        _log(f"  {name}: RMSE={rmse:.4f}  MAE={mae:.4f}")

    # ── 3. Select best (lowest RMSE) ────────────────────────────────────────
    best_name = min(metrics, key=lambda k: metrics[k]["rmse"])
    best_model = models[best_name]
    _log(f"Best model: {best_name} (RMSE={metrics[best_name]['rmse']:.4f})")

    # ── 4. Serialise ────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    _log(f"Model saved -> {BEST_MODEL_PATH}")

    return {
        "metrics": metrics,
        "best_model": best_name,
        "model_path": BEST_MODEL_PATH,
        "cutoff_date": cutoff_date,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }


def load_model(path: str | Path = BEST_MODEL_PATH) -> Pipeline:
    """Load and return a trained model pipeline from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. "
            "Please run Model Training first."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_loader import build_dataset

    print("Building dataset (this may take a minute)…")
    df = build_dataset()

    print("Training models…")
    results = train_and_evaluate(df, progress_callback=print)

    print("\n=== Results ===")
    for model, m in results["metrics"].items():
        print(f"  {model:20s}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")
    print(f"\nBest model : {results['best_model']}")
    print(f"Saved to   : {results['model_path']}")
