"""
tests/test_data_loader.py
=========================
Tests for data loading, preprocessing, and feature engineering.
Uses lightweight synthetic DataFrames — no large CSV files required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    merge_tables,
    preprocess,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic data
# ---------------------------------------------------------------------------

def _make_train(n=20) -> pd.DataFrame:
    """Minimal train.csv stand-in."""
    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "id": range(n),
        "date": dates,
        "store_nbr": [1] * n,
        "family": ["BEVERAGES"] * (n // 2) + ["GROCERY I"] * (n - n // 2),
        "sales": np.random.uniform(100, 500, n),
        "onpromotion": np.random.randint(0, 5, n),
    })


def _make_stores() -> pd.DataFrame:
    return pd.DataFrame({
        "store_nbr": [1, 2],
        "city": ["Quito", "Guayaquil"],
        "state": ["Pichincha", "Guayas"],
        "type": ["A", "B"],
        "cluster": [1, 2],
    })


def _make_oil(n=25) -> pd.DataFrame:
    dates = pd.date_range("2016-12-25", periods=n, freq="D")
    prices = [np.nan if i in (0, 1) else 50.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame({"date": dates, "dcoilwtico": prices})


def _make_holidays(n=5) -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", periods=n, freq="7D")
    return pd.DataFrame({
        "date": dates,
        "type": ["Holiday"] * n,
        "locale": ["National"] * n,
        "locale_name": ["Ecuador"] * n,
        "description": ["Test Holiday"] * n,
        "transferred": [False] * n,
    })


def _make_transactions(n=20) -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "store_nbr": [1] * n,
        "transactions": np.random.randint(500, 2000, n),
    })


@pytest.fixture
def synthetic_dfs():
    return {
        "train": _make_train(),
        "stores": _make_stores(),
        "oil": _make_oil(),
        "holidays": _make_holidays(),
        "transactions": _make_transactions(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMergeTables:
    def test_merge_produces_same_row_count_as_train(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        assert len(merged) == len(synthetic_dfs["train"])

    def test_merge_adds_type_column(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        assert "type" in merged.columns

    def test_merge_adds_cluster_column(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        assert "cluster" in merged.columns

    def test_merge_adds_oil_column(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        assert "dcoilwtico" in merged.columns

    def test_merge_adds_is_holiday_column(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        assert "is_holiday" in merged.columns

    def test_merge_adds_transactions_column(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        assert "transactions" in merged.columns

    def test_holiday_dedup_no_row_explosion(self, synthetic_dfs):
        """Duplicate holiday dates should not inflate row count."""
        holidays = synthetic_dfs["holidays"]
        dup_holidays = pd.concat([holidays, holidays], ignore_index=True)
        dfs = {**synthetic_dfs, "holidays": dup_holidays}
        merged = merge_tables(dfs)
        assert len(merged) == len(dfs["train"])

    def test_is_holiday_is_binary(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        merged_proc = preprocess(merged)
        assert set(merged_proc["is_holiday"].unique()).issubset({0, 1})


class TestPreprocess:
    def test_all_feature_columns_present(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        processed = preprocess(merged)
        for col in FEATURE_COLUMNS:
            assert col in processed.columns, f"Missing feature column: {col}"

    def test_no_nan_in_feature_columns(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        processed = preprocess(merged)
        missing = processed[FEATURE_COLUMNS].isna().sum().sum()
        assert missing == 0, f"Feature columns have {missing} NaN values after preprocessing"

    def test_oil_price_forward_filled(self, synthetic_dfs):
        """Oil prices with leading NaNs should be back-filled."""
        merged = merge_tables(synthetic_dfs)
        processed = preprocess(merged)
        assert processed["dcoilwtico"].isna().sum() == 0

    def test_family_enc_is_integer(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        processed = preprocess(merged)
        assert pd.api.types.is_integer_dtype(processed["family_enc"])

    def test_type_enc_is_integer(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        processed = preprocess(merged)
        assert pd.api.types.is_integer_dtype(processed["type_enc"])

    def test_day_of_week_range(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        processed = preprocess(merged)
        assert processed["day_of_week"].between(0, 6).all()

    def test_month_range(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        processed = preprocess(merged)
        assert processed["month"].between(1, 12).all()

    def test_quarter_range(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        processed = preprocess(merged)
        assert processed["quarter"].between(1, 4).all()

    def test_is_weekend_binary(self, synthetic_dfs):
        merged = merge_tables(synthetic_dfs)
        processed = preprocess(merged)
        assert set(processed["is_weekend"].unique()).issubset({0, 1})

    def test_transactions_filled_not_nan(self, synthetic_dfs):
        """Transactions should be filled with store median — not NaN."""
        dfs = {**synthetic_dfs}
        # Introduce NaN transactions
        tx = dfs["transactions"].copy()
        tx.loc[0, "transactions"] = np.nan
        dfs["transactions"] = tx
        merged = merge_tables(dfs)
        processed = preprocess(merged)
        assert processed["transactions"].isna().sum() == 0


class TestEdgeCases:
    def test_empty_sales_column(self, synthetic_dfs):
        """A DataFrame with all-zero sales should still process cleanly."""
        dfs = {**synthetic_dfs}
        train = dfs["train"].copy()
        train["sales"] = 0.0
        dfs["train"] = train
        merged = merge_tables(dfs)
        processed = preprocess(merged)
        assert (processed["sales"] == 0).all()

    def test_single_family(self, synthetic_dfs):
        """Works even if only one product family exists."""
        dfs = {**synthetic_dfs}
        train = dfs["train"].copy()
        train["family"] = "BEVERAGES"
        dfs["train"] = train
        merged = merge_tables(dfs)
        processed = preprocess(merged)
        assert "family_enc" in processed.columns

    def test_missing_transactions_table(self):
        """Gracefully handles missing transactions (all NaN) via store median fill."""
        dfs = {
            "train": _make_train(),
            "stores": _make_stores(),
            "oil": _make_oil(),
            "holidays": _make_holidays(),
            "transactions": pd.DataFrame({"date": [], "store_nbr": [], "transactions": []}),
        }
        merged = merge_tables(dfs)
        # All transactions will be NaN — preprocess should fill with 0
        processed = preprocess(merged)
        assert processed["transactions"].isna().sum() == 0
