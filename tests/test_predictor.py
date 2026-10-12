"""
tests/test_predictor.py
=======================
Tests for the predictor module.
Uses the real saved model (best_model.joblib) and category_maps.json.
Does NOT require train.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import FEATURE_COLUMNS, DATA_DIR
from src.model_trainer import BEST_MODEL_PATH
from src.predictor import build_future_frame, predict, TRAINING_END_DATE

# Skip all tests if the model file is absent (e.g. first run before training)
pytestmark = pytest.mark.skipif(
    not BEST_MODEL_PATH.exists(),
    reason="best_model.joblib not found — run model training first"
)


class TestBuildFutureFrame:
    def test_returns_correct_number_of_rows(self):
        df = build_future_frame("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        assert len(df) == 7

    def test_all_feature_columns_present(self):
        df = build_future_frame("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_no_nan_in_features(self):
        df = build_future_frame("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        missing = df[FEATURE_COLUMNS].isna().sum().sum()
        assert missing == 0, f"Feature frame has {missing} NaN values"

    def test_transactions_not_zero(self):
        """Transactions should use store median, not 0."""
        df = build_future_frame("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        # Median transactions for store 1 is well above 0
        assert df["transactions"].mean() > 0, "Transactions should not be 0"

    def test_oil_price_is_scalar(self):
        df = build_future_frame("2017-08-01", "2017-08-07", store_nbr=1,
                                family="BEVERAGES", oil_price=55.0)
        assert (df["dcoilwtico"] == 55.0).all()

    def test_custom_oil_price(self):
        df = build_future_frame("2017-08-01", "2017-08-03", store_nbr=1,
                                family="BEVERAGES", oil_price=100.0)
        assert (df["dcoilwtico"] == 100.0).all()

    def test_onpromotion_flag(self):
        df0 = build_future_frame("2017-08-01", "2017-08-03", store_nbr=1,
                                 family="BEVERAGES", onpromotion=0)
        df1 = build_future_frame("2017-08-01", "2017-08-03", store_nbr=1,
                                 family="BEVERAGES", onpromotion=1)
        assert (df0["onpromotion"] == 0).all()
        assert (df1["onpromotion"] == 1).all()

    def test_date_range_is_daily(self):
        df = build_future_frame("2017-08-01", "2017-08-10", store_nbr=1, family="BEVERAGES")
        assert len(df) == 10
        diffs = pd.to_datetime(df["date"]).diff().dropna()
        assert (diffs == pd.Timedelta("1D")).all()

    def test_invalid_store_raises_error(self):
        with pytest.raises(ValueError, match="store_nbr"):
            build_future_frame("2017-08-01", "2017-08-03", store_nbr=999, family="BEVERAGES")

    def test_end_before_start_raises_error(self):
        with pytest.raises(ValueError):
            build_future_frame("2017-08-10", "2017-08-01", store_nbr=1, family="BEVERAGES")

    def test_day_of_week_range(self):
        df = build_future_frame("2017-08-01", "2017-08-14", store_nbr=1, family="BEVERAGES")
        assert df["day_of_week"].between(0, 6).all()

    def test_is_weekend_binary(self):
        df = build_future_frame("2017-08-01", "2017-08-14", store_nbr=1, family="BEVERAGES")
        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_weekend_flag_correct(self):
        # 2017-08-05 is Saturday, 2017-08-06 is Sunday
        df = build_future_frame("2017-08-05", "2017-08-06", store_nbr=1, family="BEVERAGES")
        assert (df["is_weekend"] == 1).all()

    def test_weekday_flag_correct(self):
        # 2017-08-07 is Monday
        df = build_future_frame("2017-08-07", "2017-08-11", store_nbr=1, family="BEVERAGES")
        assert (df["is_weekend"] == 0).all()

    def test_single_day_forecast(self):
        df = build_future_frame("2017-08-16", "2017-08-16", store_nbr=1, family="BEVERAGES")
        assert len(df) == 1


class TestPredict:
    def test_output_shape(self):
        fc = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        assert fc.shape == (7, 5)  # date, store_nbr, family, onpromotion, predicted_sales

    def test_output_columns(self):
        fc = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        expected = {"date", "store_nbr", "family", "onpromotion", "predicted_sales"}
        assert set(fc.columns) == expected

    def test_predictions_nonnegative(self):
        fc = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        assert (fc["predicted_sales"] >= 0).all(), "Predictions must be ≥ 0"

    def test_predictions_finite(self):
        fc = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        assert np.isfinite(fc["predicted_sales"]).all()

    def test_predictions_rounded_to_2dp(self):
        fc = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        # All values should be rounded to 2 decimal places
        rounded = fc["predicted_sales"].round(2)
        assert (fc["predicted_sales"] == rounded).all()

    def test_promotion_affects_predictions(self):
        """Promotion flag should change predictions (not necessarily increase them)."""
        fc_no = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES", onpromotion=0)
        fc_yes = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES", onpromotion=1)
        # Predictions should differ
        assert not (fc_no["predicted_sales"] == fc_yes["predicted_sales"]).all()

    def test_different_families_produce_different_results(self):
        fc1 = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        fc2 = predict("2017-08-01", "2017-08-07", store_nbr=1, family="GROCERY I")
        assert not (fc1["predicted_sales"] == fc2["predicted_sales"]).all()

    def test_different_stores_produce_different_results(self):
        fc1 = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        fc2 = predict("2017-08-01", "2017-08-07", store_nbr=2, family="BEVERAGES")
        assert not (fc1["predicted_sales"] == fc2["predicted_sales"]).all()

    def test_all_33_families_predict_without_error(self):
        """No family should cause a crash or produce NaN/negative values."""
        import json
        maps_path = DATA_DIR / "category_maps.json"
        if not maps_path.exists():
            pytest.skip("category_maps.json not found")
        with open(maps_path) as f:
            families = json.load(f)["families"]
        for fam in families:
            fc = predict("2017-08-01", "2017-08-03", store_nbr=1, family=fam)
            assert (fc["predicted_sales"] >= 0).all(), f"Negative prediction for {fam}"
            assert not fc["predicted_sales"].isna().any(), f"NaN prediction for {fam}"

    def test_training_end_date_constant(self):
        """TRAINING_END_DATE should be 2017-08-15."""
        assert TRAINING_END_DATE == pd.Timestamp("2017-08-15")

    def test_csv_export_valid(self):
        """The forecast DataFrame should be exportable to CSV without errors."""
        fc = predict("2017-08-01", "2017-08-07", store_nbr=1, family="BEVERAGES")
        csv_str = fc.to_csv(index=False)
        re_read = pd.read_csv(__import__("io").StringIO(csv_str))
        assert len(re_read) == 7
        assert "predicted_sales" in re_read.columns


class TestModelTrainerSplit:
    """Tests for time-based split logic without running full training."""

    def test_time_based_split_no_leakage(self):
        from src.model_trainer import time_based_split

        df = pd.DataFrame({
            "date": pd.date_range("2017-01-01", periods=100, freq="D"),
            "sales": np.random.rand(100),
        })
        train, test = time_based_split(df, test_fraction=0.2)
        # Ensure train and test do not overlap in dates
        assert train["date"].max() < test["date"].min()

    def test_time_based_split_sizes(self):
        from src.model_trainer import time_based_split

        n_dates = 100
        df = pd.DataFrame({
            "date": pd.date_range("2017-01-01", periods=n_dates, freq="D"),
            "sales": np.random.rand(n_dates),
        })
        train, test = time_based_split(df, test_fraction=0.2)
        total = len(train) + len(test)
        assert total == n_dates

    def test_time_based_split_chronological(self):
        from src.model_trainer import time_based_split

        df = pd.DataFrame({
            "date": pd.date_range("2017-01-01", periods=50, freq="D"),
            "sales": np.random.rand(50),
        })
        train, test = time_based_split(df, test_fraction=0.1)
        # Test set should be the LAST portion of data
        assert test["date"].min() >= train["date"].max()
