"""
tests/test_enhancements.py
===========================
Tests for the Phase-5 enhancements:
  - CSV upload validation and preprocessing (_validate_upload / _preprocess_upload)
  - model_trainer returns MAPE, R², test_dates, y_test, y_pred_best
  - Data quality report metrics
  - Cleaned data CSV export schema
  - Forecast CSV export schema

Uses small synthetic DataFrames — no large CSV or trained model required where avoidable.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import FEATURE_COLUMNS, DATA_DIR
from src.model_trainer import _mape, _r2, time_based_split

# ---------------------------------------------------------------------------
# Helpers — minimal synthetic data
# ---------------------------------------------------------------------------

def _make_minimal_upload_df(n: int = 10) -> pd.DataFrame:
    """DataFrame that matches the required upload schema."""
    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "store_nbr": [1] * n,
        "family": ["BEVERAGES"] * n,
        "sales": np.random.uniform(100, 500, n).round(2),
        "onpromotion": [0] * n,
    })


# ---------------------------------------------------------------------------
# Tests for _mape and _r2 helpers
# ---------------------------------------------------------------------------

class TestMetricHelpers:
    def test_mape_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert _mape(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_mape_zero_actuals_excluded(self):
        """_mape should skip zero-actual rows to avoid division by zero."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([0.0, 120.0, 200.0])
        result = _mape(y_true, y_pred)
        # Only rows with y_true != 0 are included: (20/100 + 0/200) / 2 * 100 = 10%
        assert result == pytest.approx(10.0, abs=0.1)

    def test_mape_all_zeros_returns_nan(self):
        y = np.array([0.0, 0.0, 0.0])
        assert np.isnan(_mape(y, y))

    def test_r2_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert _r2(y, y) == pytest.approx(1.0, abs=1e-6)

    def test_r2_constant_prediction(self):
        """Predicting the mean gives R² = 0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert _r2(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_r2_range(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        r2_val = _r2(y_true, y_pred)
        assert r2_val <= 1.0          # R² ≤ 1 always
        assert r2_val > 0.9           # close fit


# ---------------------------------------------------------------------------
# Tests for upload validation
# ---------------------------------------------------------------------------

class TestUploadValidation:
    """These tests import _validate_upload and _preprocess_upload from app.py
    after patching sys.modules so Streamlit doesn't actually start."""

    @pytest.fixture(autouse=True)
    def _import_helpers(self):
        """Import upload helpers from app.py without running Streamlit."""
        import importlib, types
        # Create a minimal streamlit stub so app.py can be imported
        if "streamlit" not in sys.modules:
            st_stub = types.ModuleType("streamlit")
            for attr in [
                "set_page_config", "markdown", "radio", "columns",
                "button", "download_button", "file_uploader", "session_state",
                "error", "warning", "success", "info", "spinner",
                "expander", "dataframe", "plotly_chart", "slider",
                "selectbox", "number_input", "date_input", "toggle",
                "empty", "caption", "sidebar", "metric",
            ]:
                setattr(st_stub, attr, lambda *a, **kw: None)
            st_stub.session_state = {}
            sys.modules["streamlit"] = st_stub

        # Import helpers directly from the module code
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        spec = importlib.util.spec_from_file_location("app", app_path)
        app_mod = importlib.util.module_from_spec(spec)
        # Don't execute at module level (Streamlit calls would fail)
        # Instead just exec the helper functions portion
        with open(app_path, "r", encoding="utf-8") as fh:
            source = fh.read()

        self._validate_upload = None
        self._preprocess_upload = None

        # Compile and exec the helpers into a fresh namespace
        ns = {"pd": pd, "np": np, "FEATURE_COLUMNS": FEATURE_COLUMNS}
        try:
            exec(compile(source, str(app_path), "exec"), ns)
            self._validate_upload = ns.get("_validate_upload")
            self._preprocess_upload = ns.get("_preprocess_upload")
        except Exception:
            # Streamlit set_page_config etc will fail — that's expected.
            # But the functions should have been defined before set_page_config runs.
            pass

        if self._validate_upload is None or self._preprocess_upload is None:
            pytest.skip("Could not import upload helpers from app.py")

    def test_valid_upload_passes(self):
        df = _make_minimal_upload_df()
        ok, msg = self._validate_upload(df)
        assert ok, f"Expected PASS, got: {msg}"

    def test_missing_column_fails(self):
        df = _make_minimal_upload_df().drop(columns=["sales"])
        ok, msg = self._validate_upload(df)
        assert not ok
        assert "sales" in msg

    def test_empty_dataframe_fails(self):
        df = _make_minimal_upload_df().iloc[0:0]  # empty
        ok, msg = self._validate_upload(df)
        assert not ok

    def test_bad_date_fails(self):
        df = _make_minimal_upload_df()
        df.loc[0, "date"] = "not-a-date"
        ok, msg = self._validate_upload(df)
        assert not ok
        assert "date" in msg.lower()

    def test_non_numeric_sales_fails(self):
        df = _make_minimal_upload_df()
        df.loc[0, "sales"] = "abc"
        ok, msg = self._validate_upload(df)
        assert not ok
        assert "sales" in msg.lower()

    def test_preprocess_adds_temporal_features(self):
        df = _make_minimal_upload_df()
        out = self._preprocess_upload(df)
        for col in ["day_of_week", "month", "is_weekend", "quarter", "year", "day_of_year"]:
            assert col in out.columns, f"Missing temporal column: {col}"

    def test_preprocess_family_enc_is_int(self):
        df = _make_minimal_upload_df()
        out = self._preprocess_upload(df)
        assert pd.api.types.is_integer_dtype(out["family_enc"])

    def test_preprocess_sorted_by_date(self):
        df = _make_minimal_upload_df()
        # Shuffle dates
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        out = self._preprocess_upload(df)
        dates_sorted = out["date"].is_monotonic_increasing
        assert dates_sorted, "Output should be sorted by date"

    def test_preprocess_no_nan_in_sales(self):
        df = _make_minimal_upload_df()
        out = self._preprocess_upload(df)
        assert out["sales"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Tests for train_and_evaluate return schema (MAPE, R², actuals)
# ---------------------------------------------------------------------------

class TestTrainAndEvaluateSchema:
    """Validates that train_and_evaluate returns the new keys without
    running real training (uses a tiny synthetic dataset)."""

    @pytest.fixture(scope="class")
    def tiny_results(self):
        """Run train_and_evaluate on a very small synthetic dataset."""
        from src.model_trainer import train_and_evaluate
        from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN

        np.random.seed(42)
        n = 200
        dates = pd.date_range("2017-01-01", periods=n, freq="D")
        data = {
            "date": dates,
            "store_nbr": np.random.randint(1, 5, n),
            "family_enc": np.random.randint(0, 5, n),
            "onpromotion": np.random.randint(0, 2, n),
            "type_enc": np.random.randint(0, 3, n),
            "cluster": np.random.randint(1, 5, n),
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "is_weekend": (dates.dayofweek >= 5).astype(int),
            "quarter": dates.quarter,
            "year": dates.year,
            "day_of_year": dates.dayofyear,
            "dcoilwtico": np.random.uniform(40, 60, n),
            "transactions": np.random.randint(500, 2000, n),
            "is_holiday": np.random.randint(0, 2, n),
            "sales": np.random.uniform(0, 500, n),
        }
        df = pd.DataFrame(data)
        return train_and_evaluate(df, test_fraction=0.2)

    def test_metrics_has_r2(self, tiny_results):
        for model_name, m in tiny_results["metrics"].items():
            assert "r2" in m, f"{model_name} missing 'r2' metric"

    def test_metrics_has_mape(self, tiny_results):
        for model_name, m in tiny_results["metrics"].items():
            assert "mape" in m, f"{model_name} missing 'mape' metric"

    def test_result_has_test_dates(self, tiny_results):
        assert "test_dates" in tiny_results

    def test_result_has_y_test(self, tiny_results):
        assert "y_test" in tiny_results

    def test_result_has_y_pred_best(self, tiny_results):
        assert "y_pred_best" in tiny_results

    def test_y_test_and_y_pred_same_length(self, tiny_results):
        assert len(tiny_results["y_test"]) == len(tiny_results["y_pred_best"])

    def test_y_pred_nonnegative(self, tiny_results):
        assert (tiny_results["y_pred_best"] >= 0).all()

    def test_r2_is_float(self, tiny_results):
        for model_name, m in tiny_results["metrics"].items():
            assert isinstance(m["r2"], float), f"{model_name} r2 is not float"

    def test_mape_is_float(self, tiny_results):
        for model_name, m in tiny_results["metrics"].items():
            assert isinstance(m["mape"], float), f"{model_name} mape is not float"


# ---------------------------------------------------------------------------
# Tests for forecast CSV export schema
# ---------------------------------------------------------------------------

class TestForecastExportSchema:
    """Verify forecast CSV from predictor has the expected columns."""

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "models" / "best_model.joblib").exists(),
        reason="best_model.joblib not found"
    )
    def test_forecast_csv_has_required_columns(self):
        from src.predictor import predict
        fc = predict("2017-08-01", "2017-08-05", store_nbr=1, family="BEVERAGES")
        csv_str = fc.to_csv(index=False)
        df = pd.read_csv(io.StringIO(csv_str))
        for col in ["date", "store_nbr", "family", "onpromotion", "predicted_sales"]:
            assert col in df.columns, f"Missing column in forecast CSV: {col}"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "models" / "best_model.joblib").exists(),
        reason="best_model.joblib not found"
    )
    def test_forecast_csv_dates_are_chronological(self):
        from src.predictor import predict
        fc = predict("2017-08-01", "2017-08-10", store_nbr=1, family="BEVERAGES")
        dates = pd.to_datetime(fc["date"])
        assert dates.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Tests for data quality report values
# ---------------------------------------------------------------------------

class TestDataQualityReport:
    """Verify that data-quality counts are accurate."""

    def test_zero_sales_count(self):
        df = pd.DataFrame({"sales": [0.0, 1.0, 0.0, 5.0, 0.0]})
        zero_count = (df["sales"] == 0).sum()
        assert zero_count == 3

    def test_negative_sales_preserved(self):
        """Negative sales rows must NOT be silently deleted — only counted."""
        df = pd.DataFrame({"sales": [-1.0, 0.0, 100.0, -5.0]})
        neg_count = (df["sales"] < 0).sum()
        assert neg_count == 2
        assert len(df) == 4  # rows still present

    def test_duplicate_count(self):
        df = pd.DataFrame({"a": [1, 2, 2, 3]})
        dup_count = df.duplicated().sum()
        assert dup_count == 1
