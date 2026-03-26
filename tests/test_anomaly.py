"""
tests/test_anomaly.py — tests for fin_validator.checks.anomaly.

Covers:
- zscore_outliers  (known injected outliers in dirty fixture)
- iqr_outliers     (same)
- spike_rows       (synthetic spikes)
- run_all          (integration)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fin_validator.checks.anomaly import (
    iqr_outliers,
    run_all,
    spike_rows,
    zscore_outliers,
)
from tests.fixtures.generate import make_clean_df, make_dirty_df


# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dirty_df() -> pd.DataFrame:
    """Dirty DataFrame with 3 injected outliers in TR.PriceClose (as object col)."""
    return make_dirty_df()


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    return make_clean_df()


@pytest.fixture
def simple_outlier_df() -> pd.DataFrame:
    """Minimal DataFrame with one obvious Z-score outlier."""
    values = [10.0] * 20 + [1000.0]  # 1000 is ~10σ above mean
    return pd.DataFrame({"price": values})


@pytest.fixture
def spike_df() -> pd.DataFrame:
    """DataFrame with a deliberate spike between rows."""
    return pd.DataFrame({"price": [100.0, 100.0, 100.0, 500.0, 100.0]})


# ── zscore_outliers ───────────────────────────────────────────────────────────


class TestZscoreOutliers:
    def test_returns_dict(self, clean_df):
        result = zscore_outliers(clean_df)
        assert isinstance(result, dict)

    def test_detects_obvious_outlier(self, simple_outlier_df):
        result = zscore_outliers(simple_outlier_df, threshold=3.0)
        assert "price" in result
        assert 20 in result["price"]  # row 20 is the outlier

    def test_clean_df_no_outliers(self, clean_df):
        """A single-instrument slice of clean data should have no Z-score outliers.

        The full multi-instrument fixture spans instruments with very different
        price scales (e.g. LSEG.L ~9,800p vs AAPL.O ~$185), which produces
        expected cross-instrument Z-score outliers.  We test on one instrument
        to verify that within a realistic price series there are no spurious flags.
        """
        df_single = clean_df[clean_df["RIC"] == "AAPL.O"].select_dtypes("number")
        result = zscore_outliers(df_single, threshold=3.0)
        for col, rows in result.items():
            assert len(rows) == 0, f"Unexpected outlier in clean AAPL.O '{col}': {rows}"

    def test_returns_list_of_ints(self, simple_outlier_df):
        result = zscore_outliers(simple_outlier_df)
        for col, rows in result.items():
            assert all(isinstance(i, int) for i in rows)

    def test_skips_non_numeric_columns(self, dirty_df):
        """RIC (string) column should never appear in the outlier dict."""
        result = zscore_outliers(dirty_df)
        assert "RIC" not in result

    def test_skips_short_series(self):
        """Columns with fewer than 3 non-null values should be skipped."""
        df = pd.DataFrame({"tiny": [1.0, 2.0]})
        assert zscore_outliers(df) == {}

    def test_custom_threshold(self, simple_outlier_df):
        """Very high threshold should suppress the outlier."""
        result = zscore_outliers(simple_outlier_df, threshold=100.0)
        assert "price" not in result

    def test_ignores_nan_values(self):
        """NaN rows should be excluded from Z-score computation without error."""
        values = [10.0] * 18 + [None, None, 1000.0]
        df = pd.DataFrame({"price": values})
        result = zscore_outliers(df, threshold=3.0)
        assert "price" in result


# ── iqr_outliers ──────────────────────────────────────────────────────────────


class TestIqrOutliers:
    def test_returns_dict(self, clean_df):
        assert isinstance(iqr_outliers(clean_df), dict)

    def test_detects_obvious_outlier(self, simple_outlier_df):
        result = iqr_outliers(simple_outlier_df, multiplier=1.5)
        assert "price" in result
        assert 20 in result["price"]

    def test_clean_df_has_no_iqr_outliers(self, clean_df):
        """A single-instrument slice of clean data should have no IQR outliers.

        The full multi-instrument fixture mixes instruments with radically different
        value ranges in the same column (e.g. TSLA.O volume ~104M vs LSEG.L ~1.2M),
        which produces expected cross-instrument IQR outliers by design.
        """
        df_single = clean_df[clean_df["RIC"] == "AAPL.O"].select_dtypes("number")
        result = iqr_outliers(df_single, multiplier=1.5)
        for col, rows in result.items():
            assert len(rows) == 0, f"Unexpected IQR outlier in clean AAPL.O '{col}': {rows}"

    def test_returns_list_of_ints(self, simple_outlier_df):
        result = iqr_outliers(simple_outlier_df)
        for col, rows in result.items():
            assert all(isinstance(i, int) for i in rows)

    def test_skips_non_numeric_columns(self, dirty_df):
        result = iqr_outliers(dirty_df)
        assert "RIC" not in result

    def test_skips_very_short_series(self):
        df = pd.DataFrame({"tiny": [1.0, 2.0, 3.0]})
        assert iqr_outliers(df) == {}

    def test_custom_multiplier_suppresses_outlier(self):
        """A large multiplier should flag nothing when the outlier is within the fence."""
        # Use a dataset with a genuine spread so IQR > 0
        df = pd.DataFrame({"price": [10.0, 15.0, 20.0, 25.0, 30.0, 1000.0]})
        # With multiplier=1.5 the 1000 should be flagged
        assert "price" in iqr_outliers(df, multiplier=1.5)
        # With an extreme multiplier it should NOT be flagged
        result = iqr_outliers(df, multiplier=10000.0)
        assert "price" not in result

    def test_tight_multiplier_flags_more(self, clean_df):
        """An extremely tight fence (0.01) should flag many rows."""
        result = iqr_outliers(clean_df, multiplier=0.01)
        # With a nearly-zero fence almost every value is an outlier
        total_flagged = sum(len(v) for v in result.values())
        assert total_flagged > 0


# ── spike_rows ────────────────────────────────────────────────────────────────


class TestSpikeRows:
    def test_detects_spike(self, spike_df):
        result = spike_rows(spike_df, pct_change_threshold=0.5)
        assert "price" in result
        assert 3 in result["price"]  # row 3 jumps 400 %

    def test_no_spikes_in_stable_series(self):
        df = pd.DataFrame({"price": [100.0, 101.0, 99.0, 100.5]})
        result = spike_rows(df, pct_change_threshold=0.5)
        assert "price" not in result

    def test_returns_dict_of_lists(self, spike_df):
        result = spike_rows(spike_df)
        assert isinstance(result, dict)
        for col, rows in result.items():
            assert isinstance(rows, list)

    def test_returns_list_of_ints(self, spike_df):
        result = spike_rows(spike_df, pct_change_threshold=0.5)
        for col, rows in result.items():
            assert all(isinstance(i, int) for i in rows)

    def test_skips_non_numeric(self, dirty_df):
        result = spike_rows(dirty_df)
        assert "RIC" not in result

    def test_skips_single_value_series(self):
        df = pd.DataFrame({"price": [42.0]})
        assert spike_rows(df) == {}

    def test_custom_threshold(self, spike_df):
        """With a very high threshold (500 %), the 400 % spike should not be flagged."""
        result = spike_rows(spike_df, pct_change_threshold=5.0)
        assert "price" not in result

    def test_ignores_nan_in_series(self):
        df = pd.DataFrame({"price": [100.0, None, 100.0, 500.0]})
        result = spike_rows(df, pct_change_threshold=0.5)
        assert "price" in result


# ── run_all ───────────────────────────────────────────────────────────────────


class TestRunAll:
    def test_returns_expected_keys(self, clean_df):
        result = run_all(clean_df)
        assert set(result.keys()) == {"zscore_outliers", "iqr_outliers", "spike_rows"}

    def test_values_are_dicts(self, clean_df):
        result = run_all(clean_df)
        for key in ("zscore_outliers", "iqr_outliers", "spike_rows"):
            assert isinstance(result[key], dict)

    def test_thresholds_forwarded(self):
        """Passing extreme thresholds should suppress all outliers."""
        # Use a dataset with real spread so threshold suppression is meaningful
        df = pd.DataFrame({"price": [10.0, 15.0, 20.0, 25.0, 30.0, 1000.0]})
        result = run_all(
            df,
            zscore_threshold=1000.0,
            iqr_multiplier=10000.0,
            spike_threshold=1000.0,
        )
        assert result["zscore_outliers"] == {}
        assert result["iqr_outliers"] == {}
        assert result["spike_rows"] == {}

    def test_run_all_detects_anomalies_in_spiked_data(self, spike_df):
        """A DataFrame with a clear spike should surface at least one anomaly."""
        result = run_all(spike_df, spike_threshold=0.5)
        total_spikes = sum(len(v) for v in result["spike_rows"].values())
        assert total_spikes > 0, "Expected spike_rows to detect the injected spike"
