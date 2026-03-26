"""
tests/test_completeness.py — tests for fin_validator.checks.completeness.

All tests run against the dirty fixture (which has known injected issues):
- ~15 % nulls in BID and ASK (liquidity / after-hours gaps)
- 2 price spike outliers in TRDPRC_1 (flash crash / bad tick)
- 3 ACVOL_UNS rows stored as comma-formatted strings
- 2 malformed Date values ('N/A', empty string)
- 2 invalid RIC codes (missing exchange suffix)
- 1 duplicate row

TR.EPS is intentionally sparse in both clean and dirty data (populated only on
earnings announcement dates, ~4 times per year per instrument).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the package importable without installing it
sys.path.insert(0, str(Path(__file__).parent.parent))

from fin_validator.checks.completeness import (
    flag_null_severity,
    null_rate_over_time,
    null_rate_per_column,
)
from tests.fixtures.generate import make_clean_df, make_dirty_df

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dirty_df() -> pd.DataFrame:
    """Return the dirty DataFrame (generated in-memory for speed)."""
    return make_dirty_df()


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    """Return the clean DataFrame."""
    return make_clean_df()


@pytest.fixture(scope="module")
def dirty_csv_df() -> pd.DataFrame:
    """Return the dirty DataFrame loaded from the CSV fixture on disk."""
    path = Path(__file__).parent / "fixtures" / "sample_dirty.csv"
    return pd.read_csv(path)


# ── null_rate_per_column ──────────────────────────────────────────────────────


class TestNullRatePerColumn:
    def test_returns_dict_keyed_by_columns(self, dirty_df):
        rates = null_rate_per_column(dirty_df)
        assert isinstance(rates, dict)
        assert set(rates.keys()) == set(dirty_df.columns)

    def test_all_values_between_0_and_1(self, dirty_df):
        rates = null_rate_per_column(dirty_df)
        for col, rate in rates.items():
            assert 0.0 <= rate <= 1.0, f"Rate out of range for {col}: {rate}"

    def test_bid_null_rate_around_15_percent(self, dirty_df):
        """BID should have ~15 % nulls injected (liquidity gap simulation)."""
        rates = null_rate_per_column(dirty_df)
        bid_rate = rates["BID"]
        # Accept 10–20 % to account for rounding with N~250
        assert (
            0.10 <= bid_rate <= 0.20
        ), f"Expected BID null rate ~15%, got {bid_rate:.1%}"

    def test_clean_df_has_zero_nulls_except_eps(self, clean_df):
        """All columns should have zero nulls except TR.EPS, which is naturally
        sparse (populated only on quarterly earnings announcement dates)."""
        rates = null_rate_per_column(clean_df)
        for col, rate in rates.items():
            if col == "TR.EPS":
                continue  # TR.EPS is intentionally sparse in realistic LSEG data
            assert rate == 0.0, f"Clean df has unexpected nulls in {col}: {rate:.1%}"

    def test_ric_column_no_nulls(self, dirty_df):
        rates = null_rate_per_column(dirty_df)
        assert rates["RIC"] == 0.0

    def test_empty_dataframe_returns_empty_dict(self):
        df = pd.DataFrame()
        assert null_rate_per_column(df) == {}

    def test_all_null_column(self):
        df = pd.DataFrame({"A": [None, None, None], "B": [1, 2, 3]})
        rates = null_rate_per_column(df)
        assert rates["A"] == 1.0
        assert rates["B"] == 0.0


# ── null_rate_over_time ───────────────────────────────────────────────────────


class TestNullRateOverTime:
    def test_returns_dict_keyed_by_columns(self, dirty_df):
        result = null_rate_over_time(dirty_df, "Date")
        assert isinstance(result, dict)
        # Date column itself should not appear as a data column
        assert "Date" not in result

    def test_each_value_is_list_of_dicts(self, dirty_df):
        result = null_rate_over_time(dirty_df, "Date")
        for col, entries in result.items():
            assert isinstance(entries, list), f"Expected list for {col}"
            for entry in entries:
                assert "period" in entry
                assert "null_rate" in entry
                assert 0.0 <= entry["null_rate"] <= 1.0

    def test_missing_timestamp_col_returns_empty(self, dirty_df):
        result = null_rate_over_time(dirty_df, "nonexistent_col")
        assert result == {}

    def test_bid_has_nonzero_null_rate_in_some_period(self, dirty_df):
        """At least one daily bucket should have BID nulls in dirty data."""
        result = null_rate_over_time(dirty_df, "Date")
        assert "BID" in result
        any_null = any(e["null_rate"] > 0 for e in result["BID"])
        assert any_null, "Expected some null rates > 0 in BID over time"

    def test_clean_df_all_null_rates_zero_except_eps(self, clean_df):
        """TR.EPS is naturally sparse; all other columns should have zero nulls."""
        result = null_rate_over_time(clean_df, "Date")
        for col, entries in result.items():
            if col == "TR.EPS":
                continue
            for entry in entries:
                assert (
                    entry["null_rate"] == 0.0
                ), f"Clean df: unexpected null in {col} at {entry['period']}"


# ── flag_null_severity ────────────────────────────────────────────────────────


class TestFlagNullSeverity:
    def test_returns_correct_severity_labels(self):
        rates = {"A": 0.0, "B": 0.05, "C": 0.15, "D": 0.30, "E": 0.60}
        sev = flag_null_severity(rates)
        assert sev["A"] == "low"
        assert sev["B"] == "low"
        assert sev["C"] == "medium"
        assert sev["D"] == "high"
        assert sev["E"] == "high"

    def test_exactly_at_boundaries(self):
        # 10 % boundary: >10 % → medium, =10 % → low
        sev = flag_null_severity({"at_10": 0.10, "just_above_10": 0.101})
        assert sev["at_10"] == "low"
        assert sev["just_above_10"] == "medium"

        # 25 % boundary
        sev2 = flag_null_severity({"at_25": 0.25, "just_above_25": 0.251})
        assert sev2["at_25"] == "medium"
        assert sev2["just_above_25"] == "high"

    def test_bid_flagged_medium_or_high_on_dirty(self, dirty_df):
        """BID with ~15 % injected nulls should be flagged medium or high."""
        rates = null_rate_per_column(dirty_df)
        sev = flag_null_severity(rates)
        assert sev["BID"] in {"medium", "high"}

    def test_empty_input_returns_empty(self):
        assert flag_null_severity({}) == {}

    def test_all_valid_severity_values(self, dirty_df):
        rates = null_rate_per_column(dirty_df)
        sev = flag_null_severity(rates)
        valid = {"low", "medium", "high"}
        for col, s in sev.items():
            assert s in valid, f"Invalid severity '{s}' for {col}"


# ── Integration: CSV fixture on disk ─────────────────────────────────────────


class TestWithCSVFixture:
    def test_csv_fixture_loaded_and_checked(self, dirty_csv_df):
        rates = null_rate_per_column(dirty_csv_df)
        sev = flag_null_severity(rates)
        assert "BID" in rates
        assert sev["BID"] in {"medium", "high"}

    def test_csv_fixture_null_rate_over_time(self, dirty_csv_df):
        result = null_rate_over_time(dirty_csv_df, "Date")
        assert isinstance(result, dict)
        assert len(result) > 0
