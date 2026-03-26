"""
tests/test_schema_diff.py — tests for fin_validator.checks.schema_diff.

Covers:
- added_columns / removed_columns
- dtype_changes
- null_rate_delta (including flag_threshold logic)
- value_range_shifts
- run_all integration
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fin_validator.checks.schema_diff import (
    added_columns,
    dtype_changes,
    null_rate_delta,
    removed_columns,
    run_all,
    value_range_shifts,
)
from tests.fixtures.generate import make_clean_df, make_dirty_df


# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def df_a() -> pd.DataFrame:
    """Clean snapshot — the 'old' baseline."""
    return make_clean_df()


@pytest.fixture(scope="module")
def df_b() -> pd.DataFrame:
    """Dirty snapshot — the 'new' data."""
    return make_dirty_df()


# ── added_columns ──────────────────────────────────────────────────────────────


class TestAddedColumns:
    def test_returns_empty_when_identical(self, df_a):
        assert added_columns(df_a, df_a) == []

    def test_detects_new_column(self, df_a):
        df_new = df_a.copy()
        df_new["TR.Dividend"] = 1.0
        result = added_columns(df_a, df_new)
        assert result == ["TR.Dividend"]

    def test_detects_multiple_new_columns(self, df_a):
        df_new = df_a.copy()
        df_new["Col1"] = 0
        df_new["Col2"] = 0
        result = added_columns(df_a, df_new)
        assert set(result) == {"Col1", "Col2"}

    def test_returns_list(self, df_a, df_b):
        assert isinstance(added_columns(df_a, df_b), list)

    def test_no_false_positives_for_removals(self, df_a):
        """Columns only in df_a should NOT appear in added."""
        df_sub = df_a.drop(columns=["TR.Volume"])
        assert "TR.Volume" not in added_columns(df_a, df_sub)


# ── removed_columns ───────────────────────────────────────────────────────────


class TestRemovedColumns:
    def test_returns_empty_when_identical(self, df_a):
        assert removed_columns(df_a, df_a) == []

    def test_detects_removed_column(self, df_a):
        df_sub = df_a.drop(columns=["TR.Volume"])
        result = removed_columns(df_a, df_sub)
        assert result == ["TR.Volume"]

    def test_detects_multiple_removed_columns(self, df_a):
        df_sub = df_a.drop(columns=["TR.Volume", "BID"])
        result = removed_columns(df_a, df_sub)
        assert set(result) == {"TR.Volume", "BID"}

    def test_no_false_positives_for_additions(self, df_a):
        """Columns only in df_b should NOT appear in removed."""
        df_new = df_a.copy()
        df_new["Extra"] = 0
        assert "Extra" not in removed_columns(df_a, df_new)


# ── dtype_changes ─────────────────────────────────────────────────────────────


class TestDtypeChanges:
    def test_returns_empty_when_identical(self, df_a):
        assert dtype_changes(df_a, df_a) == {}

    def test_detects_float_to_object(self, df_a):
        df_changed = df_a.copy()
        df_changed["TR.PriceClose"] = df_changed["TR.PriceClose"].astype(object)
        result = dtype_changes(df_a, df_changed)
        assert "TR.PriceClose" in result
        assert result["TR.PriceClose"]["from"] == "float64"
        assert result["TR.PriceClose"]["to"] == "object"

    def test_schema_both_from_and_to_keys(self, df_a):
        df_changed = df_a.copy()
        df_changed["TR.Volume"] = df_changed["TR.Volume"].astype(str)
        result = dtype_changes(df_a, df_changed)
        for col, info in result.items():
            assert "from" in info
            assert "to" in info

    def test_returns_dict(self, df_a, df_b):
        assert isinstance(dtype_changes(df_a, df_b), dict)

    def test_ignores_columns_not_in_both(self, df_a):
        df_extra = df_a.copy()
        df_extra["NewCol"] = 0
        # NewCol only in df_extra — should not appear in dtype_changes
        result = dtype_changes(df_a, df_extra)
        assert "NewCol" not in result


# ── null_rate_delta ───────────────────────────────────────────────────────────


class TestNullRateDelta:
    def test_returns_dict_for_all_shared_columns(self, df_a, df_b):
        shared = set(df_a.columns) & set(df_b.columns)
        result = null_rate_delta(df_a, df_b)
        assert set(result.keys()) == shared

    def test_entry_schema(self, df_a, df_b):
        result = null_rate_delta(df_a, df_b)
        for col, info in result.items():
            assert "null_rate_a" in info
            assert "null_rate_b" in info
            assert "delta" in info
            assert "flagged" in info

    def test_clean_to_clean_not_flagged(self, df_a):
        """Same DataFrame compared to itself — delta = 0, never flagged."""
        result = null_rate_delta(df_a, df_a)
        for col, info in result.items():
            assert info["delta"] == 0.0
            assert not info["flagged"]

    def test_bid_null_increase_flagged(self, df_a, df_b):
        """Clean df_a has 0 % BID nulls; dirty df_b has ~15 % — should be flagged."""
        result = null_rate_delta(df_a, df_b)
        assert result["BID"]["flagged"]
        assert result["BID"]["delta"] > 0

    def test_custom_threshold_changes_flagging(self, df_a, df_b):
        """With a very high threshold, the BID delta should NOT be flagged."""
        result = null_rate_delta(df_a, df_b, flag_threshold=0.99)
        assert not result["BID"]["flagged"]

    def test_values_are_rounded(self, df_a, df_b):
        result = null_rate_delta(df_a, df_b)
        for col, info in result.items():
            for key in ("null_rate_a", "null_rate_b", "delta"):
                assert len(str(info[key]).split(".")[-1]) <= 4


# ── value_range_shifts ────────────────────────────────────────────────────────


class TestValueRangeShifts:
    def test_returns_only_numeric_columns(self, df_a, df_b):
        result = value_range_shifts(df_a, df_b)
        numeric_shared = set(df_a.select_dtypes(include="number").columns) & set(
            df_b.select_dtypes(include="number").columns
        )
        assert set(result.keys()) == numeric_shared

    def test_entry_schema(self, df_a, df_b):
        result = value_range_shifts(df_a, df_b)
        for col, info in result.items():
            for key in ("mean_a", "mean_b", "mean_delta", "std_a", "std_b", "std_delta"):
                assert key in info

    def test_identical_df_zero_delta(self, df_a):
        result = value_range_shifts(df_a, df_a)
        for col, info in result.items():
            assert info["mean_delta"] == 0.0
            assert info["std_delta"] == 0.0

    def test_shifted_mean_reflected(self):
        """Artificially shift mean of one column and verify delta is positive."""
        df_x = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
        df_y = pd.DataFrame({"price": [110.0, 120.0, 130.0]})
        result = value_range_shifts(df_x, df_y)
        assert result["price"]["mean_delta"] == pytest.approx(100.0)

    def test_no_numeric_columns_returns_empty(self):
        df_str = pd.DataFrame({"name": ["a", "b"], "sym": ["x", "y"]})
        assert value_range_shifts(df_str, df_str) == {}


# ── run_all ───────────────────────────────────────────────────────────────────


class TestRunAll:
    def test_returns_all_top_level_keys(self, df_a, df_b):
        result = run_all(df_a, df_b)
        assert set(result.keys()) == {
            "added_columns",
            "removed_columns",
            "dtype_changes",
            "null_rate_delta",
            "value_range_shifts",
        }

    def test_identical_dfs_minimal_result(self, df_a):
        result = run_all(df_a, df_a)
        assert result["added_columns"] == []
        assert result["removed_columns"] == []
        assert result["dtype_changes"] == {}
        for col, info in result["null_rate_delta"].items():
            assert info["delta"] == 0.0

    def test_null_flag_threshold_forwarded(self, df_a, df_b):
        """Custom null_flag_threshold should propagate into null_rate_delta."""
        result_low = run_all(df_a, df_b, null_flag_threshold=0.01)
        result_high = run_all(df_a, df_b, null_flag_threshold=0.99)
        # With low threshold, BID should be flagged (dirty has ~15 % nulls); with high, not.
        assert result_low["null_rate_delta"]["BID"]["flagged"]
        assert not result_high["null_rate_delta"]["BID"]["flagged"]

    def test_dirty_vs_clean_shows_bid_null_delta(self, df_a, df_b):
        """BID null rate increases from 0 % (clean) to ~15 % (dirty)."""
        result = run_all(df_a, df_b)
        assert result["null_rate_delta"]["BID"]["delta"] > 0
