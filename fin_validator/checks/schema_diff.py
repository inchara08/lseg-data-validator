"""
fin_validator.checks.schema_diff — schema drift detection between two DataFrames.

Compares two DataFrames (e.g. snapshots taken at different times) and reports:
- Columns added or removed.
- Data-type changes per shared column.
- Null-rate delta per shared column (flags changes > 5 % by default).
- Value-range shifts (mean and std delta) for numeric columns.

All public functions accept two pandas DataFrames and return typed dicts.
"""

from __future__ import annotations

import pandas as pd


def added_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    """Return column names present in *df_b* but not in *df_a*.

    Parameters
    ----------
    df_a, df_b:
        DataFrames to compare (a = old, b = new).

    Returns
    -------
    list[str]
    """
    return [c for c in df_b.columns if c not in df_a.columns]


def removed_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    """Return column names present in *df_a* but not in *df_b*.

    Parameters
    ----------
    df_a, df_b:
        DataFrames to compare (a = old, b = new).

    Returns
    -------
    list[str]
    """
    return [c for c in df_a.columns if c not in df_b.columns]


def dtype_changes(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict[str, dict]:
    """Return dtype changes for columns shared by both DataFrames.

    Parameters
    ----------
    df_a, df_b:
        DataFrames to compare.

    Returns
    -------
    dict[str, dict]
        Mapping of column name → ``{"from": dtype_a, "to": dtype_b}`` for
        columns whose dtype changed.
    """
    shared = set(df_a.columns) & set(df_b.columns)
    result: dict[str, dict] = {}
    for col in shared:
        da = str(df_a[col].dtype)
        db = str(df_b[col].dtype)
        if da != db:
            result[col] = {"from": da, "to": db}
    return result


def null_rate_delta(
    df_a: pd.DataFrame, df_b: pd.DataFrame, flag_threshold: float = 0.05
) -> dict[str, dict]:
    """Return null-rate deltas for shared columns, flagging large changes.

    Parameters
    ----------
    df_a, df_b:
        DataFrames to compare.
    flag_threshold:
        Absolute null-rate change above which a column is flagged (default 5 %).

    Returns
    -------
    dict[str, dict]
        Mapping of column name →
        ``{"null_rate_a": float, "null_rate_b": float, "delta": float, "flagged": bool}``.
    """
    shared = set(df_a.columns) & set(df_b.columns)
    result: dict[str, dict] = {}
    for col in shared:
        rate_a = df_a[col].isna().mean()
        rate_b = df_b[col].isna().mean()
        delta = rate_b - rate_a
        result[col] = {
            "null_rate_a": round(float(rate_a), 4),
            "null_rate_b": round(float(rate_b), 4),
            "delta": round(float(delta), 4),
            "flagged": abs(delta) > flag_threshold,
        }
    return result


def value_range_shifts(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict[str, dict]:
    """Return mean and std deltas for shared numeric columns.

    Parameters
    ----------
    df_a, df_b:
        DataFrames to compare.

    Returns
    -------
    dict[str, dict]
        Mapping of column name →
        ``{"mean_a": float, "mean_b": float, "mean_delta": float,
           "std_a": float, "std_b": float, "std_delta": float}``.
    """
    shared_num = set(df_a.select_dtypes(include="number").columns) & set(
        df_b.select_dtypes(include="number").columns
    )
    result: dict[str, dict] = {}
    for col in shared_num:
        ma, sa = float(df_a[col].mean()), float(df_a[col].std())
        mb, sb = float(df_b[col].mean()), float(df_b[col].std())
        result[col] = {
            "mean_a": round(ma, 4),
            "mean_b": round(mb, 4),
            "mean_delta": round(mb - ma, 4),
            "std_a": round(sa, 4),
            "std_b": round(sb, 4),
            "std_delta": round(sb - sa, 4),
        }
    return result


def run_all(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    null_flag_threshold: float = 0.05,
) -> dict:
    """Run all schema diff checks and return a combined result dict.

    Parameters
    ----------
    df_a, df_b:
        DataFrames to compare (a = old snapshot, b = new snapshot).
    null_flag_threshold:
        Absolute null-rate delta above which a column is flagged.

    Returns
    -------
    dict
        Keys: ``added_columns``, ``removed_columns``, ``dtype_changes``,
        ``null_rate_delta``, ``value_range_shifts``.
    """
    return {
        "added_columns": added_columns(df_a, df_b),
        "removed_columns": removed_columns(df_a, df_b),
        "dtype_changes": dtype_changes(df_a, df_b),
        "null_rate_delta": null_rate_delta(
            df_a, df_b, flag_threshold=null_flag_threshold
        ),
        "value_range_shifts": value_range_shifts(df_a, df_b),
    }
