"""
fin_validator.checks.consistency — consistency checks for financial DataFrames.

Detects:
- Columns that are numeric in intent but stored as strings (type mismatches).
- Malformed timestamps (non-parseable date strings).
- RIC codes that do not match the expected format (alphanumeric + '.' + suffix).
- Duplicate rows.

All public functions accept a pandas DataFrame and return a typed dict so
they can be used standalone or composed inside DataQualityReport.
"""

from __future__ import annotations

import re

import pandas as pd

_RIC_PATTERN = re.compile(r"^[A-Za-z0-9]+\.[A-Z]{1,4}$")


def numeric_string_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that appear numeric but are stored as object/string dtype.

    A column is flagged when it has object dtype and at least 80 % of its
    non-null values can be coerced to float.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    list[str]
        Names of columns that look numeric but are stored as strings.
    """
    flagged: list[str] = []
    for col in df.select_dtypes(include="object").columns:
        series = df[col].dropna()
        if series.empty:
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        ratio = coerced.notna().sum() / len(series)
        if ratio >= 0.8:
            flagged.append(col)
    return flagged


def malformed_timestamp_columns(
    df: pd.DataFrame, timestamp_cols: list[str] | None = None
) -> dict[str, int]:
    """Detect columns with unparseable timestamp strings.

    Parameters
    ----------
    df:
        Input DataFrame.
    timestamp_cols:
        Explicit list of column names to check.  If *None*, every object-dtype
        column whose name contains 'time', 'date', or 'timestamp' (case-insensitive)
        is checked.

    Returns
    -------
    dict[str, int]
        Mapping of column name → count of malformed values.
    """
    if timestamp_cols is None:
        pattern = re.compile(r"time|date|timestamp", re.IGNORECASE)
        timestamp_cols = [c for c in df.columns if pattern.search(c)]

    result: dict[str, int] = {}
    for col in timestamp_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        parsed = pd.to_datetime(series, errors="coerce", utc=False)
        bad_count = int(parsed.isna().sum())
        if bad_count > 0:
            result[col] = bad_count
    return result


def invalid_ric_rows(df: pd.DataFrame, ric_col: str = "RIC") -> pd.Index:
    """Return the index of rows whose RIC code does not match the expected format.

    Expected format: alphanumeric characters, a single dot, then 1-4 uppercase
    letters (e.g. ``MSFT.O``, ``VOD.L``).

    Parameters
    ----------
    df:
        Input DataFrame.
    ric_col:
        Name of the RIC column.

    Returns
    -------
    pd.Index
        Index of invalid rows (empty Index if the column is absent).
    """
    if ric_col not in df.columns:
        return pd.Index([])
    mask = df[ric_col].dropna().apply(lambda v: not bool(_RIC_PATTERN.match(str(v))))
    return df[ric_col].dropna()[mask].index


def duplicate_row_count(df: pd.DataFrame) -> int:
    """Return the number of fully-duplicated rows (excluding the first occurrence).

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    int
        Count of duplicate rows.
    """
    return int(df.duplicated().sum())


def run_all(df: pd.DataFrame, ric_col: str = "RIC") -> dict:
    """Run all consistency checks and return a combined result dict.

    Parameters
    ----------
    df:
        Input DataFrame.
    ric_col:
        Name of the RIC column.

    Returns
    -------
    dict
        Keys: ``numeric_string_columns``, ``malformed_timestamp_columns``,
        ``invalid_ric_count``, ``duplicate_row_count``.
    """
    return {
        "numeric_string_columns": numeric_string_columns(df),
        "malformed_timestamp_columns": malformed_timestamp_columns(df),
        "invalid_ric_count": len(invalid_ric_rows(df, ric_col=ric_col)),
        "duplicate_row_count": duplicate_row_count(df),
    }
