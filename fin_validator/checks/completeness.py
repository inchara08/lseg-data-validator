"""
fin_validator.checks.completeness — null-rate and severity checks.

Public functions
----------------
null_rate_per_column(df)         -> dict[str, float]
null_rate_over_time(df, ts_col)  -> dict[str, list]
flag_null_severity(null_rates)   -> dict[str, str]

All functions accept a pandas DataFrame (or a pre-computed dict for
flag_null_severity) and return typed dicts so they can be used standalone
or composed inside DataQualityReport.
"""

from __future__ import annotations

import pandas as pd


def null_rate_per_column(df: pd.DataFrame) -> dict[str, float]:
    """Return the fraction of null values per column.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    dict[str, float]
        Mapping of column name → null rate in [0, 1].
    """
    return {col: float(df[col].isna().mean()) for col in df.columns}


def null_rate_over_time(
    df: pd.DataFrame,
    timestamp_col: str,
    freq: str = "D",
) -> dict[str, list]:
    """Return per-column null rates grouped by time period.

    Useful for detecting data-feed outages (a sudden spike in nulls on a
    particular date indicates a gap in the upstream feed).

    Parameters
    ----------
    df:
        Input DataFrame.  Must contain *timestamp_col*.
    timestamp_col:
        Name of the column to use for time grouping.
    freq:
        Pandas resample frequency string (default ``"D"`` = daily).

    Returns
    -------
    dict[str, list]
        Mapping of column name → list of dicts ``{"period": str, "null_rate": float}``.
        Returns an empty dict if *timestamp_col* is not present or not parseable.
    """
    if timestamp_col not in df.columns:
        return {}

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    if ts.isna().all():
        return {}

    tmp = df.copy()
    tmp["_ts"] = ts
    tmp = tmp.dropna(subset=["_ts"]).set_index("_ts")

    result: dict[str, list] = {}
    for col in tmp.columns:
        if col == timestamp_col:
            continue
        grouped = tmp[col].resample(freq).apply(lambda s: float(s.isna().mean()))
        result[col] = [
            {"period": str(period.date()), "null_rate": rate}
            for period, rate in grouped.items()
            if not (rate != rate)  # skip NaN entries (empty buckets / weekend gaps)
        ]
    return result


def flag_null_severity(null_rates: dict[str, float]) -> dict[str, str]:
    """Assign a severity label to each column based on its null rate.

    Severity thresholds:
    - ``"low"``    — null rate ≤ 10 %
    - ``"medium"`` — 10 % < null rate ≤ 25 %
    - ``"high"``   — null rate > 25 %

    Parameters
    ----------
    null_rates:
        Mapping of column name → null rate in [0, 1], as returned by
        :func:`null_rate_per_column`.

    Returns
    -------
    dict[str, str]
        Mapping of column name → severity string (``"low"``, ``"medium"``, or ``"high"``).
    """
    severity: dict[str, str] = {}
    for col, rate in null_rates.items():
        if rate > 0.25:
            severity[col] = "high"
        elif rate > 0.10:
            severity[col] = "medium"
        else:
            severity[col] = "low"
    return severity
