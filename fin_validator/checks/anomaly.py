"""
fin_validator.checks.anomaly — anomaly detection for financial DataFrames.

Detects:
- Z-score outliers per numeric column (default threshold: |z| > 3).
- IQR outliers per numeric column (default multiplier: 1.5).
- Sudden value spikes between consecutive rows (configurable % change threshold).

All public functions accept a pandas DataFrame and return typed dicts so they
can be used standalone or composed inside DataQualityReport.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def zscore_outliers(df: pd.DataFrame, threshold: float = 3.0) -> dict[str, list[int]]:
    """Return row indices of Z-score outliers per numeric column.

    Parameters
    ----------
    df:
        Input DataFrame.
    threshold:
        Absolute Z-score above which a value is considered an outlier.

    Returns
    -------
    dict[str, list[int]]
        Mapping of column name → list of integer row positions that are outliers.
    """
    result: dict[str, list[int]] = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        z = np.abs(stats.zscore(series))
        outlier_positions = series.index[z > threshold].tolist()
        if outlier_positions:
            result[col] = [int(i) for i in outlier_positions]
    return result


def iqr_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> dict[str, list[int]]:
    """Return row indices of IQR outliers per numeric column.

    Parameters
    ----------
    df:
        Input DataFrame.
    multiplier:
        Fence multiplier applied to the IQR (default 1.5 = Tukey fences).

    Returns
    -------
    dict[str, list[int]]
        Mapping of column name → list of integer row positions that are outliers.
    """
    result: dict[str, list[int]] = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = (series < lower) | (series > upper)
        outlier_positions = series.index[mask].tolist()
        if outlier_positions:
            result[col] = [int(i) for i in outlier_positions]
    return result


def spike_rows(
    df: pd.DataFrame, pct_change_threshold: float = 0.5
) -> dict[str, list[int]]:
    """Return row indices where a column value spikes vs. the previous row.

    A spike is defined as ``|pct_change| > pct_change_threshold``.

    Parameters
    ----------
    df:
        Input DataFrame (rows should be in time order for meaningful results).
    pct_change_threshold:
        Fractional change that triggers a spike flag (default 0.5 = 50 %).

    Returns
    -------
    dict[str, list[int]]
        Mapping of column name → list of integer row positions with spikes.
    """
    result: dict[str, list[int]] = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        pct = series.pct_change().abs()
        mask = pct > pct_change_threshold
        spike_positions = series.index[mask].tolist()
        if spike_positions:
            result[col] = [int(i) for i in spike_positions]
    return result


def run_all(
    df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    spike_threshold: float = 0.5,
) -> dict:
    """Run all anomaly checks and return a combined result dict.

    Parameters
    ----------
    df:
        Input DataFrame.
    zscore_threshold:
        Z-score threshold for outlier detection.
    iqr_multiplier:
        IQR fence multiplier for outlier detection.
    spike_threshold:
        Fractional change threshold for spike detection.

    Returns
    -------
    dict
        Keys: ``zscore_outliers``, ``iqr_outliers``, ``spike_rows``.
    """
    return {
        "zscore_outliers": zscore_outliers(df, threshold=zscore_threshold),
        "iqr_outliers": iqr_outliers(df, multiplier=iqr_multiplier),
        "spike_rows": spike_rows(df, pct_change_threshold=spike_threshold),
    }
