"""
fin_validator.report — DataQualityReport: the main public API class.

Composes all check modules (completeness, consistency, anomaly) into a
single report object with human-readable summary and structured dict output.

Example::

    from fin_validator import DataQualityReport
    report = DataQualityReport(df)
    report.summary()       # prints coloured terminal summary
    report.to_dict()       # returns structured dict for programmatic use
    report.to_html()       # returns HTML string
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from fin_validator.checks import anomaly, completeness, consistency


class DataQualityReport:
    """Runs all quality checks on a DataFrame and exposes results.

    Parameters
    ----------
    df:
        DataFrame to analyse.
    timestamp_col:
        Name of the timestamp column.  Auto-detected if *None*.
    ric_col:
        Name of the RIC column.  Defaults to ``"RIC"``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str | None = None,
        ric_col: str = "RIC",
    ) -> None:
        self._df = df.copy()
        self._timestamp_col = timestamp_col or self._detect_timestamp_col()
        self._ric_col = ric_col
        self._results: dict | None = None

    def _detect_timestamp_col(self) -> str | None:
        """Return the first column whose name suggests a timestamp."""
        import re

        pattern = re.compile(r"time|date|timestamp", re.IGNORECASE)
        for col in self._df.columns:
            if pattern.search(col):
                return col
        return None

    def _run(self) -> dict:
        """Execute all checks and cache results."""
        if self._results is not None:
            return self._results

        null_rates = completeness.null_rate_per_column(self._df)
        self._results = {
            "completeness": {
                "null_rate_per_column": null_rates,
                "null_rate_over_time": (
                    completeness.null_rate_over_time(self._df, self._timestamp_col)
                    if self._timestamp_col
                    else {}
                ),
                "null_severity": completeness.flag_null_severity(null_rates),
            },
            "consistency": consistency.run_all(self._df, ric_col=self._ric_col),
            "anomaly": anomaly.run_all(self._df),
        }
        return self._results

    def to_dict(self) -> dict:
        """Return the full quality report as a structured dict.

        Returns
        -------
        dict
            Nested dict with keys ``completeness``, ``consistency``, ``anomaly``.
        """
        return self._run()

    def summary(self) -> None:
        """Print a human-readable quality summary to stdout."""
        r = self._run()

        print("\n=== Financial Data Quality Report ===\n")

        print("── Completeness ──")
        for col, rate in r["completeness"]["null_rate_per_column"].items():
            severity = r["completeness"]["null_severity"].get(col, "low")
            print(f"  {col}: {rate:.1%} null  [{severity}]")

        print("\n── Consistency ──")
        c = r["consistency"]
        print(f"  Numeric-string columns: {c['numeric_string_columns']}")
        print(f"  Malformed timestamps:   {c['malformed_timestamp_columns']}")
        print(f"  Invalid RIC codes:      {c['invalid_ric_count']}")
        print(f"  Duplicate rows:         {c['duplicate_row_count']}")

        print("\n── Anomalies ──")
        a = r["anomaly"]
        for col, rows in a["zscore_outliers"].items():
            print(f"  Z-score outliers in '{col}': {len(rows)} row(s)")
        for col, rows in a["iqr_outliers"].items():
            print(f"  IQR outliers in '{col}':     {len(rows)} row(s)")

        print()

    def to_html(self) -> str:
        """Render the report as an HTML string using the Jinja2 template.

        Returns
        -------
        str
            Rendered HTML.
        """
        templates_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template("report.html.j2")
        return template.render(report=self._run(), df_shape=self._df.shape)

    def to_json(self, indent: int = 2) -> str:
        """Return the report as a JSON string.

        Parameters
        ----------
        indent:
            JSON indentation level.

        Returns
        -------
        str
        """
        return json.dumps(self._run(), indent=indent, default=str)
