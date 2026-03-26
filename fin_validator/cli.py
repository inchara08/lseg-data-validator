"""
fin_validator.cli — typer-based CLI entry point.

Commands
--------
check  <file>                    Run quality checks on a CSV or Parquet file.
diff   <file_a> <file_b>         Compare two snapshots for schema drift.
report <file> --output <out>     Generate a standalone HTML report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(
    name="fin-validator",
    help="Data quality toolkit for financial market data.",
    add_completion=False,
)


def _read(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


@app.command()
def check(
    file: Path = typer.Argument(..., help="CSV or Parquet file to check."),
    ric_col: str = typer.Option("RIC", help="Name of the RIC column."),
    zscore: float = typer.Option(3.0, help="Z-score outlier threshold."),
) -> None:
    """Run data quality checks on FILE and print a summary."""
    from fin_validator import DataQualityReport

    df = _read(file)
    report = DataQualityReport(df, ric_col=ric_col)
    report.summary()


@app.command()
def diff(
    file_a: Path = typer.Argument(..., help="Older snapshot (CSV or Parquet)."),
    file_b: Path = typer.Argument(..., help="Newer snapshot (CSV or Parquet)."),
    null_threshold: float = typer.Option(
        0.05, help="Null-rate delta above which a column is flagged."
    ),
) -> None:
    """Compare two snapshots FILE_A and FILE_B for schema drift."""
    import json

    from fin_validator.checks.schema_diff import run_all

    df_a = _read(file_a)
    df_b = _read(file_b)
    result = run_all(df_a, df_b, null_flag_threshold=null_threshold)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def report(
    file: Path = typer.Argument(..., help="CSV or Parquet file."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output HTML file (default: <file>.report.html)."
    ),
    ric_col: str = typer.Option("RIC", help="Name of the RIC column."),
) -> None:
    """Generate a standalone HTML quality report for FILE."""
    from fin_validator import DataQualityReport

    df = _read(file)
    rpt = DataQualityReport(df, ric_col=ric_col)
    html = rpt.to_html()

    out_path = output or file.with_suffix(".report.html")
    out_path.write_text(html, encoding="utf-8")
    typer.echo(f"Report saved to {out_path}")


if __name__ == "__main__":
    app()
