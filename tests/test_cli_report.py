"""
tests/test_cli_report.py — tests for the CLI (typer) and DataQualityReport.to_html().

Covers:
- CLI `check` command (via typer CliRunner)
- CLI `diff` command
- CLI `report` command (writes HTML file)
- DataQualityReport.to_html() returns valid HTML
- DataQualityReport.to_json() round-trips cleanly
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent))

from fin_validator import DataQualityReport
from fin_validator.cli import app
from tests.fixtures.generate import make_clean_df, make_dirty_df

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


# ── DataQualityReport ─────────────────────────────────────────────────────────


class TestDataQualityReport:
    def test_to_dict_returns_expected_keys(self):
        df = make_clean_df()
        r = DataQualityReport(df)
        d = r.to_dict()
        assert set(d.keys()) == {"completeness", "consistency", "anomaly"}

    def test_to_html_returns_string(self):
        df = make_clean_df()
        html = DataQualityReport(df).to_html()
        assert isinstance(html, str)

    def test_to_html_contains_doctype(self):
        html = DataQualityReport(make_clean_df()).to_html()
        assert "<!DOCTYPE html>" in html

    def test_to_html_contains_completeness_section(self):
        html = DataQualityReport(make_clean_df()).to_html()
        assert "Completeness" in html

    def test_to_html_contains_anomaly_section(self):
        html = DataQualityReport(make_clean_df()).to_html()
        assert "Anomal" in html

    def test_to_html_shows_row_count(self):
        df = make_clean_df()
        html = DataQualityReport(df).to_html()
        assert str(len(df)) in html

    def test_to_json_is_valid_json(self):
        d = json.loads(DataQualityReport(make_clean_df()).to_json())
        assert "completeness" in d

    def test_to_json_round_trips(self):
        df = make_clean_df()
        r = DataQualityReport(df)
        d1 = r.to_dict()
        d2 = json.loads(r.to_json())
        assert set(d1.keys()) == set(d2.keys())

    def test_summary_prints_without_error(self, capsys):
        DataQualityReport(make_clean_df()).summary()
        captured = capsys.readouterr()
        assert "Completeness" in captured.out

    def test_dirty_df_summary_mentions_duplicate_rows(self, capsys):
        """Dirty fixture has 1 injected duplicate row — summary should report it."""
        DataQualityReport(make_dirty_df()).summary()
        captured = capsys.readouterr()
        assert "1" in captured.out  # duplicate_row_count = 1


# ── CLI check command ─────────────────────────────────────────────────────────


class TestCliCheck:
    def test_check_exits_zero(self):
        result = runner.invoke(app, ["check", str(FIXTURES / "sample_clean.csv")])
        assert result.exit_code == 0, result.output

    def test_check_prints_completeness(self):
        result = runner.invoke(app, ["check", str(FIXTURES / "sample_clean.csv")])
        assert "Completeness" in result.output

    def test_check_dirty_file(self):
        result = runner.invoke(app, ["check", str(FIXTURES / "sample_dirty.csv")])
        assert result.exit_code == 0
        assert "Anomal" in result.output or "outlier" in result.output.lower()

    def test_check_missing_file_nonzero_exit(self):
        result = runner.invoke(app, ["check", "nonexistent_file.csv"])
        assert result.exit_code != 0


# ── CLI diff command ──────────────────────────────────────────────────────────


class TestCliDiff:
    def test_diff_exits_zero(self):
        result = runner.invoke(
            app,
            [
                "diff",
                str(FIXTURES / "sample_clean.csv"),
                str(FIXTURES / "sample_dirty.csv"),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_diff_outputs_json(self):
        result = runner.invoke(
            app,
            [
                "diff",
                str(FIXTURES / "sample_clean.csv"),
                str(FIXTURES / "sample_dirty.csv"),
            ],
        )
        parsed = json.loads(result.output)
        assert "added_columns" in parsed
        assert "removed_columns" in parsed

    def test_diff_null_threshold_option(self):
        result = runner.invoke(
            app,
            [
                "diff",
                str(FIXTURES / "sample_clean.csv"),
                str(FIXTURES / "sample_dirty.csv"),
                "--null-threshold",
                "0.01",
            ],
        )
        assert result.exit_code == 0


# ── CLI report command ────────────────────────────────────────────────────────


class TestCliReport:
    def test_report_creates_html_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.html"
            result = runner.invoke(
                app,
                ["report", str(FIXTURES / "sample_clean.csv"), "--output", str(out_path)],
            )
            assert result.exit_code == 0, result.output
            assert out_path.exists()

    def test_report_html_file_not_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.html"
            runner.invoke(
                app,
                ["report", str(FIXTURES / "sample_clean.csv"), "--output", str(out_path)],
            )
            content = out_path.read_text(encoding="utf-8")
            assert len(content) > 100

    def test_report_html_contains_doctype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.html"
            runner.invoke(
                app,
                ["report", str(FIXTURES / "sample_clean.csv"), "--output", str(out_path)],
            )
            assert "<!DOCTYPE html>" in out_path.read_text(encoding="utf-8")

    def test_report_default_output_path(self):
        """Without --output, the report should be written next to the input file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the fixture CSV into a temp dir so we don't pollute fixtures/
            import shutil

            src = FIXTURES / "sample_clean.csv"
            dst = Path(tmpdir) / "sample_clean.csv"
            shutil.copy(src, dst)
            result = runner.invoke(app, ["report", str(dst)])
            assert result.exit_code == 0
            expected_out = dst.with_suffix(".report.html")
            assert expected_out.exists()

    def test_report_prints_saved_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.html"
            result = runner.invoke(
                app,
                ["report", str(FIXTURES / "sample_clean.csv"), "--output", str(out_path)],
            )
            assert "Report saved" in result.output
