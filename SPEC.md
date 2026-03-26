# fin-data-validator — Project Spec

## Purpose
A free, open-source data quality toolkit for financial market data. It takes a
pandas DataFrame and produces a rich quality report: null rates, type inference,
anomaly detection, and schema drift between two time periods.

Designed and tested against LSEG Data Library for Python (`lseg-data`) outputs,
but works with any tabular financial data.

Target audience: quant developers, financial data engineers, and data scientists
who work with market data — especially those using LSEG Workspace, RDP, or Eikon —
and want fast confidence checks on the data they pull.

---

## Core Features (MVP — 1-2 weeks)

### 1. CLI tool (`fin-validator`)
- `fin-validator check <file.csv>` — run quality checks on a CSV/parquet snapshot
- `fin-validator diff <file_a.csv> <file_b.csv>` — compare two snapshots for schema drift
- `fin-validator report <file.csv> --output report.html` — generate standalone HTML report
- Outputs: coloured terminal summary + optional HTML report

### 2. Python API (importable)
```python
from fin_validator import DataQualityReport
report = DataQualityReport(df)
report.summary()       # prints to terminal
report.to_html()       # returns HTML string
report.to_dict()       # returns structured dict for programmatic use
```

### 3. Streamlit Web UI (`streamlit run app.py`)
- Drag-and-drop CSV/parquet upload
- Auto-detect field naming conventions (TR.* fields, RIC codes, timestamps)
- Visual null heatmap (seaborn or plotly)
- Field distribution charts (histogram per numeric column)
- Anomaly flags table (Z-score > 3 or IQR outliers, per column)
- Schema diff view (side-by-side when two files uploaded)
- One-click HTML report download

---

## Quality Check Modules

### Completeness
- Null rate per column (% missing)
- Null rate over time (if timestamp column detected) — catches data feed outages
- Flag columns with >10%, >25%, >50% nulls with severity levels

### Consistency
- Type inference — detect columns that are numeric but stored as strings
- Date/timestamp parsing — detect malformed timestamps
- RIC code format validation (basic regex: alphanumeric + `.` + exchange suffix)
- Duplicate row detection

### Anomaly Detection
- Z-score outlier flagging (threshold configurable, default: |z| > 3)
- IQR outlier flagging (default: 1.5x IQR)
- Sudden value spikes (% change > configurable threshold between consecutive rows)

### Schema Drift (diff mode)
- New columns added
- Columns removed
- Data type changes per column
- Null rate delta per column (> 5% change flagged)
- Value range shifts (mean/std delta)

---

## Tech Stack
- Python 3.10+
- pandas, numpy — core data handling
- scipy — Z-score, IQR stats
- plotly — interactive charts in Streamlit
- streamlit — web UI
- typer — CLI framework
- jinja2 — HTML report templating
- pytest — tests
- black + ruff — formatting/linting

No credentials required — works on any pandas DataFrame or CSV file.
Includes a synthetic data generator for CI: generates realistic financial
DataFrames with known quality issues injected (nulls, outliers, type mismatches).

---

## Project Structure
```
fin-data-validator/
├── SPEC.md
├── README.md
├── pyproject.toml
├── fin_validator/
│   ├── __init__.py
│   ├── checks/
│   │   ├── completeness.py
│   │   ├── consistency.py
│   │   ├── anomaly.py
│   │   └── schema_diff.py
│   ├── report.py              ← DataQualityReport class
│   ├── cli.py                 ← typer CLI
│   └── templates/
│       └── report.html.j2     ← Jinja2 HTML report template
├── app/
│   └── streamlit_app.py       ← Streamlit UI
├── tests/
│   ├── fixtures/              ← synthetic financial CSVs
│   ├── test_completeness.py
│   ├── test_anomaly.py
│   └── test_schema_diff.py
└── docs/
    └── field-reference.md     ← TR.* field naming conventions doc
```

---

## Phased Build Plan

### Phase 1 — Core engine (Days 1–3)
- Scaffold project with pyproject.toml, ruff, black, pytest
- Implement completeness.py, consistency.py, anomaly.py
- Write tests against synthetic fixtures
- DataQualityReport class with .summary() and .to_dict()

### Phase 2 — CLI + HTML report (Days 4–6)
- typer CLI: check, diff, report commands
- Jinja2 HTML report template (clean, professional design)
- schema_diff.py module
- Full test coverage

### Phase 3 — Streamlit UI (Days 7–9)
- File upload + auto field detection
- Plotly null heatmap + distribution charts
- Anomaly flags table
- Schema diff side-by-side view
- Download report button

### Phase 4 — Polish + Launch (Days 10–14)
- README with clear framing, badges, screenshots/GIF
- Deploy Streamlit app to Streamlit Cloud (free)
