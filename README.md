# fin-data-validator

> Is your financial data actually clean?

A free, open-source data quality toolkit for financial market data. Point it at any CSV, Parquet file, or pandas DataFrame and get an instant quality report — null rates, type mismatches, anomalies, and schema drift.

Works great out of the box with **LSEG Data Library** outputs — TR.* fields, RIC codes, and OHLCV time series are all natively understood.

---

## Features

- **Completeness** — null rate per column, null rate over time, severity flags (low / medium / high)
- **Consistency** — numeric-string type mismatches, malformed timestamps, invalid RIC codes, duplicate rows
- **Anomaly detection** — Z-score outliers, IQR outliers, sudden value spikes
- **Schema drift** — column additions/removals, dtype changes, null rate deltas, value range shifts between two snapshots
- **Streamlit web UI** — drag-and-drop upload, null heatmap, distribution charts, schema diff side-by-side view, one-click HTML report download
- **CLI** — `fin-validator check / diff / report` commands
- **Python API** — importable `DataQualityReport` class

---

## Quick start

```bash
pip install -e .

# CLI
fin-validator check data.csv
fin-validator diff snapshot_jan.csv snapshot_feb.csv
fin-validator report data.csv --output report.html

# Web UI
streamlit run app/streamlit_app.py
```

### Python API

```python
import pandas as pd
from fin_validator import DataQualityReport

df = pd.read_csv("data.csv")
report = DataQualityReport(df)

report.summary()       # coloured terminal output
report.to_dict()       # structured dict for programmatic use
report.to_html()       # standalone HTML string
report.to_json()       # JSON string
```

---

## Sample output

```
Dataset: 250 rows × 16 columns
RIC column: AAPL.O   Timestamp: Date

Completeness
  BID            15.2%  ▶ medium
  TR.EPS         96.0%  ▶ high   (sparse — earnings dates only)

Consistency
  Numeric-string columns : 1  (ACVOL_UNS)
  Invalid RIC codes      : 2
  Duplicate rows         : 1

Anomalies
  TRDPRC_1   Z-score   2 outliers
  TRDPRC_1   Spike     2 outliers
```

---

## Works with

Any DataFrame containing financial market data, including:

- `lseg.data.get_history()` — daily OHLCV time series
- `lseg.data.get_data()` — cross-sectional fundamentals
- Eikon Data API responses
- CSV / Parquet exports from LSEG Workspace, CodeBook, or RDP
- Any tabular financial data with a RIC or ticker column

Field naming conventions natively supported: `TR.*`, `CF_*`, `TRDPRC_1`, `OPEN_PRC`, `HIGH_1`, `LOW_1`, `ACVOL_UNS`, `BID`, `ASK`.

---

## Project structure

```
fin-data-validator/
├── fin_validator/
│   ├── checks/
│   │   ├── completeness.py
│   │   ├── consistency.py
│   │   ├── anomaly.py
│   │   └── schema_diff.py
│   ├── report.py          # DataQualityReport class
│   ├── cli.py             # Typer CLI
│   └── templates/
│       └── report.html.j2
├── app/
│   └── streamlit_app.py
├── tests/
│   └── fixtures/          # synthetic financial CSVs
└── docs/
    └── field-reference.md
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v          # 98 tests
black . && ruff check .
```

---

## License

MIT
