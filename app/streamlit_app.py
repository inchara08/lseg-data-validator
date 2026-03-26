"""
app/streamlit_app.py — Streamlit web UI for fin-data-validator.

This module is intentionally thin: all business logic lives in fin_validator/.
This file only handles file upload, rendering, and layout.

Run with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import re as _re

import pandas as pd
import plotly.express as px
import streamlit as st

from fin_validator import DataQualityReport
from fin_validator.checks.schema_diff import run_all as schema_diff

st.set_page_config(page_title="Financial Data Validator", layout="wide")
st.title("Financial Data Validator")
st.caption("Drag-and-drop your CSV snapshot to run quality checks instantly. Tested with LSEG Data Library outputs.")


def _load(uploaded) -> pd.DataFrame:
    """Parse an uploaded file into a DataFrame."""
    if uploaded.name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    return pd.read_csv(uploaded)


# ── File upload ──────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    file_a = st.file_uploader("Primary snapshot (required)", type=["csv", "parquet"])
with col_b:
    file_b = st.file_uploader(
        "Comparison snapshot (optional, for diff)", type=["csv", "parquet"]
    )

if file_a is None:
    st.info("Upload a CSV or Parquet file to begin.")
    st.stop()

df = _load(file_a)
df_b = _load(file_b) if file_b is not None else None

report = DataQualityReport(df)
results = report.to_dict()

st.subheader(f"Dataset: `{file_a.name}` — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Field detection banner ────────────────────────────────────────────────────
_ts_pattern = _re.compile(r"time|date|timestamp", _re.IGNORECASE)
_ric_detected = "RIC" if "RIC" in df.columns else next(
    (c for c in df.columns if c.upper() == "RIC"), None
)
_ts_cols = list(results["consistency"]["malformed_timestamp_columns"].keys()) or [
    c for c in df.columns if _ts_pattern.search(c)
]
_tr_cols = [c for c in df.columns if c.startswith("TR.")]

b1, b2, b3 = st.columns(3)
with b1:
    if _ric_detected:
        st.success(f"RIC column: `{_ric_detected}`")
    else:
        st.warning("No RIC column detected")
with b2:
    if _ts_cols:
        st.info(f"Timestamp columns: {', '.join(f'`{c}`' for c in _ts_cols)}")
    else:
        st.info("No timestamp columns detected")
with b3:
    if _tr_cols:
        _tr_preview = ", ".join(f"`{c}`" for c in _tr_cols[:5])
        _tr_suffix = "…" if len(_tr_cols) > 5 else ""
        st.info(f"{len(_tr_cols)} TR. field(s): {_tr_preview}{_tr_suffix}")
    else:
        st.caption("No TR. prefixed columns")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Completeness", "Consistency", "Anomalies", "Schema Diff", "Distributions"]
)

# ── Completeness ──────────────────────────────────────────────────────────────
with tab1:
    null_rates = results["completeness"]["null_rate_per_column"]
    severity = results["completeness"]["null_severity"]
    null_df = pd.DataFrame(
        {
            "Column": list(null_rates.keys()),
            "Null Rate (%)": [v * 100 for v in null_rates.values()],
            "Severity": [severity.get(c, "low") for c in null_rates],
        }
    )
    st.dataframe(null_df, use_container_width=True)

    if null_rates:
        fig = px.bar(
            null_df,
            x="Column",
            y="Null Rate (%)",
            color="Severity",
            color_discrete_map={"low": "green", "medium": "orange", "high": "red"},
            title="Null Rate per Column",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Null Heatmap")
    _hdf = df if len(df) <= 500 else df.sample(500, random_state=42)
    if len(df) > 500:
        st.caption(f"Heatmap sampled to 500 rows (dataset has {len(df):,} rows).")
    _hmap_cols = (
        [c for c, r in null_rates.items() if r > 0]
        if len(df.columns) > 50
        else list(df.columns)
    )
    if len(df.columns) > 50 and _hmap_cols:
        st.caption(
            f"Heatmap shows {len(_hmap_cols)} columns with at least one null "
            f"(of {len(df.columns)} total)."
        )
    if not _hdf.empty and _hmap_cols:
        _null_mask = _hdf[_hmap_cols].isnull().astype(int)
        _heat_fig = px.imshow(
            _null_mask,
            color_continuous_scale=[[0, "#d4edda"], [1, "#f5c6cb"]],
            title="Null presence per cell (green=present, red=null)",
            aspect="auto",
        )
        _heat_fig.update_coloraxes(showscale=False)
        st.plotly_chart(_heat_fig, use_container_width=True)
    else:
        st.success("No null values found — heatmap not needed.")

# ── Consistency ───────────────────────────────────────────────────────────────
with tab2:
    c = results["consistency"]
    st.metric("Numeric-string columns", len(c["numeric_string_columns"]))
    st.metric("Invalid RIC codes", c["invalid_ric_count"])
    st.metric("Duplicate rows", c["duplicate_row_count"])
    if c["numeric_string_columns"]:
        st.warning(
            f"Columns stored as strings but appear numeric: {c['numeric_string_columns']}"
        )

# ── Anomalies ─────────────────────────────────────────────────────────────────
with tab3:
    a = results["anomaly"]
    rows = []
    for col, idxs in a["zscore_outliers"].items():
        rows.append({"Column": col, "Method": "Z-score", "Outlier count": len(idxs)})
    for col, idxs in a["iqr_outliers"].items():
        rows.append({"Column": col, "Method": "IQR", "Outlier count": len(idxs)})
    for col, idxs in a["spike_rows"].items():
        rows.append({"Column": col, "Method": "Spike", "Outlier count": len(idxs)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.success("No anomalies detected.")

# ── Schema Diff ───────────────────────────────────────────────────────────────
with tab4:
    if df_b is None:
        st.info("Upload a second snapshot to see schema diff.")
    else:
        diff_result = schema_diff(df, df_b)

        st.subheader("Column Changes")
        _add_col, _rem_col = st.columns(2)
        with _add_col:
            st.markdown("**Added columns** (in new, not in old)")
            if diff_result["added_columns"]:
                for _c in diff_result["added_columns"]:
                    st.success(_c)
            else:
                st.caption("None")
        with _rem_col:
            st.markdown("**Removed columns** (in old, not in new)")
            if diff_result["removed_columns"]:
                for _c in diff_result["removed_columns"]:
                    st.error(_c)
            else:
                st.caption("None")

        st.subheader("Data Type Changes")
        if diff_result["dtype_changes"]:
            _dtype_rows = [
                {"Column": col, "From Type": v["from"], "To Type": v["to"]}
                for col, v in diff_result["dtype_changes"].items()
            ]
            st.dataframe(
                pd.DataFrame(_dtype_rows), use_container_width=True, hide_index=True
            )
        else:
            st.success("No dtype changes detected.")

        st.subheader("Null Rate Delta")
        if diff_result["null_rate_delta"]:
            _nrd_rows = [
                {
                    "Column": col,
                    "Null Rate A": f"{v['null_rate_a']:.1%}",
                    "Null Rate B": f"{v['null_rate_b']:.1%}",
                    "Delta": f"{v['delta']:+.1%}",
                    "Flagged": v["flagged"],
                }
                for col, v in diff_result["null_rate_delta"].items()
            ]
            _nrd_df = pd.DataFrame(_nrd_rows)

            def _highlight_flagged(row):
                return [
                    "background-color: #fff3cd" if row["Flagged"] else "" for _ in row
                ]

            st.dataframe(
                _nrd_df.style.apply(_highlight_flagged, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No shared columns to compare.")

        st.subheader("Value Range Shifts (Numeric Columns)")
        if diff_result["value_range_shifts"]:
            _vrs_rows = [
                {
                    "Column": col,
                    "Mean A": f"{v['mean_a']:.4g}",
                    "Mean B": f"{v['mean_b']:.4g}",
                    "Mean Delta": f"{v['mean_delta']:+.4g}",
                    "Std A": f"{v['std_a']:.4g}",
                    "Std B": f"{v['std_b']:.4g}",
                    "Std Delta": f"{v['std_delta']:+.4g}",
                }
                for col, v in diff_result["value_range_shifts"].items()
            ]
            st.dataframe(
                pd.DataFrame(_vrs_rows), use_container_width=True, hide_index=True
            )
        else:
            st.caption("No shared numeric columns to compare.")

# ── Distributions ─────────────────────────────────────────────────────────────
with tab5:
    _num_cols = df.select_dtypes(include="number").columns.tolist()
    if not _num_cols:
        st.info("No numeric columns found in this dataset.")
    else:
        _sel = st.selectbox(
            "Select a column to inspect",
            options=_num_cols,
            index=0,
            key="dist_col_selector",
        )
        _dist_fig = px.histogram(
            df,
            x=_sel,
            nbins=50,
            title=f"Distribution of `{_sel}`",
            color_discrete_sequence=["#636EFA"],
        )
        st.plotly_chart(_dist_fig, use_container_width=True)

        _col_series = df[_sel].dropna()
        if not _col_series.empty:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Mean", f"{_col_series.mean():.4g}")
            s2.metric("Std Dev", f"{_col_series.std():.4g}")
            s3.metric("Min", f"{_col_series.min():.4g}")
            s4.metric("Max", f"{_col_series.max():.4g}")

# ── Download HTML report ─────────────────────────────────────────────────────
st.divider()
html = report.to_html()
st.download_button(
    label="Download HTML Report",
    data=html.encode(),
    file_name="financial_quality_report.html",
    mime="text/html",
)
