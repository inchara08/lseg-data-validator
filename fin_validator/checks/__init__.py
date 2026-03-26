"""
fin_validator.checks — individual quality check modules.

Each module is self-contained: it accepts a pandas DataFrame and returns
a typed dict.  Modules never import from each other; report.py composes them.

Modules
-------
completeness  — null rates and severity flags
consistency   — type inference, RIC validation, duplicate detection
anomaly       — Z-score / IQR outlier detection, spike detection
schema_diff   — column-level diff between two DataFrames
"""
