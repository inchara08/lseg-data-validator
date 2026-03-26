"""
fin_validator — data quality toolkit for financial market data.

Public API:
    DataQualityReport  — main entry point; composes all check modules.

Example::

    from fin_validator import DataQualityReport
    report = DataQualityReport(df)
    report.summary()
    report.to_dict()
"""

from fin_validator.report import DataQualityReport

__all__ = ["DataQualityReport"]
