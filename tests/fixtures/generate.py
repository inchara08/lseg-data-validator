"""
tests/fixtures/generate.py — synthetic LSEG-shaped DataFrame generator.

Mimics data returned by ``lseg.data.get_history()`` for a multi-instrument
daily OHLCV + fundamentals query, then reshaped to long format with a ``RIC``
column.  Field names, RIC codes, value ranges, and data quality patterns are
derived from real LSEG Data Library for Python API responses.

Field naming conventions
------------------------
- ``TRDPRC_1``, ``OPEN_PRC``, ``HIGH_1``, ``LOW_1``, ``ACVOL_UNS``
  Real-time / historical pricing fields (historical_pricing.summaries endpoint)
- ``BID``, ``ASK``
  Level 1 quote fields
- ``TR.*``
  Refinitiv (LSEG) fundamental / TR analytics namespace
- ``CF_*``
  Legacy Eikon Content Framework fields still returned by some endpoints

Modes
-----
- **clean**: no quality issues injected.
- **dirty**: realistic quality issues injected — natural nulls, outlier price
  spikes, type mismatches, malformed timestamps, invalid RIC codes, duplicate
  rows.

Usage (CLI)::

    python tests/fixtures/generate.py

Writes ``sample_clean.csv`` and ``sample_dirty.csv`` to the same directory.

Usage (import)::

    from tests.fixtures.generate import make_clean_df, make_dirty_df
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Instrument universe ───────────────────────────────────────────────────────
# RIC format: TICKER.EXCHANGE  (O = NASDAQ, N = NYSE, L = London Stock Exchange)
# Prices are in native currency:
#   USD for .O / .N instruments
#   GBp (pence, integer-equivalent) for .L instruments
#
# Columns: base_price, avg_daily_volume, quarterly_eps, annual_div_yield_pct, pe_ratio
_INSTRUMENTS: dict[str, tuple[float, int, float, float, float]] = {
    "AAPL.O":  (185.50,  55_000_000,  1.52, 0.51, 28.8),
    "MSFT.O":  (415.20,  22_000_000,  2.94, 0.72, 35.4),
    "NVDA.O":  (673.40,  44_000_000,  3.71, 0.03, 68.5),
    "GOOGL.O": (168.30,  26_000_000,  1.89, 0.00, 25.1),
    "AMZN.O":  (188.90,  46_000_000,  0.98, 0.00, 60.3),
    "TSLA.O":  (219.90, 108_000_000,  0.71, 0.00, 62.1),
    "LSEG.L":  (9840.0,   1_150_000, 134.2, 1.18, 73.4),
    "BP.L":    ( 492.5,  17_500_000, 52.10, 4.82,  9.5),
    "SHEL.L":  (2658.0,   8_800_000, 230.5, 3.91, 11.5),
    "AZN.L":   (10750.0,  3_800_000, 412.3, 2.08, 26.2),
}

# Rows per instrument (~5 weeks of business days)
_N_PER_RIC = 25
SEED = 42


# ── Internal helpers ──────────────────────────────────────────────────────────


def _geometric_random_walk(
    rng: np.random.Generator,
    start: float,
    steps: int,
    daily_vol: float,
) -> np.ndarray:
    """Simulate a log-normal price walk anchored at *start*."""
    log_returns = rng.normal(0.0, daily_vol, size=steps)
    prices = start * np.exp(np.cumsum(log_returns))
    prices = np.concatenate([[start], prices[:-1]])  # keep first row at start
    return prices


def _eps_series(
    rng: np.random.Generator,
    quarterly_eps: float,
    n: int,
    earnings_cycle: int = 63,  # ~13 weeks in trading days
) -> np.ndarray:
    """Return a sparse EPS array — non-null only on earnings announcement dates.

    TR.EPS is a point-in-time field populated only when an earnings
    announcement occurs; all other dates carry ``NaN``.
    """
    eps = np.full(n, np.nan)
    for offset in range(0, n, earnings_cycle):
        noise = rng.normal(0.0, quarterly_eps * 0.05)
        eps[offset] = round(quarterly_eps + noise, 4)
    return eps


def _base_df(n_per_ric: int = _N_PER_RIC, seed: int = SEED) -> pd.DataFrame:
    """Build a clean long-format LSEG-shaped DataFrame.

    Each instrument contributes ``n_per_ric`` rows of business-day OHLCV data
    plus TR.* fundamentals and legacy CF_* fields.
    """
    rng = np.random.default_rng(seed)

    # Business-day timestamps
    dates = pd.date_range("2024-01-02", periods=n_per_ric, freq="B")
    date_strs = dates.strftime("%Y-%m-%d %H:%M:%S")

    rows: list[dict] = []

    for ric, (base_price, base_vol, eps_q, div_yield, _pe) in _INSTRUMENTS.items():
        # Daily volatility: tech stocks more volatile than energy/pharma
        daily_vol = 0.018 if ric.endswith(".O") else 0.012

        close = _geometric_random_walk(rng, base_price, n_per_ric, daily_vol)

        # OPEN_PRC: within ±0.4% of prior close (gap up/down)
        open_prc = close * (1.0 + rng.uniform(-0.004, 0.004, n_per_ric))

        # HIGH/LOW: realistic intraday range derived from close
        intraday_range = rng.uniform(0.003, 0.015, n_per_ric)
        high = np.maximum(close, open_prc) * (1.0 + intraday_range * 0.6)
        low  = np.minimum(close, open_prc) * (1.0 - intraday_range * 0.4)

        # Volume: log-normal dispersion around daily average
        vol = (base_vol * rng.lognormal(0.0, 0.30, n_per_ric)).astype(np.int64)

        # Bid/Ask spread: tight for liquid large-caps, wider for LSE small-caps
        spread_pct = 0.0001 if base_vol > 10_000_000 else 0.0004
        bid = close * (1.0 - spread_pct)
        ask = close * (1.0 + spread_pct)

        # TR.EPS — sparse; only non-null on earnings dates
        eps_arr = _eps_series(rng, eps_q, n_per_ric)

        # TR.DividendYield — annual figure, small daily variation
        if div_yield > 0:
            div_arr = rng.uniform(div_yield * 0.96, div_yield * 1.04, n_per_ric)
        else:
            div_arr = np.zeros(n_per_ric)

        # TR.PE — computed from current price relative to annualised EPS
        pe_arr = close / (eps_q * 4.0)

        # CF_CLOSE / CF_VOLUME — legacy fields with minor rounding differences
        cf_close  = close  + rng.uniform(-0.02, 0.02, n_per_ric)
        cf_volume = (vol   * rng.uniform(0.99, 1.01, n_per_ric)).astype(np.int64)

        for i, date in enumerate(date_strs):
            rows.append(
                {
                    "RIC":              ric,
                    "Date":             date,
                    "TRDPRC_1":         round(float(close[i]),   4),
                    "OPEN_PRC":         round(float(open_prc[i]),4),
                    "HIGH_1":           round(float(high[i]),     4),
                    "LOW_1":            round(float(low[i]),      4),
                    "ACVOL_UNS":        int(vol[i]),
                    "BID":              round(float(bid[i]),      4),
                    "ASK":              round(float(ask[i]),      4),
                    "TR.PriceClose":    round(float(close[i]),    4),
                    "TR.Volume":        int(vol[i]),
                    "TR.EPS":           float(eps_arr[i]) if not np.isnan(eps_arr[i]) else float("nan"),
                    "TR.DividendYield": round(float(div_arr[i]),  4),
                    "TR.PE":            round(float(pe_arr[i]),   2),
                    "CF_CLOSE":         round(float(cf_close[i]), 4),
                    "CF_VOLUME":        int(cf_volume[i]),
                }
            )

    return pd.DataFrame(rows)


# ── Public API ────────────────────────────────────────────────────────────────


def make_clean_df(n_per_ric: int = _N_PER_RIC, seed: int = SEED) -> pd.DataFrame:
    """Return a clean LSEG-shaped DataFrame with no quality issues.

    Parameters
    ----------
    n_per_ric:
        Number of business-day rows per instrument.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape: ``(len(instruments) * n_per_ric, 16)``
    """
    return _base_df(n_per_ric=n_per_ric, seed=seed)


def make_dirty_df(n_per_ric: int = _N_PER_RIC, seed: int = SEED) -> pd.DataFrame:
    """Return a dirty LSEG-shaped DataFrame with injected quality issues.

    Issues:

    1. **Sparse nulls in TR.EPS** — naturally null between earnings dates
       (already present in base data; no additional injection needed).
    2. **~15 % nulls in BID / ASK** — mimics liquidity gaps and after-hours
       windows when quotes are unavailable.
    3. **2 price spike outliers in TRDPRC_1** — simulates flash-crash or
       data-error events (value ~8–12× the instrument mean).
    4. **ACVOL_UNS type mismatch** — 3 rows have volume stored as a formatted
       string (e.g. ``"12,345,678"``) rather than an integer, a common artefact
       of CSV exports from terminal applications.
    5. **2 malformed Date values** — ``"N/A"`` and an empty string, mimicking
       timestamp parse failures in the delivery pipeline.
    6. **2 invalid RIC codes** — missing exchange suffix (e.g. ``"AAPL"``),
       which the RIC validator flags as malformed.
    7. **1 duplicate row** — row 0 appended again, simulating a double-delivery
       from a real-time feed.

    Parameters
    ----------
    n_per_ric:
        Number of business-day rows per instrument.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed + 1)  # different seed to get independent choices
    df = _base_df(n_per_ric=n_per_ric, seed=seed)
    n = len(df)

    # 1. ~15 % nulls in BID and ASK (liquidity / after-hours gaps)
    bid_null_idx = rng.choice(n, size=max(1, int(n * 0.15)), replace=False)
    ask_null_idx = rng.choice(n, size=max(1, int(n * 0.15)), replace=False)
    df.loc[bid_null_idx, "BID"] = float("nan")
    df.loc[ask_null_idx, "ASK"] = float("nan")

    # 2. 2 price spike outliers in TRDPRC_1 (flash crash / bad tick)
    spike_idx = rng.choice(n, size=2, replace=False)
    for idx in spike_idx:
        multiplier = rng.uniform(8.0, 12.0)
        df.loc[idx, "TRDPRC_1"] = round(df.loc[idx, "TRDPRC_1"] * multiplier, 4)

    # 3. ACVOL_UNS type mismatch — store 3 rows as comma-formatted strings
    str_idx = rng.choice(n, size=3, replace=False)
    df["ACVOL_UNS"] = df["ACVOL_UNS"].astype(object)
    for idx in str_idx:
        raw_val = int(df.loc[idx, "ACVOL_UNS"])
        df.loc[idx, "ACVOL_UNS"] = f"{raw_val:,}"   # e.g. "12,345,678"

    # 4. 2 malformed Date values (timestamp parse failures)
    date_bad_idx = rng.choice(n, size=2, replace=False)
    df.loc[date_bad_idx[0], "Date"] = "N/A"
    df.loc[date_bad_idx[1], "Date"] = ""

    # 5. 2 invalid RIC codes (missing exchange suffix)
    ric_bad_idx = rng.choice(n, size=2, replace=False)
    df.loc[ric_bad_idx[0], "RIC"] = "AAPL"          # no .O suffix
    df.loc[ric_bad_idx[1], "RIC"] = "MSFT-INVALID"  # invalid separator

    # 6. 1 duplicate row (double-delivery from feed)
    duplicate = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate], ignore_index=True)

    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent

    clean = make_clean_df()
    dirty = make_dirty_df()

    clean_path = fixtures_dir / "sample_clean.csv"
    dirty_path  = fixtures_dir / "sample_dirty.csv"

    clean.to_csv(clean_path, index=False)
    dirty.to_csv(dirty_path,  index=False)

    print(f"Written {len(clean):,} rows × {len(clean.columns)} columns → {clean_path}")
    print(f"Written {len(dirty):,} rows × {len(dirty.columns)} columns → {dirty_path}")
    print()
    print("Clean columns:", list(clean.columns))
    print()
    print("Dirty injections:")
    print(f"  BID nulls:      {clean['BID'].isna().sum()} → {dirty['BID'].isna().sum()}")
    print(f"  TRDPRC_1 spikes: 2 rows inflated 8–12×")
    print(f"  ACVOL_UNS str:   3 rows as comma-formatted strings")
    print(f"  Bad dates:       2 rows ('N/A', '')")
    print(f"  Invalid RICs:    2 rows ('AAPL', 'MSFT-INVALID')")
    print(f"  Duplicate rows:  1 (row 0 repeated at end)")
