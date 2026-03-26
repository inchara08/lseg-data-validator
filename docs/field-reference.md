# Field Reference

Quick reference for field naming conventions used in financial market data,
particularly as returned by the LSEG Data Library for Python.

## Common TR.* Fields

TR.* fields are part of the LSEG Data Library (Refinitiv) analytics namespace.

| Field Name       | Type    | Description                              |
|------------------|---------|------------------------------------------|
| TR.PriceClose    | float   | Closing price for the period             |
| TR.PriceOpen     | float   | Opening price for the period             |
| TR.PriceHigh     | float   | High price for the period                |
| TR.PriceLow      | float   | Low price for the period                 |
| TR.Volume        | float   | Total traded volume                      |
| TR.TotalReturn   | float   | Total return including dividends (%)     |
| TR.VWAP          | float   | Volume-weighted average price            |
| TR.MarketCap     | float   | Market capitalisation                    |

## RIC Code Format

RIC (Reuters Instrument Code) format: `<ticker>.<exchange_suffix>`

Examples:
- `MSFT.O`  — Microsoft on NASDAQ
- `LSEG.L`  — LSEG plc on London Stock Exchange
- `AAPL.O`  — Apple on NASDAQ
- `TSLA.O`  — Tesla on NASDAQ
- `BP.L`    — BP on London Stock Exchange

Exchange suffixes:
- `.O`  — NASDAQ
- `.N`  — NYSE
- `.L`  — London Stock Exchange
- `.PA` — Euronext Paris
- `.T`  — Tokyo Stock Exchange

## Timestamp Format

LSEG timestamps are typically ISO 8601 in UTC:
`2024-01-15T09:30:00Z`

or date-only for daily data:
`2024-01-15`
