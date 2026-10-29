"""
Free, no-signup data source using yfinance -- for testing the backtest
engine's plumbing ONLY, while your Upstox account clears KYC.

Do NOT use this for real strategy validation. Two real limitations:
  1. Yahoo Finance caps 1-minute intraday data at roughly the last 7 days
     for NSE symbols -- nowhere near enough history for a trustworthy
     backtest sample size.
  2. Data quality/completeness for NSE symbols via Yahoo is inconsistent
     compared to a direct broker feed -- occasional gaps, minor timestamp
     issues. Fine for "does my engine run without crashing", not fine for
     "is this strategy actually profitable".

Once your Upstox account is KYC-verified, switch back to data/fetcher.py
(Upstox) for anything you intend to draw real conclusions from. This
module deliberately mirrors fetcher.py's function signature (fetch,
then storage.save) so swapping back is a one-line change in your script,
not a rewrite.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf


def fetch_recent_1min(trading_symbol: str) -> pd.DataFrame:
    """
    NSE symbols need a '.NS' suffix on Yahoo Finance (e.g. 'RELIANCE.NS').
    Returns whatever 1-minute data Yahoo has -- typically the last ~7
    calendar days, NOT a date range you control.
    """
    yahoo_symbol = f"{trading_symbol.upper()}.NS"
    ticker = yf.Ticker(yahoo_symbol)
    df = ticker.history(period="7d", interval="1m")

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })[["open", "high", "low", "close", "volume"]]

    # Yahoo returns tz-aware timestamps (often UTC or US/Eastern depending
    # on version) -- convert to naive IST-local so it lines up with how
    # storage.py and the engine treat market-hours strings ("09:15" etc).
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)

    return df.sort_index()
