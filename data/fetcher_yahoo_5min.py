"""
RESEARCH-ONLY supplementary data source: Yahoo Finance's unauthenticated
chart endpoint, hit directly via requests (NOT the yfinance library,
which caps NSE 1-minute data at ~7 days -- see data/fetcher_yfinance.py).
This endpoint gives ~60 calendar days of 5-MINUTE NSE intraday data, no
API key or session cookies needed -- roughly double the trading-day depth
of Upstox's own ~30-day 1-minute window, at 5-minute instead of 1-minute
granularity.

Empirically verified before this module was written (2026-07-13), not
assumed from the endpoint's documentation:
  - RELIANCE.NS at interval=5m&range=60d returned 4501 bars spanning
    EXACTLY 60 distinct trading days (2026-04-21 to 2026-07-13).
  - 225 null-OHLCV rows found -- fully and precisely explained: exactly 3
    dates (2026-05-01, 2026-05-28, 2026-06-26) x 75 bars each, which are
    the 3 real NSE holidays in that window per config/nse_holidays.json.
    Not missing/bad data -- placeholder timestamp slots for days the
    market didn't trade at all. Filtered out below via dropna().
  - Data quality vs Upstox: comparing OPEN prices at matching bar-start
    labels (the correct alignment -- both sources timestamp a bar by its
    OWN start, confirmed by this exact test) showed mean diff 0.008%,
    only 5/1425 overlapping bars exceeding a 0.1% threshold. A naive
    close-to-close comparison at the same labels looked worse (0.07%
    mean, 22.5% of bars over threshold) purely because Yahoo's 5-min
    bar's close is 4 minutes later in time than a same-labeled 1-min
    Upstox bar's close, not because of any real data disagreement --
    confirmed by shifting the comparison to the correct end-of-window
    bar, which brought it back down to 0.008% mean.

DELIBERATELY NOT wired into storage.py, BacktestEngine, or any live
path -- this is 5-minute granularity, a coarser timeframe than every
existing strategy's entry logic assumes (all built on 1-minute bars,
some layering 5-min/15-min derived views on TOP of that 1-min base via
data/resample.py). Mixing this in as if it were equivalent 1-min history
would silently change what every existing detector actually sees. Use
this only for research that's explicitly fine with 5-minute resolution
(e.g. extending backtest/walk_forward.py's date range for a longer
lookback), never as a drop-in replacement for storage.load()/
load_combined() in the live/backtest path.
"""
from __future__ import annotations

import pandas as pd
import requests

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def fetch_5min_60d(trading_symbol: str, timeout: float = 15.0) -> pd.DataFrame:
    """
    NSE symbols need a '.NS' suffix for Yahoo (e.g. 'RELIANCE.NS' -- this
    function adds it automatically, pass the plain trading symbol).
    Returns whatever 5-minute data Yahoo currently has for the last ~60
    calendar days -- not a date range you control, same "whatever the
    source currently has" caveat as fetch_recent_1min() in
    fetcher_yfinance.py. Empty DataFrame (same open/high/low/close/volume
    columns) on any failure -- never raises, so a bad symbol/network
    blip doesn't crash a caller iterating over many symbols.
    """
    yahoo_symbol = f"{trading_symbol.upper()}.NS"
    try:
        resp = requests.get(
            CHART_URL.format(symbol=yahoo_symbol),
            params={"interval": "5m", "range": "60d"},
            headers=_HEADERS, timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    chart = payload.get("chart", {})
    if chart.get("error") is not None:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    results = chart.get("result")
    if not results:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    result = results[0]
    timestamps = result.get("timestamp")
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    if not timestamps or not quote:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # epoch seconds -> tz-aware IST, then drop the tz so this matches
    # every other cached DataFrame's naive-IST-local convention (same
    # reasoning as fetcher_yfinance.py's tz handling).
    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None)

    df = pd.DataFrame({
        "open": quote.get("open"), "high": quote.get("high"),
        "low": quote.get("low"), "close": quote.get("close"),
        "volume": quote.get("volume"),
    }, index=index)

    # Drop null rows -- these are placeholder slots for days the market
    # didn't trade (holidays), not real missing data -- see module
    # docstring for the exact verification of this on RELIANCE.
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df.sort_index()
