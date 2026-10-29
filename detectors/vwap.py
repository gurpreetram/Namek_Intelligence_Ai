"""
Intraday VWAP (Volume Weighted Average Price): cumulative sum of
(typical_price * volume) / cumulative volume, RESETTING at the start of
every trading day -- VWAP is inherently an intraday, session-based
benchmark (the institutional "did I get a good fill today" reference
price), not a rolling multi-day indicator. Computing it across day
boundaries without resetting would silently blend separate trading
sessions into one meaningless running average.

typical_price = (high + low + close) / 3, the standard VWAP convention
(not just close) -- incorporates the bar's full range, not only where it
settled.
"""
from __future__ import annotations

import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    df: 1-minute OHLCV data (any number of trading days) -- must have
    open/high/low/close/volume columns and a datetime index.

    Returns a Series aligned to df's index: each bar's VWAP computed from
    cumulative (typical_price * volume) / cumulative volume, reset at the
    first bar of each calendar day. A bar with zero cumulative volume so
    far today (the very first tick before any volume prints, or a
    genuinely dead/illiquid symbol) falls back to that bar's own
    typical_price rather than propagating NaN/inf from a divide-by-zero.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]

    day = df.index.to_series().dt.date
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()

    return (cum_pv / cum_vol).where(cum_vol > 0, typical_price)
