"""
Sector-trend proxy: equal-weighted average return of a sector's constituent
stocks over a recent lookback window, computed entirely from our own cached
1-minute OHLCV data (data/storage.py) -- no new data source.

Deliberately simple, matching this project's own "reject features with low
ROI" principle: this is NOT a market-cap-weighted index replica, not
volume-adjusted, not survivorship-bias-corrected -- it's a plain average of
simple returns across whatever constituents we have price data for, meant
as a lightweight confluence signal, not a tradeable index proxy in its own
right.
"""
from __future__ import annotations

import pandas as pd


def compute_sector_trend(
    as_of: pd.Timestamp,
    sector_price_data: dict[str, pd.DataFrame],
    lookback_bars: int = 20,
) -> float | None:
    """
    sector_price_data: {symbol: full_1min_dataframe} for every symbol
        mapped to one sector -- typically loaded once via data/storage.py
        and reused across a whole backtest (see sector_confluence_trend_strategy.py).
        Each frame is truncated to `as_of` HERE, not by the caller -- this
        is the no-lookahead guard, same discipline as everywhere else in
        this codebase.
    lookback_bars: window (in 1-min bars) each constituent's return is
        measured over. Constituents with fewer than lookback_bars+1 bars
        available as of `as_of` are skipped (e.g. still early in the day).

    Returns the equal-weighted mean simple return (e.g. 0.015 == +1.5%)
    across whatever constituents had enough data, or None if none did.
    """
    returns: list[float] = []
    for df in sector_price_data.values():
        truncated = df.loc[:as_of]
        if len(truncated) < lookback_bars + 1:
            continue
        window = truncated.iloc[-lookback_bars:]
        start_price = float(window["close"].iloc[0])
        end_price = float(window["close"].iloc[-1])
        if start_price <= 0:
            continue
        returns.append((end_price - start_price) / start_price)

    if not returns:
        return None
    return sum(returns) / len(returns)


def load_sector_price_data(sector_mapping: dict[str, str]) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Loads and groups cached 1-min data by sector: {sector: {symbol: df}}.
    Loaded ONCE per backtest run (not per bar) -- callers should build this
    before running the backtest and pass the result into
    SectorConfluenceTrend, since re-loading from disk per bar would be the
    exact same redundant-I/O mistake already fixed elsewhere in this engine.
    """
    from data import storage

    by_sector: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol, sector in sector_mapping.items():
        df = storage.load(symbol)
        if df.empty:
            continue
        by_sector.setdefault(sector, {})[symbol] = df
    return by_sector
