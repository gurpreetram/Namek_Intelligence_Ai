"""
Cross-sectional intraday relative-strength ranking: for a given trading
day, ranks every cached symbol with usable data by its return from that
day's market open to a fixed formation cutoff time. This is the basis
for examples/run_cross_sectional_backtest.py's market-neutral long-top-N
/ short-bottom-N basket strategy.

RANKING BASIS CHOSEN, and why: raw % return from the day's open (reusing
detectors.sector_momentum.compute_stock_return_today() as-is, not
reimplemented), NOT detectors/relative_strength.py's benchmark-relative
excess return. Two reasons:
  1. No clean NSE index benchmark is actually integrated into this
     pipeline's cached data. storage/candles/ has NIFTY-TRACKING ETFs
     (e.g. BANKNIFTY1, BSLNIFTY) but not the real NIFTY 50 index itself
     -- sourcing/validating a genuine index benchmark would be its own
     separate data project, out of scope here.
  2. More fundamentally, this strategy shape doesn't need an external
     benchmark at all: ranking stocks against EACH OTHER and going long
     the top decile / short the bottom decile is ITSELF a benchmark-
     relative construction (relative to that day's cross-sectional mean),
     which is the standard academic/practitioner convention for a
     market-neutral factor portfolio (e.g. classic cross-sectional
     momentum/reversal decile studies). An external benchmark would add
     a second, redundant relative-strength reference on top of one the
     strategy already has built in by construction.

Same MIN_STOCK_PRICE floor as detectors/scanner.py -- excludes symbols
priced below it at the formation cutoff, so extreme % moves in illiquid
penny stocks don't dominate the ranking (same reasoning as scanner.py's
own filter, reused here rather than inventing a second threshold).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import pandas as pd

from detectors.scanner import MIN_STOCK_PRICE
from detectors.sector_momentum import compute_stock_return_today


@dataclass
class RankedSymbol:
    symbol: str
    return_pct: float  # % return from day's open to the formation cutoff
    formation_close: float  # price AT the cutoff bar -- entry fills at the NEXT bar's open, not this price


def rank_symbols_for_day(
    symbol_data: dict[str, pd.DataFrame],
    trading_day: date,
    formation_cutoff: str = "10:00",
    min_price: float = MIN_STOCK_PRICE,
) -> list[RankedSymbol]:
    """
    Ranks every symbol in `symbol_data` that has usable data on
    `trading_day`, descending by % return from that day's open to
    `formation_cutoff` (default 10:00 -- 45 minutes after the 09:15
    market open, giving the formation window enough time to reflect real
    relative strength rather than pure opening-print noise).

    Returns [] if no symbol qualifies (e.g. a day with no cached data at
    all, or every symbol below min_price).
    """
    cutoff_time = pd.Timestamp(formation_cutoff).time()
    ranked: list[RankedSymbol] = []

    for symbol, df in symbol_data.items():
        if df.empty:
            continue
        day_df = df[df.index.date == trading_day]
        if day_df.empty:
            continue

        cutoff_ts = pd.Timestamp.combine(trading_day, cutoff_time).tz_localize(day_df.index.tz)
        formation_df = day_df[day_df.index <= cutoff_ts]
        if formation_df.empty:
            continue

        formation_close = float(formation_df["close"].iloc[-1])
        if formation_close < min_price:
            continue

        ret = compute_stock_return_today(df, cutoff_ts)
        if ret is None:
            continue

        ranked.append(RankedSymbol(symbol=symbol, return_pct=ret, formation_close=formation_close))

    return sorted(ranked, key=lambda r: r.return_pct, reverse=True)


def next_bar_after(df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.Series | None:
    """The first 1-min bar strictly AFTER cutoff_ts -- the no-lookahead
    entry fill point (same 'next-bar-open fill' convention backtest/
    engine.py uses everywhere else). None if there isn't one that day."""
    trading_day = cutoff_ts.date()
    day_df = df[df.index.date == trading_day]
    after = day_df[day_df.index > cutoff_ts]
    if after.empty:
        return None
    return after.iloc[0]


def bar_at_or_before(df: pd.DataFrame, target_ts: pd.Timestamp) -> pd.Series | None:
    """The last 1-min bar at or before target_ts on the same day -- used
    for the EOD-square-off exit fill. None if there isn't one."""
    trading_day = target_ts.date()
    day_df = df[df.index.date == trading_day]
    at_or_before = day_df[day_df.index <= target_ts]
    if at_or_before.empty:
        return None
    return at_or_before.iloc[-1]
