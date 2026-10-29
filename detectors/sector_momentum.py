"""
Sector-level and within-sector stock momentum for the Sectors dashboard
tab -- informational/contextual ("which sectors are hot today, and which
stocks are driving it"), explicitly NOT a trading signal or strategy
input. detectors/sector_trend.py already exists and is used by
SectorConfluenceTrend as a hard bar-by-bar confluence GATE on trade
entries -- that experiment (sector agreement required to trade) made
results WORSE on thin (21-symbol) coverage, confirmed via backtest
comparison. This module is deliberately NOT that: it has no opinion
about entries/targets/stops, it just ranks and displays. Don't wire this
into a strategy without re-deriving that decision from scratch.

Also deliberately NOT the same computation as sector_trend.py's
compute_sector_trend(): that function uses a rolling recent-bar lookback
window (suited to a bar-by-bar confluence check). This module's default
(lookback_minutes=None) answers "how has this sector moved TODAY, from
market open to now" -- a fixed intraday-so-far window.

Optional lookback_minutes (added for the Sectors tab's "Live Momentum
Picks" section): a ROLLING trailing window instead of since-open. IMPORTANT
CONTEXT for that use: examples/research_sector_momentum_predictive.py and
examples/research_sector_momentum_rolling.py both tested whether this
rolling-detection idea (a sector moving NOW predicts its stocks' NEAR-TERM
future momentum) has any real edge over random selection, across 4
lookback/forward combinations on 21 days of cached history. Neither script
found one -- every combination's day-level edge was far below the pre-
committed bar, despite a mildly suggestive (but not tradeable, not
statistically solid on this sample size) win-rate pattern in 3 of 4
combos. "Live Momentum Picks" ships anyway at the user's explicit request
made with that finding in hand -- it is display-only, informational, NOT
validated as predictive, and the frontend must keep that disclaimer
visible at all times, not just on first load. The specific lookback used
there (15 minutes) is an arbitrary reasonable default, not a value the
research identified as better than the alternatives -- it wasn't.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from detectors.support_resistance import average_true_range


@dataclass
class StockMomentum:
    symbol: str
    return_pct: float  # today's open-to-as_of % return, e.g. 1.23 == +1.23%
    # Qualitative size of that move relative to this stock's OWN typical
    # full daily range -- "small" | "moderate" | "strong" | "unknown" (not
    # enough prior-day history to judge). See classify_move_magnitude().
    magnitude: str = "unknown"


@dataclass
class SectorMomentum:
    sector: str
    return_pct: float  # equal-weighted average of its constituents' return_pct
    num_constituents: int  # how many cached, sector-labeled symbols had usable data today
    top_stocks: list[StockMomentum] = field(default_factory=list)
    magnitude: str = "unknown"  # same classification, against the sector's avg constituent ATR%


def pick_reference_as_of(symbol_data: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    """The single shared 'today' timestamp compute_sector_momentum() uses
    when as_of=None -- the majority (most common) last-cached DATE across
    symbol_data, using the latest timestamp on that date. Exposed
    separately so a caller (e.g. the API) can report the SAME as_of it
    actually computed with, instead of naively taking max() across all
    symbols and reporting a date that doesn't match what was used (an
    outlier symbol one day ahead would misreport the reference date)."""
    max_per_symbol = [df.index.max() for df in symbol_data.values() if not df.empty]
    if not max_per_symbol:
        return None
    reference_date = Counter(ts.date() for ts in max_per_symbol).most_common(1)[0][0]
    return max(ts for ts in max_per_symbol if ts.date() == reference_date)


def compute_stock_return_today(
    df: pd.DataFrame, as_of: pd.Timestamp, lookback_minutes: int | None = None,
) -> float | None:
    """
    % return from a reference point to the latest bar at/before as_of's
    close. None if there's no usable data.

    lookback_minutes=None (default): reference point is today's first
        bar's open -- "how far has this stock moved since market open."
    lookback_minutes=N: reference point is N minutes before as_of --
        "how far has this stock moved in just the last N minutes." Still
        clipped to today's own bars (never reaches into a prior day).
    """
    today = as_of.date()
    truncated = df.loc[:as_of]
    today_bars = truncated[truncated.index.date == today]
    if today_bars.empty:
        return None
    if lookback_minutes is not None:
        window_start = as_of - pd.Timedelta(minutes=lookback_minutes)
        today_bars = today_bars[today_bars.index >= window_start]
        if today_bars.empty:
            return None
    open_price = float(today_bars["open"].iloc[0])
    last_price = float(today_bars["close"].iloc[-1])
    if open_price <= 0:
        return None
    return (last_price - open_price) / open_price * 100.0


def compute_daily_atr_pct(df: pd.DataFrame, as_of: pd.Timestamp, period: int = 14) -> float | None:
    """
    This stock's typical FULL trading day's range, as a % of its latest
    price -- "how big a move is normal for this stock." Built by
    aggregating the cached 1-min bars into daily OHLC bars, then reusing
    average_true_range()'s exact true-range formula (support_resistance.py)
    on those daily bars, rather than reimplementing true-range here.

    Only uses bars STRICTLY BEFORE as_of's date -- today's own (still-
    forming, partial) day is excluded so an early-morning read never
    biases the "typical" range estimate low. None if there isn't at least
    2 prior trading days cached (average_true_range needs >=2 bars to
    compute even one true-range value).
    """
    prior = df[df.index.date < as_of.date()]
    if prior.empty:
        return None
    daily = prior.groupby(prior.index.date).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
    )
    if len(daily) < 2:
        return None
    atr = average_true_range(daily, period=period)
    last_close = float(daily["close"].iloc[-1])
    if last_close <= 0 or atr <= 0:
        return None
    return atr / last_close * 100.0


def classify_move_magnitude(return_pct: float, atr_pct: float | None) -> str:
    """
    Qualitative label for today's-so-far move, relative to this stock's
    OWN typical full-day range -- not a fixed %-of-price threshold, since
    e.g. a 1% move is huge for a low-volatility stock and unremarkable for
    a high-volatility one. Thresholds are a reasonable, simple split (not
    backtested/validated -- this is a display aid, not a signal, same
    spirit as this module's other informational-only content):
      < 30% of typical daily range so far  -> "small"
      30-80% of typical daily range so far -> "moderate"
      >= 80% of typical daily range so far -> "strong"
    "unknown" when atr_pct isn't available (not enough prior-day history).
    """
    if atr_pct is None or atr_pct <= 0:
        return "unknown"
    ratio = abs(return_pct) / atr_pct
    if ratio < 0.3:
        return "small"
    if ratio < 0.8:
        return "moderate"
    return "strong"


def compute_sector_momentum(
    sector_mapping: dict[str, str],
    symbol_data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp | None = None,
    top_n_stocks: int = 5,
    lookback_minutes: int | None = None,
) -> list[SectorMomentum]:
    """
    symbol_data: {symbol: full_1min_dataframe} -- typically every cached
        symbol (only ones present in sector_mapping are used; the rest
        are ignored here, not an error).
    as_of: None (default) picks ONE shared reference timestamp for ALL
        symbols -- the majority (most common) last-cached DATE across
        symbol_data, using the latest timestamp on that date. This is
        deliberately DIFFERENT from detectors/scanner.py's scan_universe(),
        which uses each symbol's own latest bar independently -- correct
        there because each symbol is scanned independently, but WRONG
        here: this function averages returns ACROSS symbols within a
        sector, so every included symbol's return must be measured over
        the SAME trading session or the average silently mixes different
        days. Confirmed this isn't hypothetical: 5 of 180 real sector-
        mapped symbols had their latest cached bar one day ahead of the
        other 175 (bulk-fetch timing drift, same root cause documented in
        scan_universe()'s docstring). A symbol whose data doesn't reach
        the reference date is excluded from that sector's average (see
        compute_stock_return_today -- its today_bars slice comes back
        empty for such a symbol), not silently included with a
        mismatched-day return.

    Returns one SectorMomentum per sector that had at least one
    constituent with usable today-data, sorted by |return_pct| descending
    (most-trending first, in EITHER direction -- the caller splits into
    up/down columns for display; this function just ranks trend strength).

    Within-sector top_stocks are ranked by their move in the SAME
    direction as the sector's own trend (the stocks actually driving it),
    not by raw |return_pct| -- a lone stock moving hard against an
    otherwise-trending sector is a different kind of observation than
    "which stocks are leading this trend", and the latter is what this
    tab is for.
    """
    if as_of is None:
        as_of = pick_reference_as_of(symbol_data)
        if as_of is None:
            return []

    by_sector: dict[str, list[StockMomentum]] = {}
    atr_by_sector: dict[str, list[float]] = {}
    for symbol, sector in sector_mapping.items():
        df = symbol_data.get(symbol)
        if df is None or df.empty:
            continue
        ret = compute_stock_return_today(df, as_of, lookback_minutes=lookback_minutes)
        if ret is None:
            continue
        atr_pct = compute_daily_atr_pct(df, as_of)
        by_sector.setdefault(sector, []).append(
            StockMomentum(symbol=symbol, return_pct=ret, magnitude=classify_move_magnitude(ret, atr_pct))
        )
        if atr_pct is not None:
            atr_by_sector.setdefault(sector, []).append(atr_pct)

    results: list[SectorMomentum] = []
    for sector, stocks in by_sector.items():
        avg_return = sum(s.return_pct for s in stocks) / len(stocks)
        direction_sign = 1.0 if avg_return >= 0 else -1.0
        top_stocks = sorted(stocks, key=lambda s: s.return_pct * direction_sign, reverse=True)[:top_n_stocks]
        sector_atrs = atr_by_sector.get(sector, [])
        avg_atr_pct = sum(sector_atrs) / len(sector_atrs) if sector_atrs else None
        results.append(SectorMomentum(
            sector=sector, return_pct=avg_return, num_constituents=len(stocks), top_stocks=top_stocks,
            magnitude=classify_move_magnitude(avg_return, avg_atr_pct),
        ))

    results.sort(key=lambda r: abs(r.return_pct), reverse=True)
    return results


def select_top_bottom_sectors(
    results: list[SectorMomentum], n: int = 2, min_constituents: int = 3,
) -> list[SectorMomentum]:
    """
    Top-n most-positive and top-n most-negative sectors from an already-
    computed compute_sector_momentum() result -- the "top-2 up / top-2
    down" selection used by examples/research_sector_momentum_rolling.py
    and the Sectors tab's "Live Momentum Picks" section. min_constituents
    matches that research script's MIN_SECTOR_MEMBERS gate: a sector's
    return computed from only 1-2 stocks isn't a real sector signal.
    Returns up to 2*n sectors (fewer if not enough qualify), sorted
    positive-first then negative-first within each group.
    """
    eligible = [r for r in results if r.num_constituents >= min_constituents]
    ranked = sorted(eligible, key=lambda r: r.return_pct, reverse=True)
    up = [r for r in ranked if r.return_pct >= 0][:n]
    down = list(reversed([r for r in ranked if r.return_pct < 0][-n:]))
    return up + down
