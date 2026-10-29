"""
"Strong Trend Stocks" for the Daily Picks (F4) dashboard tab -- an
informational list of stocks holding a consistent intraday trend, meant
for extended-hold/partial-exit consideration, NOT a new strategy or
backtested signal (see STRONG_TREND_DISCLAIMER in api/main.py, which the
frontend must keep visible at all times, same convention as detectors/
sector_momentum.py's and detectors/crash_watch.py's informational
sections).

Five gates, ALL must pass for a symbol to appear (direction is inferred
from which side the first gate qualifies on, not passed in):
  1. Initial move: >= min_move_pct from today's open -- reuses detectors/
     pick_verification.py's check_minimum_move() rather than
     reimplementing the "% move from open" definition a second time.
  2. Trend consistency: within the last window_minutes of cached data
     (resampled to 5-min bars via data/resample.py -- reused, not
     reimplemented), a run of >= min_consecutive bars each making a
     higher high AND higher low (LONG) or lower high AND lower low
     (SHORT) than the bar before it, ending at the most recent bar.
  3. Volume consistency: the last 3 five-minute bars each have volume >=
     volume_min_ratio of today's average 5-min bar volume so far.
  4. Opening-range hold: same invalidation condition examples/
     opening_range_breakout.py's check_invalidation() uses for a real ORB
     trade (closing back through the broken opening-range level) -- here
     checked across the WHOLE day so far, not just the current bar, so a
     stock that broke out and later failed doesn't still qualify.
  5. Sector alignment (only if the symbol has a sector mapping): reuses
     detectors/sector_momentum.py's compute_sector_momentum() to check the
     stock's sector is trending the same direction. Unmapped symbols pass
     this gate vacuously (most of the ~2373-symbol universe has no
     official sector label -- see sector_momentum.py's coverage note).

IMPORTANT: sector_momentum.py's own module docstring warns against wiring
sector agreement in as a hard trade-entry GATE (SectorConfluenceTrend
tried that and made results worse). Gate 5 here is still a hard filter,
but for a display-only informational list explicitly disclaimed as "not a
backtested signal" -- not a trade entry condition -- which is a different
use than the one that docstring warns against.

check_trend_consistency(), check_volume_consistency(), check_opening_
range_hold(), and find_trend_start() (gates 2-4 above, plus the display-
duration helper) are public -- promoted from module-private on 2026-08-17
when detectors/momentum_entry_strategy.py became a second real caller,
reusing exactly these gates (NOT gate 1's 1.5% threshold or gate 5's
sector check, both specific to this module's own informational list) as
part of an actual trade-entry Strategy's evaluate(). Keep signatures
stable for that caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backtest.strategy_base import SignalDirection
from data.resample import resample
from detectors.pick_verification import check_minimum_move
from detectors.sector_momentum import compute_sector_momentum, pick_reference_as_of

STRONG_TREND_MIN_MOVE_PCT = 1.5
STRONG_TREND_WINDOW_MINUTES = 60
STRONG_TREND_MIN_CONSECUTIVE = 3
STRONG_TREND_VOLUME_MIN_RATIO = 0.70
STRONG_TREND_OPENING_RANGE_MINUTES = 15
STRONG_TREND_RESAMPLE_MINUTES = 5


@dataclass
class StrongTrendStock:
    symbol: str
    direction: SignalDirection
    current_price: float
    move_pct_from_open: float
    # First bar where the move from open (in `direction`) reached
    # min_move_pct AND has stayed at or beyond it continuously ever since
    # (walked backward from the latest bar) -- deliberately NOT the same
    # timestamp as gate 2's run_start (that's bounded to window_minutes by
    # design, so it can never show more than ~window_minutes of duration;
    # this answers "how long has the qualifying move actually held",
    # which is the more useful display value and can span hours).
    trending_since: pd.Timestamp
    volume_status: str  # human-readable, from check_volume_consistency()
    sector: Optional[str] = None
    sector_aligned: Optional[bool] = None  # None if the symbol has no sector mapping


def _todays_bars(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    truncated = df.loc[:as_of]
    return truncated[truncated.index.date == as_of.date()]


def _consecutive_run_end_timestamp(
    five_min: pd.DataFrame, direction: SignalDirection, min_consecutive: int,
) -> Optional[pd.Timestamp]:
    """Start timestamp of the longest run of consecutive HH+HL (LONG) /
    LH+LL (SHORT) 5-min bars ENDING at the last bar in `five_min` -- None
    if that run is shorter than min_consecutive bars. Must end at the
    last bar (not just exist anywhere in the window) since the point is
    "is the trend CURRENTLY still intact", not "did it ever happen"."""
    if len(five_min) < min_consecutive + 1:
        return None
    highs = five_min["high"].values
    lows = five_min["low"].values
    n = len(five_min)
    run_len = 0
    for i in range(n - 1, 0, -1):
        if direction == SignalDirection.LONG:
            step_ok = highs[i] > highs[i - 1] and lows[i] > lows[i - 1]
        else:
            step_ok = highs[i] < highs[i - 1] and lows[i] < lows[i - 1]
        if not step_ok:
            break
        run_len += 1
    if run_len >= min_consecutive:
        return five_min.index[n - 1 - run_len]
    return None


def check_trend_consistency(
    day_df: pd.DataFrame, as_of: pd.Timestamp, direction: SignalDirection,
    window_minutes: int, min_consecutive: int,
) -> tuple[bool, str]:
    five_min = resample(day_df, STRONG_TREND_RESAMPLE_MINUTES)
    if five_min.empty:
        return False, "Not enough data to resample to 5-min bars"
    windowed = five_min[five_min.index >= as_of - pd.Timedelta(minutes=window_minutes)]
    run_start = _consecutive_run_end_timestamp(windowed, direction, min_consecutive)
    label = "higher-high+higher-low" if direction == SignalDirection.LONG else "lower-high+lower-low"
    if run_start is None:
        return False, f"No run of >= {min_consecutive} consecutive {label} 5-min bars in the last {window_minutes} min"
    return True, f"{label} sequence intact since {run_start.strftime('%H:%M')}"


def check_volume_consistency(day_df: pd.DataFrame, min_ratio: float) -> tuple[bool, str]:
    five_min = resample(day_df, STRONG_TREND_RESAMPLE_MINUTES)
    if len(five_min) < 4:
        return False, "Not enough 5-min bars yet to judge volume consistency"
    avg_bar_volume = float(five_min["volume"].mean())
    if avg_bar_volume <= 0:
        return False, "No volume data available"
    last_3 = five_min["volume"].iloc[-3:]
    threshold = avg_bar_volume * min_ratio
    if bool((last_3 >= threshold).all()):
        min_ratio_seen = float(last_3.min()) / avg_bar_volume
        return True, f"Volume steady -- last 3 five-min bars all >= {min_ratio:.0%} of today's avg ({min_ratio_seen:.0%} lowest)"
    return False, f"Volume has dried up -- at least one of the last 3 five-min bars fell below {min_ratio:.0%} of today's avg"


def check_opening_range_hold(
    day_df: pd.DataFrame, direction: SignalDirection, range_minutes: int,
) -> tuple[bool, str]:
    """Same condition as examples/opening_range_breakout.py's
    check_invalidation() (closing back through the broken opening-range
    level = the breakout failed), applied across the whole day so far
    rather than just the current bar. Only checked from the FIRST
    breakout bar onward -- bars before the breakout naturally sit on the
    wrong side of the level and aren't a "broke back" event."""
    market_open = day_df.index[0]
    range_end = market_open + pd.Timedelta(minutes=range_minutes)
    range_bars = day_df[day_df.index < range_end]
    if range_bars.empty:
        return False, "Not enough bars yet to establish the opening range"
    range_high = float(range_bars["high"].max())
    range_low = float(range_bars["low"].min())
    post_range = day_df[day_df.index >= range_end]
    if post_range.empty:
        return False, "Not enough bars yet since the opening range closed"

    if direction == SignalDirection.LONG:
        broke_out = post_range[post_range["close"] > range_high]
        if broke_out.empty:
            return False, f"Hasn't broken above the opening range high ({range_high:.2f}) yet today"
        after = post_range[post_range.index >= broke_out.index[0]]
        if bool((after["close"] < range_high).any()):
            return False, f"Closed back below the opening range high ({range_high:.2f}) after breaking out"
        return True, f"Held above the opening range high ({range_high:.2f}) since breaking out at {broke_out.index[0].strftime('%H:%M')}"
    else:
        broke_out = post_range[post_range["close"] < range_low]
        if broke_out.empty:
            return False, f"Hasn't broken below the opening range low ({range_low:.2f}) yet today"
        after = post_range[post_range.index >= broke_out.index[0]]
        if bool((after["close"] > range_low).any()):
            return False, f"Closed back above the opening range low ({range_low:.2f}) after breaking out"
        return True, f"Held below the opening range low ({range_low:.2f}) since breaking out at {broke_out.index[0].strftime('%H:%M')}"


def find_trend_start(
    day_df: pd.DataFrame, direction: SignalDirection, open_price: float, min_move_pct: float,
) -> pd.Timestamp:
    """Backward walk from the last bar (same pattern as detectors/
    pick_verification.py's _infer_trigger_time()) to find how long the
    qualifying move has held continuously -- see StrongTrendStock.
    trending_since's docstring for why this differs from gate 2's window."""
    if direction == SignalDirection.LONG:
        move_pct = (day_df["close"] - open_price) / open_price * 100.0
    else:
        move_pct = (open_price - day_df["close"]) / open_price * 100.0
    beyond = move_pct >= min_move_pct
    idx = len(beyond) - 1
    while idx > 0 and bool(beyond.iloc[idx - 1]):
        idx -= 1
    return day_df.index[idx]


def find_strong_trend_stocks(
    symbol_data: dict[str, pd.DataFrame],
    sector_mapping: dict[str, str],
    as_of: pd.Timestamp | None = None,
    min_move_pct: float = STRONG_TREND_MIN_MOVE_PCT,
    window_minutes: int = STRONG_TREND_WINDOW_MINUTES,
    min_consecutive: int = STRONG_TREND_MIN_CONSECUTIVE,
    volume_min_ratio: float = STRONG_TREND_VOLUME_MIN_RATIO,
    opening_range_minutes: int = STRONG_TREND_OPENING_RANGE_MINUTES,
    top_n_per_side: int = 5,
) -> tuple[list[StrongTrendStock], list[StrongTrendStock]]:
    """
    Returns (longs, shorts), each sorted strongest-move-first and capped
    to top_n_per_side. as_of=None picks the same shared reference
    timestamp compute_sector_momentum() would (see pick_reference_as_of) --
    pass it explicitly if the caller already computed it, so the reported
    as_of always matches what was actually used.
    """
    if as_of is None:
        as_of = pick_reference_as_of(symbol_data)
        if as_of is None:
            return [], []

    # Computed once for the whole batch (gate 5), not re-derived per
    # symbol -- reuses detectors/sector_momentum.py as-is.
    sector_results = compute_sector_momentum(sector_mapping, symbol_data, as_of=as_of)
    sector_return_by_name = {r.sector: r.return_pct for r in sector_results}

    candidates: list[StrongTrendStock] = []
    for symbol, df in symbol_data.items():
        day_df = _todays_bars(df, as_of)
        if day_df.empty:
            continue
        current_price = float(day_df["close"].iloc[-1])
        open_price = float(day_df["open"].iloc[0])
        if open_price <= 0:
            continue

        raw_move_pct = (current_price - open_price) / open_price * 100.0
        if raw_move_pct >= min_move_pct:
            direction = SignalDirection.LONG
        elif raw_move_pct <= -min_move_pct:
            direction = SignalDirection.SHORT
        else:
            continue  # gate 1 failed

        move_ok, _ = check_minimum_move(day_df, current_price, direction, min_pct=min_move_pct)
        if not move_ok:
            continue

        consistent, _ = check_trend_consistency(day_df, as_of, direction, window_minutes, min_consecutive)
        if not consistent:
            continue

        volume_ok, volume_reason = check_volume_consistency(day_df, volume_min_ratio)
        if not volume_ok:
            continue

        range_ok, _ = check_opening_range_hold(day_df, direction, opening_range_minutes)
        if not range_ok:
            continue

        sector = sector_mapping.get(symbol)
        sector_aligned: Optional[bool] = None
        if sector is not None and sector in sector_return_by_name:
            sector_return = sector_return_by_name[sector]
            sector_aligned = (sector_return > 0) if direction == SignalDirection.LONG else (sector_return < 0)
            if not sector_aligned:
                continue  # gate 5 failed -- sector trending against this stock

        trending_since = find_trend_start(day_df, direction, open_price, min_move_pct)

        candidates.append(StrongTrendStock(
            symbol=symbol, direction=direction, current_price=current_price,
            move_pct_from_open=raw_move_pct, trending_since=trending_since,
            volume_status=volume_reason, sector=sector, sector_aligned=sector_aligned,
        ))

    longs = sorted(
        (c for c in candidates if c.direction == SignalDirection.LONG),
        key=lambda c: c.move_pct_from_open, reverse=True,
    )[:top_n_per_side]
    shorts = sorted(
        (c for c in candidates if c.direction == SignalDirection.SHORT),
        key=lambda c: c.move_pct_from_open,
    )[:top_n_per_side]
    return longs, shorts
