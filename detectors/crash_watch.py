"""
Live "Crash Watch" detector: which stocks, in the cached universe, are
currently showing a rapid, large intraday decline -- the same pattern
examples/run_crash_detection.py and examples/run_crash_detection_refinements.py
tested exhaustively as a TRADING hypothesis (>=2.5% decline in a trailing
15-minute window, on elevated volume) and found NO edge in: individual
continuation (SHORT), individual reversal (LONG), sector-impact spread,
confirmation-filtered entries, severity tiers, or time-of-day splits (420+
opening events plus 1400+ across the four refinements, none cleared the
project's pre-registered Sharpe>=0.5/win-rate>=60% bar).

THIS MODULE IS INFORMATIONAL ONLY. It answers "where did a crash happen
today", not "what to do about it" -- there is deliberately no entry,
target, or stop-loss anywhere in this module or its API/frontend
consumers, because there is no validated strategy to attach one to.

METHODOLOGY -- matches Refinement 4 (examples/run_crash_detection_refinements.py)
exactly, the "more general, any-time-of-day" variant, not the original
opening-window-only detector, since a live dashboard needs to catch a
crash whenever it happens during the day, not just 09:15-09:30:
  - return definition: a TRAILING 15-minute return (close now vs close
    ~15 bars back), not "since today's open" -- measuring since-open at
    13:00 would just be the day's cumulative drift, not a "rapid" move.
  - volume baseline: SAME-DAY rolling (the preceding 30 bars on the SAME
    day), reusing detectors.breakout_validator.volume_zscore()'s exact
    z-score formula -- NOT the original opening-detector's prior-day
    time-of-day-matched pool, which has no single matching slot to pool
    across days for an arbitrary minute. Requires >=30 preceding same-day
    bars, which naturally excludes the first ~30 minutes of trading (a
    deliberate, documented limitation, not an oversight -- see
    run_crash_detection_refinements.py's own module docstring for the
    full reasoning, identical here).
  - liquidity filter: detectors.liquidity_analysis.analyze_liquidity() on
    the most recent PRIOR trading day's data, applied BEFORE a symbol is
    even eligible -- same "liquidity filter from the start" discipline
    used throughout the crash-detection research, not bolted on after.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from detectors.liquidity_analysis import analyze_liquidity
from detectors.scanner import MIN_STOCK_PRICE

DECLINE_THRESHOLD_PCT = -2.5
VOLUME_ZSCORE_MIN = 1.0
RETURN_LOOKBACK_MINUTES = 15
SAME_DAY_LOOKBACK_BARS = 30


@dataclass
class CrashWatchHit:
    symbol: str
    trigger_time: pd.Timestamp
    decline_pct: float
    volume_zscore: float
    current_price: float
    avg_traded_value: float  # liquidity proxy, Rs./min -- see detectors/liquidity_analysis.py


def scan_for_crashes(
    symbol_data: dict[str, pd.DataFrame],
    as_of_date: date,
    require_liquid: bool = True,
    decline_threshold_pct: float = DECLINE_THRESHOLD_PCT,
    volume_zscore_min: float = VOLUME_ZSCORE_MIN,
) -> list[CrashWatchHit]:
    """
    symbol_data: {symbol: full cached df} -- must contain at least one
        prior trading day (for the liquidity filter) plus as_of_date's
        own bars. A symbol missing either is silently skipped (excluded,
        not assumed liquid/eligible -- same conservative convention
        run_crash_detection.py uses).
    as_of_date: restricts detection to THIS date's own bars only, even if
        symbol_data's df also contains other days -- "crashed TODAY", not
        any day this df happens to cover.

    Returns one hit per qualifying symbol (its FIRST qualifying bar
    today, matching the backtest convention), sorted deepest decline
    first. A symbol can appear at most once.
    """
    hits: list[CrashWatchHit] = []
    for symbol, df in symbol_data.items():
        today_df = df[df.index.date == as_of_date]
        if today_df.empty:
            continue
        prior_df = df[df.index.date < as_of_date]
        if prior_df.empty:
            continue

        yesterday_close = float(prior_df["close"].iloc[-1])
        if yesterday_close < MIN_STOCK_PRICE:
            continue
        liquidity = analyze_liquidity(prior_df)
        if require_liquid and (liquidity is None or not liquidity.is_liquid):
            continue

        shifted_close = today_df["close"].shift(RETURN_LOOKBACK_MINUTES)
        pct = (today_df["close"] - shifted_close) / shifted_close * 100.0

        roll_mean = today_df["volume"].shift(1).rolling(SAME_DAY_LOOKBACK_BARS).mean()
        roll_std = today_df["volume"].shift(1).rolling(SAME_DAY_LOOKBACK_BARS).std()
        vol_z = (today_df["volume"] - roll_mean) / roll_std

        trigger_mask = (pct <= decline_threshold_pct) & (vol_z >= volume_zscore_min)
        trigger_mask = trigger_mask.fillna(False)
        if not trigger_mask.any():
            continue

        trigger_time = trigger_mask[trigger_mask].index[0]
        hits.append(CrashWatchHit(
            symbol=symbol, trigger_time=trigger_time,
            decline_pct=float(pct.loc[trigger_time]), volume_zscore=float(vol_z.loc[trigger_time]),
            current_price=float(today_df["close"].iloc[-1]),
            avg_traded_value=liquidity.avg_traded_value if liquidity is not None else float("nan"),
        ))

    hits.sort(key=lambda h: h.decline_pct)
    return hits
