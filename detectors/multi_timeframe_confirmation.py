"""
Checks whether multiple timeframes agree on trend direction before an
entry is trusted -- a classic institutional practice: higher timeframes
set context, lower timeframes time the entry, and a trade taken against
the higher timeframe is fighting a bigger current than the setup alone
suggests.

Trend direction per timeframe is measured via EMA(fast) vs EMA(slow)
relationship -- a standard, well-understood method (not this project's
invention): fast EMA above slow EMA = up-trending on that timeframe, and
vice versa. Simple by design -- this module's job is alignment checking,
not re-deriving trend detection (that's what market_regime.py's ADX
already does at a single-timeframe level; this checks CONSISTENCY across
timeframes).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


@dataclass
class AlignmentResult:
    aligned: bool
    direction: TrendDirection
    per_timeframe: dict[int, TrendDirection]
    reasoning: list[str]


def _ema_direction(df: pd.DataFrame, fast: int = 9, slow: int = 21,
                    flat_threshold_pct: float = 0.05) -> TrendDirection:
    """
    flat_threshold_pct: if the fast/slow EMAs are within this % of each
    other, call it FLAT rather than forcing a direction -- an EMA
    crossover with a razor-thin gap is noise, not a real trend read on
    that timeframe.
    """
    if len(df) < slow + 1:
        return TrendDirection.FLAT

    fast_ema = df["close"].ewm(span=fast, adjust=False).mean().iloc[-1]
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean().iloc[-1]

    if slow_ema == 0:
        return TrendDirection.FLAT

    gap_pct = abs(fast_ema - slow_ema) / slow_ema * 100
    if gap_pct < flat_threshold_pct:
        return TrendDirection.FLAT

    return TrendDirection.UP if fast_ema > slow_ema else TrendDirection.DOWN


def check_alignment(
    multi_tf_data: dict[int, pd.DataFrame],
    timeframes: list[int],
    fast: int = 9,
    slow: int = 21,
    require_all: bool = True,
) -> AlignmentResult:
    """
    multi_tf_data: {timeframe_minutes: df} -- typically from
        data.resample.multi_timeframe_view, already truncated for
        no-lookahead by the caller.
    timeframes: which timeframes must agree, e.g. [5, 15].
    require_all: if True, ALL given timeframes must agree on the same
        non-FLAT direction. If False, a simple majority is enough --
        looser, use when you want confirmation without demanding
        unanimous agreement across every timeframe.
    """
    per_timeframe: dict[int, TrendDirection] = {}
    for tf in timeframes:
        df = multi_tf_data.get(tf)
        direction = _ema_direction(df, fast, slow) if df is not None and not df.empty else TrendDirection.FLAT
        per_timeframe[tf] = direction

    up_count = sum(1 for d in per_timeframe.values() if d == TrendDirection.UP)
    down_count = sum(1 for d in per_timeframe.values() if d == TrendDirection.DOWN)

    reasoning = [
        f"{tf}min: {direction.value}" for tf, direction in per_timeframe.items()
    ]

    if require_all:
        if up_count == len(timeframes):
            return AlignmentResult(True, TrendDirection.UP, per_timeframe, reasoning)
        if down_count == len(timeframes):
            return AlignmentResult(True, TrendDirection.DOWN, per_timeframe, reasoning)
        return AlignmentResult(False, TrendDirection.FLAT, per_timeframe,
                                reasoning + ["Timeframes do NOT all agree -- rejecting"])
    else:
        majority_needed = len(timeframes) // 2 + 1
        if up_count >= majority_needed:
            return AlignmentResult(True, TrendDirection.UP, per_timeframe, reasoning)
        if down_count >= majority_needed:
            return AlignmentResult(True, TrendDirection.DOWN, per_timeframe, reasoning)
        return AlignmentResult(False, TrendDirection.FLAT, per_timeframe,
                                reasoning + ["No majority direction -- rejecting"])
