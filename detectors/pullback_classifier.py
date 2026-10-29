"""
Classifies a pullback against the prevailing trend as likely CONTINUATION
(the trend resumes) or likely REVERSAL (the trend is actually turning) --
directly targeting: "after one red candle we thought sellers active and
went opposite, but after one candle it again pump and move same direction"
and its mirror case, "sometimes the pullback is fake after 1-2 candles it
goes the same direction" vs the times it doesn't.

Research grounding for the factors used:
  - Retracement depth: how far the pullback has cut into the prior trend
    leg, measured as a fraction of that leg's range. Shallow retracements
    (roughly <38%) are classically associated with strong trends that are
    likely to resume; deep retracements (>61%) are associated with trends
    that are genuinely weakening (this is the standard Fibonacci-retracement
    heuristic used across technical trading literature -- treated here as
    a probability weighting, not a magic number).
  - Volume contraction: in a healthy trend, pullbacks typically occur on
    LOWER volume than the trend leg itself -- participants pushing the
    trend aren't the ones selling into the dip, it's lighter profit-taking.
    A pullback on volume that MATCHES OR EXCEEDS the trend leg's volume
    suggests real distribution/accumulation against the trend, not just
    profit-taking -- a genuine warning sign, not noise.
  - Candle body compression: indecision candles (small real bodies,
    long wicks, doji-like) during a pullback suggest the counter-move
    lacks conviction. Large-bodied candles firmly in the counter-trend
    direction suggest real participation on the other side.
  - Structure break: does the pullback take price beyond the most recent
    swing point in the trend direction (e.g., in an uptrend, does the
    pullback low break below the prior swing low)? This is the single
    strongest signal -- it's the market-structure definition of a
    potential trend change, not just a deep pullback within an intact
    structure.

As with the breakout validator: no single factor is trusted alone. This
is intentionally the same design philosophy as Phase 2 -- multiple
independent factors must agree before a call is made, and "uncertain" is
a valid, honest output when they don't.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from detectors.support_resistance import average_true_range


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"


class PullbackVerdict(Enum):
    LIKELY_CONTINUATION = "likely_continuation"
    LIKELY_REVERSAL = "likely_reversal"
    UNCERTAIN = "uncertain"


@dataclass
class TrendLeg:
    direction: TrendDirection
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_price: float
    end_price: float
    avg_volume: float

    @property
    def range_points(self) -> float:
        return abs(self.end_price - self.start_price)


@dataclass
class PullbackClassification:
    verdict: PullbackVerdict
    confidence: float
    reasoning: list[str]
    retracement_pct: float
    volume_ratio: float  # pullback volume / trend leg volume
    structure_broken: bool


def find_recent_trend_leg(
    df: pd.DataFrame,
    lookback_bars: int = 20,
    min_move_atr_mult: float = 2.0,
    min_directional_consistency: float = 0.65,
) -> TrendLeg | None:
    """
    Looks back over `lookback_bars` for a directional move large enough
    (relative to ATR) to call it a trend leg worth evaluating pullbacks
    against. Returns None if recent price action is too choppy/small to
    qualify -- classifying a pullback against a leg that isn't really a
    trend produces meaningless output, so this guards against that.

    min_directional_consistency: fraction of bars within the window that
    must close in the trend's direction. This exists because a pure
    random walk will often drift far enough to clear the ATR-multiple
    distance check by chance alone (net displacement over N random steps
    grows with sqrt(N), so it's easy to accidentally "look like" a trend)
    -- but a real trend also has most of its individual bars moving the
    same way, not just a lucky net sum. Verified empirically: without
    this check, pure synthetic noise was misclassified as valid trend
    legs; this check is what closes that gap.
    """
    if len(df) < lookback_bars + 5:
        return None

    window = df.iloc[-lookback_bars:]
    # Confirmed bug: this was passing the full, ever-growing `df` (all
    # history up to current_index) instead of the already-bounded `window`
    # computed above -- ATR's rolling(14) only needs window's ~20 rows to
    # produce the identical value (window's last 14-15 rows ARE df's last
    # 14-15 rows), so this was pure wasted recomputation, not a correctness
    # difference.
    atr = average_true_range(window)
    if atr <= 0:
        return None

    start_price = float(window["close"].iloc[0])
    end_price = float(window["close"].iloc[-2])  # exclude the current/last bar -- that's the candidate pullback bar
    move = end_price - start_price

    if abs(move) < atr * min_move_atr_mult:
        return None

    direction = TrendDirection.UP if move > 0 else TrendDirection.DOWN

    # Directional consistency: what fraction of individual bar-to-bar
    # closes moved in the trend direction (vs against it)?
    bar_deltas = window["close"].diff().dropna()
    if direction == TrendDirection.UP:
        consistent_bars = (bar_deltas > 0).sum()
    else:
        consistent_bars = (bar_deltas < 0).sum()
    consistency_ratio = consistent_bars / len(bar_deltas) if len(bar_deltas) > 0 else 0.0

    if consistency_ratio < min_directional_consistency:
        return None

    return TrendLeg(
        direction=direction,
        start_time=window.index[0],
        end_time=window.index[-2],
        start_price=start_price,
        end_price=end_price,
        avg_volume=float(window["volume"].iloc[:-1].mean()),
    )


def classify_pullback(
    df: pd.DataFrame,
    trend_leg: TrendLeg,
    pullback_bars: int = 2,
) -> PullbackClassification | None:
    """
    df: 1-minute data up to and including the current (candidate pullback)
        bar -- caller's responsibility to slice for no-lookahead.
    trend_leg: the leg this pullback is being evaluated against (from
        find_recent_trend_leg, computed on data strictly before the
        pullback bars).
    pullback_bars: how many of the most recent bars constitute "the
        pullback" being classified.
    """
    if len(df) < pullback_bars + 1:
        return None

    pullback_window = df.iloc[-pullback_bars:]
    current_price = float(pullback_window["close"].iloc[-1])

    if trend_leg.direction == TrendDirection.UP:
        retracement = trend_leg.end_price - current_price
    else:
        retracement = current_price - trend_leg.end_price

    if trend_leg.range_points == 0:
        return None
    retracement_pct = retracement / trend_leg.range_points

    # Negative retracement means price hasn't actually pulled back at all
    # (it's still extending the trend) -- nothing to classify.
    if retracement_pct <= 0:
        return None

    pullback_volume = float(pullback_window["volume"].mean())
    volume_ratio = pullback_volume / trend_leg.avg_volume if trend_leg.avg_volume > 0 else 1.0

    # Candle body compression: average |close-open| relative to average
    # bar range across the pullback window.
    bodies = (pullback_window["close"] - pullback_window["open"]).abs()
    ranges = (pullback_window["high"] - pullback_window["low"])
    body_ratios = (bodies / ranges.replace(0, np.nan)).dropna()
    avg_body_ratio = float(body_ratios.mean()) if not body_ratios.empty else 0.5
    indecisive_candles = avg_body_ratio < 0.4  # small bodies relative to range = indecision

    # Structure break check: did the pullback take price beyond the swing
    # point at the start of the trend leg?
    if trend_leg.direction == TrendDirection.UP:
        structure_broken = current_price < trend_leg.start_price
    else:
        structure_broken = current_price > trend_leg.start_price

    shallow = retracement_pct < 0.382
    deep = retracement_pct > 0.618
    volume_contracted = volume_ratio < 0.8
    volume_expanded = volume_ratio > 1.2

    continuation_votes = sum([shallow, volume_contracted, indecisive_candles, not structure_broken])
    reversal_votes = sum([deep, volume_expanded, not indecisive_candles, structure_broken])

    reasoning = [
        f"Trend leg: {trend_leg.direction.value} {trend_leg.range_points:.2f} pts "
        f"({trend_leg.start_time.strftime('%H:%M')}-{trend_leg.end_time.strftime('%H:%M')})",
        f"Retracement: {retracement_pct:.1%} of the leg "
        f"({'shallow' if shallow else 'deep' if deep else 'moderate'})",
        f"Pullback volume vs trend-leg volume: {volume_ratio:.2f}x "
        f"({'contracted' if volume_contracted else 'expanded' if volume_expanded else 'similar'})",
        f"Pullback candles: {'indecisive/small-bodied' if indecisive_candles else 'strong-bodied'} "
        f"(avg body/range ratio {avg_body_ratio:.2f})",
        f"Structure: {'BROKEN -- pullback exceeded trend leg start' if structure_broken else 'intact'}",
    ]

    if structure_broken:
        # Structure break overrides -- this is the strongest single signal,
        # matches the design note above.
        verdict = PullbackVerdict.LIKELY_REVERSAL
        confidence = 0.6 + 0.1 * reversal_votes
    elif continuation_votes >= 3:
        verdict = PullbackVerdict.LIKELY_CONTINUATION
        confidence = continuation_votes / 4.0
    elif reversal_votes >= 3:
        verdict = PullbackVerdict.LIKELY_REVERSAL
        confidence = reversal_votes / 4.0
    else:
        verdict = PullbackVerdict.UNCERTAIN
        confidence = 0.5

    return PullbackClassification(
        verdict=verdict,
        confidence=min(confidence, 1.0),
        reasoning=reasoning,
        retracement_pct=retracement_pct,
        volume_ratio=volume_ratio,
        structure_broken=structure_broken,
    )
