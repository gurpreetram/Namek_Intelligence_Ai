"""
Fibonacci retracement levels: given a directional swing leg (a swing low
followed by a swing high for an up-leg, or a swing high followed by a
swing low for a down-leg -- see detectors/support_resistance.py's
find_swing_points() for how swing points are identified), computes the
standard retracement levels traders watch for a pullback to end and the
prior trend to resume: 23.6%, 38.2%, 50%, 61.8%, 78.6% of the leg's range.

HONEST CAVEAT (same standard as detectors/smart_money_concepts.py):
Fibonacci retracement is extremely popular in retail technical analysis,
but its predictive validity is CONTESTED in academic literature -- some
studies find retracement levels perform no better than arbitrary round
numbers or randomly chosen levels, others find weak support for specific
ratios (38.2%/61.8%) under specific conditions. The levels computed here
are correct per the standard, widely-used definition; whether price
actually respects them is an empirical question for the strategy backtest
to answer (see detectors/fibonacci_strategy.py), not something to assume
works going in.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from detectors.support_resistance import average_true_range, find_swing_points

# Standard Fibonacci retracement ratios, ascending (shallow -> deep).
RETRACEMENT_RATIOS: list[float] = [0.236, 0.382, 0.5, 0.618, 0.786]


class LegDirection(Enum):
    UP = "up"      # swing low -> swing high; a pullback retraces DOWN from the high
    DOWN = "down"  # swing high -> swing low; a pullback retraces UP from the low


@dataclass
class SwingLeg:
    direction: LegDirection
    start_time: pd.Timestamp  # the earlier swing point
    start_price: float
    end_time: pd.Timestamp    # the later (most recent) swing point -- the leg's "end"
    end_price: float


@dataclass
class FibonacciLevels:
    leg: SwingLeg
    levels: dict[float, float]  # ratio -> price, e.g. {0.382: 123.45, 0.618: 119.80, ...}

    def level(self, ratio: float) -> float:
        return self.levels[ratio]


def compute_retracement_levels(leg: SwingLeg) -> FibonacciLevels:
    """
    Standard retracement formula:
      UP leg (low -> high): level = high - (high - low) * ratio -- a
        DEEPER ratio (e.g. 0.786) sits closer to the swing low.
      DOWN leg (high -> low): level = low + (high - low) * ratio -- a
        DEEPER ratio sits closer to the swing high.
    In both cases, moving to a higher ratio in RETRACEMENT_RATIOS means a
    deeper (more adverse, closer to full invalidation) retracement.
    """
    high = max(leg.start_price, leg.end_price)
    low = min(leg.start_price, leg.end_price)
    span = high - low

    levels: dict[float, float] = {}
    for ratio in RETRACEMENT_RATIOS:
        if leg.direction == LegDirection.UP:
            levels[ratio] = high - span * ratio
        else:
            levels[ratio] = low + span * ratio

    return FibonacciLevels(leg=leg, levels=levels)


def find_most_recent_leg(
    df: pd.DataFrame, fractal_window: int = 3, min_leg_atr_mult: float = 1.5,
) -> SwingLeg | None:
    """
    Finds the most recent significant directional leg: the last two swing
    points (chronologically, across highs and lows combined) -- returns
    None if:
      - there aren't at least one swing high AND one swing low yet,
      - the last two don't alternate direction (two highs or two lows in
        a row isn't a clean leg -- it means price made a new extreme
        without a confirmed reversal in between),
      - the leg is smaller than min_leg_atr_mult x ATR (filters out
        noise-sized wiggles that technically satisfy the fractal
        swing-point definition but aren't a real trend leg worth trading
        a retracement of).
    """
    swing_highs, swing_lows = find_swing_points(df, fractal_window)
    highs = swing_highs.dropna()
    lows = swing_lows.dropna()
    if highs.empty or lows.empty:
        return None

    tagged = pd.concat([
        highs.rename("price").to_frame().assign(kind="high"),
        lows.rename("price").to_frame().assign(kind="low"),
    ]).sort_index()
    if len(tagged) < 2:
        return None

    last_two = tagged.iloc[-2:]
    if last_two["kind"].iloc[0] == last_two["kind"].iloc[1]:
        return None  # two highs or two lows in a row -- not a clean alternating leg

    start_ts, start_row = last_two.index[0], last_two.iloc[0]
    end_ts, end_row = last_two.index[1], last_two.iloc[1]
    direction = LegDirection.UP if end_row["kind"] == "high" else LegDirection.DOWN

    atr = average_true_range(df.iloc[-30:])
    if atr <= 0:
        return None
    leg_size = abs(float(end_row["price"]) - float(start_row["price"]))
    if leg_size < min_leg_atr_mult * atr:
        return None

    return SwingLeg(
        direction=direction, start_time=start_ts, start_price=float(start_row["price"]),
        end_time=end_ts, end_price=float(end_row["price"]),
    )
