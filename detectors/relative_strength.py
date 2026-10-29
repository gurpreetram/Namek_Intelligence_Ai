"""
Relative Strength: measures whether a stock is outperforming or
underperforming a benchmark (typically NIFTY 50, or a sector index) over
a lookback window -- a well-established concept in technical analysis
(related to, though simpler than, IBD's proprietary RS Rating).

Why this matters for intraday setups specifically: a breakout in a stock
that's ALSO showing relative strength against the broader market is a
meaningfully different signal than the same breakout in a stock that's
lagging the market -- the former suggests genuine stock-specific buying
interest, the latter suggests the move might just be beta (moving because
everything is moving), which tends to fade faster when the broader market
pauses. This is additional context for the other detectors, not a
replacement for them -- RS alone says nothing about entry timing or
setup quality.

Note: this module is benchmark-agnostic. Pass NIFTY 50 data if you have
it (best choice for NSE cash intraday), or a sector index, or even
another stock in the same sector as a rough peer comparison -- the
math doesn't care what the benchmark represents, only that it's a
fair comparison point.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class RelativeStrengthVerdict(Enum):
    OUTPERFORMING = "outperforming"
    UNDERPERFORMING = "underperforming"
    IN_LINE = "in_line"


@dataclass
class RelativeStrengthReading:
    verdict: RelativeStrengthVerdict
    symbol_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    reasoning: list[str]


def compute_relative_strength(
    symbol_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback_bars: int = 20,
    in_line_threshold_pct: float = 0.15,
) -> RelativeStrengthReading | None:
    """
    symbol_df / benchmark_df: 1-minute OHLCV data, same time range,
        already truncated to "up to now" by the caller (no-lookahead is
        the caller's responsibility, same convention as every other
        detector in this project).
    lookback_bars: window over which returns are compared -- 20 bars
        (20 minutes on 1-min data) captures short-term intraday relative
        moves; lengthen this if you want a slower, less noisy read.
    in_line_threshold_pct: excess return smaller than this (in percentage
        points) is treated as "in line" rather than forcing a verdict --
        a 0.02% excess return isn't meaningfully different from the
        benchmark, it's noise.

    Returns None if either series doesn't have enough bars, or if the two
    series don't actually overlap in time (can't compare apples to
    apples if the timestamps don't align).
    """
    if len(symbol_df) < lookback_bars + 1 or len(benchmark_df) < lookback_bars + 1:
        return None

    symbol_window = symbol_df.iloc[-lookback_bars - 1:]
    benchmark_window = benchmark_df.reindex(symbol_window.index, method="ffill")

    if benchmark_window["close"].isna().any():
        return None  # benchmark doesn't actually cover this symbol's timestamps

    symbol_return = (
        (symbol_window["close"].iloc[-1] - symbol_window["close"].iloc[0])
        / symbol_window["close"].iloc[0] * 100
    )
    benchmark_return = (
        (benchmark_window["close"].iloc[-1] - benchmark_window["close"].iloc[0])
        / benchmark_window["close"].iloc[0] * 100
    )
    excess_return = symbol_return - benchmark_return

    if abs(excess_return) < in_line_threshold_pct:
        verdict = RelativeStrengthVerdict.IN_LINE
    elif excess_return > 0:
        verdict = RelativeStrengthVerdict.OUTPERFORMING
    else:
        verdict = RelativeStrengthVerdict.UNDERPERFORMING

    reasoning = [
        f"Symbol return over last {lookback_bars} bars: {symbol_return:+.2f}%",
        f"Benchmark return over same window: {benchmark_return:+.2f}%",
        f"Excess return: {excess_return:+.2f}pp -- {verdict.value}",
    ]

    return RelativeStrengthReading(
        verdict=verdict,
        symbol_return_pct=float(symbol_return),
        benchmark_return_pct=float(benchmark_return),
        excess_return_pct=float(excess_return),
        reasoning=reasoning,
    )
