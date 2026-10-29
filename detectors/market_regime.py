"""
Classifies the current regime of a symbol (or index) as trending up,
trending down, or range-bound/choppy -- using ADX (Average Directional
Index), a well-established trend-strength indicator from Wilder's "New
Concepts in Technical Trading Systems" (1978), still standard across
technical analysis today.

Why this matters beyond just "another indicator": regime is context that
should GATE the other detectors, not just add another opinion alongside
them. A trend-continuation setup means something different in a genuinely
trending market versus a choppy one -- in fact this is a concrete way to
reduce the false-positive problem we already found in the pullback
classifier (recall CHOPPY2 slipping through as a "valid" continuation
setup on pure noise). If the regime detector says the market is NOT
trending, a continuation signal firing anyway is much more likely to be
exactly that kind of statistical false positive, and can be filtered out.

ADX mechanics (standard Wilder's calculation):
  - +DM (directional movement up) / -DM (directional movement down) per bar
  - Smoothed via Wilder's method (not a simple moving average -- this
    specific smoothing is what makes it "ADX" rather than an ad-hoc metric)
  - +DI / -DI = smoothed DM relative to smoothed True Range
  - DX = |+DI - -DI| / (+DI + -DI) * 100
  - ADX = Wilder-smoothed DX -- this is the actual trend STRENGTH value,
    regardless of direction. +DI vs -DI tells you the direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Regime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"


@dataclass
class RegimeReading:
    regime: Regime
    adx: float
    plus_di: float
    minus_di: float
    reasoning: list[str]


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing -- NOT a simple moving average. First value is a
    plain sum/average over `period`, every value after is
    prev - (prev / period) + current. This specific recursion is what the
    original ADX formula requires; a plain SMA substitute changes the
    indicator's behavior meaningfully, especially early in a series."""
    result = series.copy().astype(float)
    result.iloc[:period] = np.nan
    if len(series) <= period:
        return result

    first_val = series.iloc[1:period + 1].sum()
    smoothed = [first_val]
    for i in range(period + 1, len(series)):
        prev = smoothed[-1]
        smoothed.append(prev - (prev / period) + series.iloc[i])

    result.iloc[period:period + len(smoothed)] = smoothed
    return result


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [plus_di, minus_di, adx] aligned to
    df's index. Requires at least ~3x period bars for the ADX value itself
    to be meaningful (Wilder's smoothing needs a warm-up period, and ADX
    is itself a smoothed average of DX).
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)

    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()  # first ADX value is a simple average of DX;
    # subsequent values are technically Wilder-smoothed too in the strict
    # original formula, but a rolling mean is the widely-used practical
    # approximation and is what most charting platforms actually display.

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def detect_regime(
    df_15min: pd.DataFrame,
    adx_period: int = 14,
    trending_threshold: float = 25.0,
) -> RegimeReading | None:
    """
    df_15min: OHLCV data on the 15-minute timeframe -- ADX on 1-minute data
        is too noisy to represent a meaningful "regime" (regime is a
        slower-changing property than individual entries); 15-min data
        gives a market-structure view the entry-timing detectors don't
        need to duplicate. Needs enough bars for ADX to stabilize --
        returns None if there isn't enough history yet.
    trending_threshold: the standard ADX convention -- above 25 is
        generally considered "trending," below is "range-bound/weak
        trend." This threshold is a widely used industry convention (not
        this project's invention), though like any single threshold it's
        a simplification of a continuous strength measure.
    """
    if len(df_15min) < adx_period * 3:
        return None

    adx_df = compute_adx(df_15min, period=adx_period)
    latest = adx_df.iloc[-1]

    if pd.isna(latest["adx"]):
        return None

    adx_val = float(latest["adx"])
    plus_di = float(latest["plus_di"])
    minus_di = float(latest["minus_di"])

    if adx_val < trending_threshold:
        regime = Regime.RANGE_BOUND
        reasoning = [
            f"ADX {adx_val:.1f} below {trending_threshold} threshold -- "
            f"no strong directional trend, treat breakouts/pullbacks here "
            f"with extra skepticism",
        ]
    elif plus_di > minus_di:
        regime = Regime.TRENDING_UP
        reasoning = [
            f"ADX {adx_val:.1f} above {trending_threshold} -- trend present",
            f"+DI ({plus_di:.1f}) > -DI ({minus_di:.1f}) -- upward direction",
        ]
    else:
        regime = Regime.TRENDING_DOWN
        reasoning = [
            f"ADX {adx_val:.1f} above {trending_threshold} -- trend present",
            f"-DI ({minus_di:.1f}) > +DI ({plus_di:.1f}) -- downward direction",
        ]

    return RegimeReading(regime=regime, adx=adx_val, plus_di=plus_di,
                          minus_di=minus_di, reasoning=reasoning)
