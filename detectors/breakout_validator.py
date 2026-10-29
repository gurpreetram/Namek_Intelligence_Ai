"""
Scores whether a break of a support/resistance level is likely genuine
or likely to fail -- this is the piece that directly targets your notes:
"sometimes once it touched the support/resistance it rapidly pull back
with big candle and make profit trade in loss."

Research grounding (why these specific factors, not arbitrary ones):
  - Close Location Value (CLV): where the close sits within the bar's
    range. A breakout candle that closes near its high (for a bullish
    break) shows buyers held control through the bar. A breakout that
    closes mid-range or with a long upper wick shows sellers already
    fought back within the same candle -- a classic fakeout precursor.
    This is a standard price-action concept (related to Larry Williams'
    and Marc Chaikin's close-location-based indicators).
  - Volume surge: a genuine break of a level that other participants
    respect should show above-average participation. A break on
    below-average volume means few traders actually committed to the
    new direction -- fragile, easily reversed.
  - Follow-through: the single highest-value check. A real breakout
    should see the NEXT bar continue away from the level, not
    immediately snap back through it. This is checked with the
    mandatory 1-bar lag the backtest engine already enforces (no
    lookahead) -- we are never claiming to know a breakout is valid
    on the same bar it happens, only on confirmation.
  - Distance beyond level (in ATR terms): a close that's barely a few
    ticks beyond the level is statistically closer to noise than a
    close that's meaningfully cleared it.

None of these factors alone is reliable -- that's exactly the "very few
setups go the same direction" problem in your notes. The validator scores
them together and only calls a breakout "confirmed" when multiple factors
agree, which by design means it will validate fewer breakouts than a naive
"price crossed the level" check -- that's the point.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from detectors.support_resistance import Level, average_true_range


class BreakoutDirection(Enum):
    BULLISH = "bullish"  # broke above resistance
    BEARISH = "bearish"  # broke below support


@dataclass
class BreakoutValidation:
    is_valid: bool
    direction: BreakoutDirection
    level: Level
    confidence: float  # 0-1, weighted score across all factors
    reasoning: list[str]
    clv: float
    volume_zscore: float
    distance_atr: float
    follow_through_confirmed: bool


def _close_location_value(bar: pd.Series) -> float:
    """Range -1 to +1. +1 = closed at the bar's high (max buying pressure
    held to the close). -1 = closed at the bar's low."""
    high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
    bar_range = high - low
    if bar_range == 0:
        return 0.0
    return ((close - low) - (high - close)) / bar_range


def volume_zscore(current_volume: float, recent_volumes: pd.Series) -> float:
    mean, std = recent_volumes.mean(), recent_volumes.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((current_volume - mean) / std)


def validate_breakout(
    df_1min: pd.DataFrame,
    level: Level,
    breakout_time: pd.Timestamp,
    confirmation_time: pd.Timestamp,
    volume_lookback: int = 20,
    min_volume_zscore: float = 0.5,
) -> BreakoutValidation | None:
    """
    df_1min: full 1-minute data available up to and including confirmation_time
        (caller is responsible for the no-lookahead slice -- this function
        does not truncate anything itself).
    level: the S/R level being broken.
    breakout_time: the bar where price first closed beyond the level.
    confirmation_time: the NEXT bar after breakout_time -- this function
        checks whether that bar followed through. Must be strictly after
        breakout_time and present in df_1min.
    min_volume_zscore: threshold the breakout bar's volume_zscore must
        clear for the `volume_ok` factor below -- default 0.5 is the
        original, always-live value. Raised (e.g. to 1.5) by
        examples/run_strategy_optimization.py's ValidatedBreakout
        volume-filter variant to test whether a more selective, higher-
        conviction volume requirement improves results; not a live change.

    Returns None if there isn't enough data to evaluate (e.g. confirmation
    bar not yet available, or insufficient volume history).
    """
    if breakout_time not in df_1min.index or confirmation_time not in df_1min.index:
        return None

    breakout_bar = df_1min.loc[breakout_time]
    confirmation_bar = df_1min.loc[confirmation_time]

    close = float(breakout_bar["close"])
    direction = BreakoutDirection.BULLISH if close > level.price else BreakoutDirection.BEARISH

    history_up_to_breakout = df_1min.loc[:breakout_time]
    if len(history_up_to_breakout) < volume_lookback + 1:
        return None

    recent_volumes = history_up_to_breakout["volume"].iloc[-(volume_lookback + 1):-1]
    vol_z = volume_zscore(float(breakout_bar["volume"]), recent_volumes)
    clv = _close_location_value(breakout_bar)
    # See support_resistance.py's find_levels() for why slicing to a small
    # recent tail here doesn't change the ATR value -- rolling(14) only
    # ever looks at the last ~15 rows regardless of how much history
    # precedes them, and history_up_to_breakout grows across the whole
    # backtest, so passing it in full recomputes over ever-more bars for
    # an identical answer.
    atr = average_true_range(history_up_to_breakout.iloc[-30:])
    distance_atr = abs(close - level.price) / atr if atr > 0 else 0.0

    confirmation_close = float(confirmation_bar["close"])
    if direction == BreakoutDirection.BULLISH:
        follow_through = confirmation_close > close
        clv_ok = clv > 0.3  # closed in upper 65% of the bar's range
    else:
        follow_through = confirmation_close < close
        clv_ok = clv < -0.3

    volume_ok = vol_z > min_volume_zscore  # meaningfully above recent average, not just noise
    distance_ok = distance_atr > 0.15  # cleared the level by a real margin, not a graze

    factors_passed = sum([follow_through, clv_ok, volume_ok, distance_ok])
    confidence = factors_passed / 4.0

    reasoning = []
    reasoning.append(
        f"Level {level.price:.2f} (touched {level.touch_count}x) broken "
        f"{'above' if direction == BreakoutDirection.BULLISH else 'below'} "
        f"at {close:.2f}"
    )
    reasoning.append(
        f"{'Follow-through confirmed' if follow_through else 'NO follow-through'}: "
        f"next bar closed at {confirmation_close:.2f}"
    )
    reasoning.append(
        f"Close location {'supports' if clv_ok else 'does not support'} breakout "
        f"(CLV={clv:.2f})"
    )
    reasoning.append(
        f"Volume {'confirmed' if volume_ok else 'did not confirm'} "
        f"(z-score={vol_z:.2f} vs {volume_lookback}-bar average)"
    )
    reasoning.append(
        f"Break distance {'adequate' if distance_ok else 'marginal'} "
        f"({distance_atr:.2f}x ATR beyond level)"
    )

    # Require follow-through AND at least 2 of the other 3 factors -- the
    # highest-value check is non-negotiable, the rest are supporting evidence.
    is_valid = follow_through and (factors_passed - int(follow_through)) >= 2

    return BreakoutValidation(
        is_valid=is_valid,
        direction=direction,
        level=level,
        confidence=confidence,
        reasoning=reasoning,
        clv=clv,
        volume_zscore=vol_z,
        distance_atr=distance_atr,
        follow_through_confirmed=follow_through,
    )
