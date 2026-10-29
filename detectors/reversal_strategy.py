"""
Reversal Detection strategy -- trades the LIKELY_REVERSAL verdict that
pullback_classifier.py already produces but nothing acted on until now
(trend_continuation_strategy.py only trades the LIKELY_CONTINUATION
verdict). This directly reuses the classifier's strongest signal for
reversal: a structure break (price taking out the trend leg's origin
point), which pullback_classifier.py already weights as the dominant
factor.

Why this needs MORE caution than the continuation strategy, not less:
a reversal trade means betting AGAINST the direction that was just
dominant. That's a fundamentally higher-risk bet than continuation
(which bets WITH established direction), so this strategy adds an extra
layer the continuation strategy doesn't need:

  - Confirmation bar: after the pullback bar shows reversal
    characteristics (structure broken), we wait for the NEXT bar to
    extend further in the reversal direction before entering -- same
    "don't trust it until it follows through" principle as
    breakout_validator.py. A structure break that immediately snaps
    back is exactly the kind of false reversal your own notes describe
    ("touched support, rapidly pulled back with big candle... in loss").
  - Stop-loss is placed beyond the ORIGINAL trend leg's extreme (the
    swing point the structure break took out), not just the pullback
    bar's high/low -- if price reclaims that level, the reversal thesis
    is specifically invalidated, not just "stopped out at an arbitrary
    point."

This is deliberately a stricter, lower-frequency strategy than trend
continuation. Expect fewer signals -- that's the design intent, not a bug.

DELIBERATELY NO check_invalidation() (see paper_trading/engine.py's
early-exit warning system, _check_early_exit()): for every other
strategy, "the price that would invalidate this setup" is a DIFFERENT,
usually nearer, level than the hard stop-loss -- which is exactly what
makes an early-exit check add value there. Here, they're the SAME price
by construction: stop_loss IS set to "the ORIGINAL trend leg's extreme"
above, precisely because reclaiming it invalidates the reversal thesis --
see that line. A close-based invalidation check on that identical level
would never actually fire first: PaperTradingEngine's hard stop-loss
check uses intrabar high/low (not close), so it always triggers on the
same or an earlier bar than any close-based check could. Adding one here
would be dead code, not a missing feature.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.strategy_base import Signal, SignalDirection, Strategy
from detectors.pullback_classifier import (
    PullbackVerdict,
    TrendDirection,
    classify_pullback,
    find_recent_trend_leg,
)
from detectors.support_resistance import average_true_range


class ReversalDetection(Strategy):
    name = "reversal_detection"

    def __init__(
        self,
        trend_lookback_bars: int = 20,
        pullback_bars: int = 2,
        min_confidence: float = 0.6,
        min_stop_distance_atr_mult: float = 0.5,
    ):
        self.trend_lookback_bars = trend_lookback_bars
        self.pullback_bars = pullback_bars
        self.min_confidence = min_confidence
        self.min_stop_distance_atr_mult = min_stop_distance_atr_mult

        self._traded_today: set[str] = set()
        # Tracks a reversal candidate awaiting its confirmation bar:
        # {symbol: (reversal_direction, trend_leg_extreme_price, candidate_time)}
        self._pending_confirmation: dict[str, tuple] = {}
        self._day_start_cache: dict[str, pd.Timestamp] = {}  # "{symbol}_{date}" -> first bar of that day

    def evaluate(
        self,
        symbol: str,
        multi_tf_data: dict[int, pd.DataFrame],
        current_index: pd.Timestamp,
    ) -> Optional[Signal]:
        trade_date = current_index.date()
        day_key = f"{symbol}_{trade_date}"
        if day_key in self._traded_today:
            return None

        df = multi_tf_data[1]
        # same cross-day fix as trend_continuation_strategy.py. df.index.date
        # forces an expensive full-index date conversion -- do that once per
        # day (cached below) instead of once per bar; every other bar of the
        # same day reuses a cheap timestamp-range slice giving the exact
        # same rows, since df is already truncated to current_index.
        if day_key not in self._day_start_cache:
            today_slice = df[df.index.date == trade_date]
            if today_slice.empty:
                return None
            self._day_start_cache[day_key] = today_slice.index[0]
            df = today_slice
        else:
            df = df.loc[self._day_start_cache[day_key]:]

        if len(df) < self.trend_lookback_bars + self.pullback_bars + 5:
            return None

        # --- Step 1: is there a pending reversal candidate awaiting confirmation? ---
        if symbol in self._pending_confirmation:
            reversal_direction, invalidation_price, candidate_time = self._pending_confirmation.pop(symbol)
            candidate_close = float(df.loc[candidate_time, "close"]) if candidate_time in df.index else None
            current_close = float(df["close"].iloc[-1])

            if candidate_close is None:
                return None

            follow_through = (
                (reversal_direction == SignalDirection.SHORT and current_close < candidate_close)
                or (reversal_direction == SignalDirection.LONG and current_close > candidate_close)
            )
            if not follow_through:
                return None  # reversal snapped back -- exactly the false-reversal pattern to avoid

            # df is already day-bounded (see the cross-day fix above), but
            # ATR's rolling(14) only needs the last ~15 rows -- slicing
            # further is a pure speed win, not a behavior change.
            atr = average_true_range(df.iloc[-30:])
            min_distance = atr * self.min_stop_distance_atr_mult
            risk = abs(invalidation_price - current_close)
            if risk < min_distance:
                risk = min_distance
                invalidation_price = (
                    current_close + min_distance if reversal_direction == SignalDirection.SHORT
                    else current_close - min_distance
                )
            if risk <= 0:
                return None

            target = (
                current_close - 2 * risk if reversal_direction == SignalDirection.SHORT
                else current_close + 2 * risk
            )

            self._traded_today.add(day_key)
            return Signal(
                direction=reversal_direction,
                confidence=self.min_confidence,  # base confidence; classifier's own
                # confidence was already checked before this candidate was accepted
                stop_loss=invalidation_price,
                target=target,
                reasoning=[
                    f"Reversal confirmed: follow-through bar closed at {current_close:.2f}, "
                    f"continuing past candidate bar's {candidate_close:.2f}",
                    f"Stop placed beyond original trend leg's extreme "
                    f"({invalidation_price:.2f}) -- reclaiming that level invalidates "
                    f"the reversal thesis specifically, not just an arbitrary stop",
                ],
            )

        # --- Step 2: look for a fresh reversal candidate ---
        pre_pullback = df.iloc[:-self.pullback_bars] if self.pullback_bars > 0 else df
        trend_leg = find_recent_trend_leg(pre_pullback, lookback_bars=self.trend_lookback_bars)
        if trend_leg is None:
            return None

        classification = classify_pullback(df, trend_leg, pullback_bars=self.pullback_bars)
        if classification is None:
            return None
        if classification.verdict != PullbackVerdict.LIKELY_REVERSAL:
            return None
        if classification.confidence < self.min_confidence:
            return None
        if not classification.structure_broken:
            return None  # structure break is the load-bearing factor here -- require it explicitly

        # Reversal direction is OPPOSITE the original trend leg's direction.
        reversal_direction = (
            SignalDirection.SHORT if trend_leg.direction == TrendDirection.UP
            else SignalDirection.LONG
        )
        # Invalidation point: the trend leg's origin (the swing point that
        # got taken out to constitute the structure break).
        invalidation_price = trend_leg.start_price

        self._pending_confirmation[symbol] = (reversal_direction, invalidation_price, current_index)
        return None
