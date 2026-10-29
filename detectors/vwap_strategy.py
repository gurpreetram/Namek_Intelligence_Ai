"""
VWAP (Volume Weighted Average Price) reclaim/reject strategy: enters LONG
when price crosses back above VWAP after trading below it, confirmed by
above-average volume on the crossing bar; mirrored for SHORT (price
crosses back below VWAP after trading above it). VWAP is the standard
institutional intraday benchmark -- price above it suggests buyers are in
control relative to the day's volume-weighted average fill price, below
suggests sellers are.

Single-bar confirmation (deliberately NOT the 2-step candidate-then-
follow-through-bar pattern detectors/reversal_strategy.py uses): the
"confirmation" here is the crossing bar's OWN volume being above its
recent average, not a separate bar after the cross. A cross with
unremarkable volume is treated as noise (VWAP gets touched/crossed
constantly on ordinary wiggle), not a signal.

Stop-loss: the recent lookback window's low (for a long) / high (for a
short) -- the extreme reached while price was on the wrong side of VWAP --
widened to an ATR floor if that raw distance is too tight. Same
min_stop_distance_atr_mult convention as validated_breakout_strategy.py
and reversal_strategy.py, reusing average_true_range() as-is (no
reimplementation). Target: entry +/- risk_reward_multiple * risk,
externalized as a constructor parameter (same convention as
ValidatedBreakout, default 2.0) rather than hardcoded, so
backtest/engine.py's gap-risk guard -- which reads risk_reward_multiple
via getattr(self.strategy, "risk_reward_multiple", 2.0) -- stays correct
for whatever multiple this runs with, with zero extra code needed here.

No liquidity check inside evaluate() -- deliberately matches every other
strategy in this project: liquidity gating happens ONLY in
detectors/scanner.py's scan_universe(), never inside a Strategy itself.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import Signal, SignalDirection, Strategy
from detectors.support_resistance import average_true_range
from detectors.vwap import compute_vwap


class VWAPStrategy(Strategy):
    name = "vwap_reclaim"

    def __init__(
        self,
        volume_lookback_bars: int = 20,
        volume_multiplier: float = 1.3,
        stop_lookback_bars: int = 15,
        min_stop_distance_atr_mult: float = 0.5,
        risk_reward_multiple: float = 2.0,
        confidence: float = 0.5,
    ):
        self.volume_lookback_bars = volume_lookback_bars
        self.volume_multiplier = volume_multiplier
        self.stop_lookback_bars = stop_lookback_bars
        self.min_stop_distance_atr_mult = min_stop_distance_atr_mult
        self.risk_reward_multiple = risk_reward_multiple
        self.confidence = confidence

        self._traded_today: set[str] = set()
        # Caches the first bar of the trading day per symbol/date, same
        # speed pattern as reversal_strategy.py/trend_continuation_strategy.py
        # -- avoids repeated expensive df.index.date == trade_date full-
        # index scans; subsequent bars of the same day reuse a cheap
        # timestamp-range slice giving the exact same rows.
        self._day_start_cache: dict[str, pd.Timestamp] = {}

    def evaluate(
        self, symbol: str, multi_tf_data: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> Optional[Signal]:
        trade_date = current_index.date()
        day_key = f"{symbol}_{trade_date}"
        if day_key in self._traded_today:
            return None

        df = multi_tf_data[1]
        if day_key not in self._day_start_cache:
            today_slice = df[df.index.date == trade_date]
            if today_slice.empty:
                return None
            self._day_start_cache[day_key] = today_slice.index[0]
            df = today_slice
        else:
            df = df.loc[self._day_start_cache[day_key]:]

        min_bars_needed = max(self.volume_lookback_bars, self.stop_lookback_bars) + 2
        if len(df) < min_bars_needed:
            return None

        vwap = compute_vwap(df)
        current_bar, prev_bar = df.iloc[-1], df.iloc[-2]
        current_close, prev_close = float(current_bar["close"]), float(prev_bar["close"])
        current_vwap, prev_vwap = float(vwap.iloc[-1]), float(vwap.iloc[-2])

        avg_volume = float(df["volume"].iloc[-(self.volume_lookback_bars + 1):-1].mean())
        current_volume = float(current_bar["volume"])
        volume_confirmed = avg_volume > 0 and current_volume >= self.volume_multiplier * avg_volume

        crossed_above = prev_close <= prev_vwap and current_close > current_vwap
        crossed_below = prev_close >= prev_vwap and current_close < current_vwap

        if crossed_above and volume_confirmed:
            direction = SignalDirection.LONG
        elif crossed_below and volume_confirmed:
            direction = SignalDirection.SHORT
        else:
            return None

        # Bounded-window ATR: rolling(14) only ever depends on the last
        # ~14-15 rows, so slicing to the last 30 is a pure speed win, not
        # a behavior change -- same pattern as every other strategy here.
        atr = average_true_range(df.iloc[-30:])
        min_distance = atr * self.min_stop_distance_atr_mult
        recent = df.iloc[-self.stop_lookback_bars:]

        if direction == SignalDirection.LONG:
            stop_loss = float(recent["low"].min())
            risk = current_close - stop_loss
            widened = risk < min_distance
            if widened:
                stop_loss = current_close - min_distance
                risk = min_distance
            if risk <= 0:
                return None
            target = current_close + self.risk_reward_multiple * risk
        else:
            stop_loss = float(recent["high"].max())
            risk = stop_loss - current_close
            widened = risk < min_distance
            if widened:
                stop_loss = current_close + min_distance
                risk = min_distance
            if risk <= 0:
                return None
            target = current_close - self.risk_reward_multiple * risk

        self._traded_today.add(day_key)
        stop_ref = "low" if direction == SignalDirection.LONG else "high"
        return Signal(
            direction=direction,
            confidence=self.confidence,
            stop_loss=stop_loss,
            target=target,
            trigger_level=current_vwap,
            reasoning=[
                f"Price crossed back {'above' if direction == SignalDirection.LONG else 'below'} "
                f"VWAP ({current_vwap:.2f}): {prev_close:.2f} -> {current_close:.2f}",
                f"Volume confirmed: {current_volume:.0f} vs {avg_volume:.0f} avg "
                f"({current_volume / avg_volume:.1f}x)",
                f"Stop at {stop_loss:.2f} ({self.stop_lookback_bars}-bar {stop_ref}"
                + (", widened to ATR floor" if widened else "") + ")",
            ],
        )

    def check_invalidation(
        self, trade: Trade, multi_tf_data: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> Optional[list[str]]:
        """
        Early-exit hook (see paper_trading/engine.py's on_new_bar()) --
        called on every bar while this strategy has an OPEN position,
        BEFORE the hard stop-loss/target check. Not part of the Strategy
        ABC -- called via getattr() duck-typing.

        Invalidation for a VWAP reclaim/reject trade: price re-crosses
        back to the wrong side of VWAP -- the reclaim/reject that
        justified entry has been given back. No new instance state
        needed: VWAP is recomputed fresh on the day's bars so far (same
        compute_vwap() used at entry) and compared against trade.direction.
        """
        df = multi_tf_data.get(1)
        if df is None or df.empty:
            return None
        day_df = df[df.index.date == trade.entry_time.date()]
        day_df = day_df.loc[:current_index]
        if len(day_df) < 2:
            return None

        vwap = compute_vwap(day_df)
        current_close = float(day_df["close"].iloc[-1])
        current_vwap = float(vwap.iloc[-1])

        if trade.direction == SignalDirection.LONG and current_close < current_vwap:
            return [f"Price closed back below VWAP ({current_vwap:.2f}) after the reclaim that triggered entry -- setup invalidated"]
        if trade.direction == SignalDirection.SHORT and current_close > current_vwap:
            return [f"Price closed back above VWAP ({current_vwap:.2f}) after the reject that triggered entry -- setup invalidated"]
        return None
