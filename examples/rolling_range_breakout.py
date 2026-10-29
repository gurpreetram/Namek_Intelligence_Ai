"""
Rolling Range Breakout -- a NEW, ADDITIONAL signal type alongside (not a
replacement for) examples/opening_range_breakout.py's original Opening
Range Breakout. That strategy's range is fixed once, from the first
`range_minutes` after market open (09:15-09:30 by default), and never
recomputed for the rest of the day -- a stock that spends the morning
quiet and then consolidates and breaks out at 1 PM has no opening-range
event to fire on at all, that entire setup is invisible to ORB by
construction, not by any bug in it.

This strategy fires on the SAME mechanics (close breaks beyond a range
with above-average volume, stop at the opposite side, target at 2x the
range width, one trade per symbol per day -- same anti-overtrading
convention every strategy in this project uses), but the range itself is
a ROLLING trailing window (default 30 minutes) recomputed on every bar
from the `range_minutes` immediately preceding it, NOT anchored to market
open. That's the one, deliberate difference: this catches a breakout at
ANY point in the trading day that follows a genuine `range_minutes` of
consolidation, not just the opening range specifically.

examples/opening_range_breakout.py itself is left completely untouched --
this is a separate file/class so ORB's own backtested history stays
exactly reproducible; anywhere both are wanted (live scanning, the
dashboard, /backtest), they're registered as two distinct, independently
selectable strategies, never as a replacement of one by the other.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import Signal, SignalDirection, Strategy


class RollingRangeBreakout(Strategy):
    name = "rolling_range_breakout"

    def __init__(self, range_minutes: int = 30, volume_multiplier: float = 1.5):
        self.range_minutes = range_minutes
        self.volume_multiplier = volume_multiplier
        self._traded_today: set[str] = set()  # "{symbol}_{date}" -- one shot per day, same as every other strategy here
        self._day_start_cache: dict[str, pd.Timestamp] = {}  # "{symbol}_{date}" -> first bar of that day
        # The specific rolling range that was actually broken to enter,
        # fixed at signal time -- needed by check_invalidation() below,
        # which must keep checking against THAT range, not whatever the
        # (continuously moving) rolling window looks like on a later bar.
        # Same role as ORB's _range_cache, just captured at entry instead
        # of once per day (there IS no single "the" range here otherwise).
        self._entry_range: dict[str, tuple[float, float]] = {}

    def evaluate(
        self,
        symbol: str,
        multi_tf_data: dict[int, pd.DataFrame],
        current_index: pd.Timestamp,
    ) -> Optional[Signal]:
        df = multi_tf_data[1]
        if df.empty:
            return None

        trade_date = current_index.date()
        day_key = f"{symbol}_{trade_date}"

        if day_key in self._traded_today:
            return None

        # Same day-boundary caching pattern as every other strategy here
        # (avoids a full df.index.date scan on every single bar).
        if day_key not in self._day_start_cache:
            today_df = df[df.index.date == trade_date]
            if today_df.empty:
                return None
            self._day_start_cache[day_key] = today_df.index[0]
        else:
            today_df = df.loc[self._day_start_cache[day_key]:]

        window_start = current_index - pd.Timedelta(minutes=self.range_minutes)
        # The range is the range_minutes STRICTLY BEFORE this bar -- the
        # current bar's own high/low must never be part of the range it's
        # being tested against, or every bar would trivially "break out"
        # of a range that already includes its own extreme.
        range_bars = today_df.loc[window_start:current_index].iloc[:-1]
        if range_bars.empty or len(range_bars) < self.range_minutes:
            return None  # not enough of today's own history yet to form a full window

        range_high = float(range_bars["high"].max())
        range_low = float(range_bars["low"].min())
        range_width = range_high - range_low
        if range_width <= 0:
            return None

        current_bar = today_df.loc[current_index]
        avg_volume = range_bars["volume"].mean()
        volume_ok = current_bar["volume"] >= avg_volume * self.volume_multiplier

        close = float(current_bar["close"])

        if close > range_high and volume_ok:
            self._traded_today.add(day_key)
            self._entry_range[day_key] = (range_high, range_low)
            return Signal(
                direction=SignalDirection.LONG,
                confidence=0.5,  # same placeholder-confidence convention as ORB -- see that file's note
                stop_loss=range_low,
                target=close + 2 * range_width,
                trigger_level=range_high,
                reasoning=[
                    f"Price {close:.2f} broke above the trailing {self.range_minutes}-min range high {range_high:.2f}",
                    f"Volume {current_bar['volume']:.0f} exceeded "
                    f"{self.volume_multiplier}x the trailing-range average ({avg_volume:.0f})",
                    f"Rolling range: {range_low:.2f}-{range_high:.2f} "
                    f"(trailing {self.range_minutes}min, {window_start.strftime('%H:%M')}-{current_index.strftime('%H:%M')})",
                ],
            )

        if close < range_low and volume_ok:
            self._traded_today.add(day_key)
            self._entry_range[day_key] = (range_high, range_low)
            return Signal(
                direction=SignalDirection.SHORT,
                confidence=0.5,
                stop_loss=range_high,
                target=close - 2 * range_width,
                trigger_level=range_low,
                reasoning=[
                    f"Price {close:.2f} broke below the trailing {self.range_minutes}-min range low {range_low:.2f}",
                    f"Volume {current_bar['volume']:.0f} exceeded "
                    f"{self.volume_multiplier}x the trailing-range average ({avg_volume:.0f})",
                    f"Rolling range: {range_low:.2f}-{range_high:.2f} "
                    f"(trailing {self.range_minutes}min, {window_start.strftime('%H:%M')}-{current_index.strftime('%H:%M')})",
                ],
            )

        return None

    def check_invalidation(
        self, trade: Trade, multi_tf_data: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> Optional[list[str]]:
        """
        Early-exit hook (see paper_trading/engine.py's on_new_bar()) --
        same invalidation condition as ORB: price closes back through the
        SAME range edge that was broken to enter, checked against the
        range CAPTURED AT ENTRY TIME (self._entry_range), not a freshly
        recomputed rolling window (which would have moved on by now and
        no longer describes what was actually broken).
        """
        day_key = f"{trade.symbol}_{trade.entry_time.date()}"
        cached = self._entry_range.get(day_key)
        if cached is None:
            return None
        range_high, range_low = cached

        df = multi_tf_data.get(1)
        if df is None or df.empty:
            return None
        current_close = float(df.loc[:current_index]["close"].iloc[-1])

        if trade.direction == SignalDirection.LONG and current_close < range_high:
            return [f"Price closed back below the trailing-range high ({range_high:.2f}) it broke above to enter -- breakout failed"]
        if trade.direction == SignalDirection.SHORT and current_close > range_low:
            return [f"Price closed back above the trailing-range low ({range_low:.2f}) it broke below to enter -- breakout failed"]
        return None
