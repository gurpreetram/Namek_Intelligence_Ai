"""
Opening Range Breakout (ORB) -- a well-documented, widely used intraday
setup, used here to validate the backtest engine end-to-end before you
build your actual detectors (breakout validator, fakeout detector) on top
of it. This is deliberately simple: if the engine can't produce sane,
correct metrics for a strategy this well-understood, it can't be trusted
with anything more complex.

Logic:
  - Define the opening range as the high/low of the first N minutes after
    market open (default 15 min: 09:15-09:30).
  - Once the range is established, a break above the range high with
    above-average volume is a long signal; a break below the range low
    with above-average volume is a short signal.
  - Stop-loss at the opposite side of the opening range. Target at 2x the
    range width (a fixed 1:2 risk-reward, common ORB convention).
  - One signal per symbol per day -- no re-entry after a stop-out, which
    directly targets your "overtrading" loss pattern.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import Signal, SignalDirection, Strategy


class OpeningRangeBreakout(Strategy):
    name = "opening_range_breakout"

    def __init__(self, range_minutes: int = 15, volume_multiplier: float = 1.5):
        self.range_minutes = range_minutes
        self.volume_multiplier = volume_multiplier
        self._traded_today: set[str] = set()  # "{symbol}_{date}" -- one shot per day
        self._range_cache: dict[str, tuple[float, float]] = {}  # "{symbol}_{date}" -> (high, low)
        self._day_start_cache: dict[str, pd.Timestamp] = {}  # "{symbol}_{date}" -> first bar of that day

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

        # df.index.date forces a full conversion of the (ever-growing)
        # DatetimeIndex to a Python object array of date objects -- expensive,
        # and wasteful to redo on every single bar of the same day. Do it
        # once when a new day starts, then reuse a cheap timestamp-range
        # slice (df.loc[day_start:]) for every other bar of that day --
        # identical rows, since df is already truncated to current_index.
        if day_key not in self._day_start_cache:
            today_df = df[df.index.date == trade_date]
            if today_df.empty:
                return None
            self._day_start_cache[day_key] = today_df.index[0]
        else:
            today_df = df.loc[self._day_start_cache[day_key]:]

        market_open = self._day_start_cache[day_key]
        range_end = market_open + pd.Timedelta(minutes=self.range_minutes)

        # Not enough bars yet to have established the opening range.
        if current_index < range_end:
            return None

        if day_key not in self._range_cache:
            range_bars = today_df[today_df.index < range_end]
            if range_bars.empty:
                return None
            self._range_cache[day_key] = (
                float(range_bars["high"].max()),
                float(range_bars["low"].min()),
            )

        range_high, range_low = self._range_cache[day_key]
        range_width = range_high - range_low
        if range_width <= 0:
            return None

        current_bar = today_df.loc[current_index]
        post_range = today_df[(today_df.index > range_end) & (today_df.index <= current_index)]
        if post_range.empty:
            return None

        avg_volume = today_df[today_df.index < range_end]["volume"].mean()
        volume_ok = current_bar["volume"] >= avg_volume * self.volume_multiplier

        close = float(current_bar["close"])

        if close > range_high and volume_ok:
            self._traded_today.add(day_key)
            return Signal(
                direction=SignalDirection.LONG,
                confidence=0.5,  # placeholder single-factor confidence -- a
                # real confidence model needs calibration against backtest
                # results, not an asserted number. Treat this as "signal
                # fired", not "50% probability" until calibrated.
                stop_loss=range_low,
                target=close + 2 * range_width,
                trigger_level=range_high,
                reasoning=[
                    f"Price {close:.2f} broke above opening range high {range_high:.2f}",
                    f"Volume {current_bar['volume']:.0f} exceeded "
                    f"{self.volume_multiplier}x opening-range average ({avg_volume:.0f})",
                    f"Opening range: {range_low:.2f}-{range_high:.2f} "
                    f"({self.range_minutes}min from market open)",
                ],
            )

        if close < range_low and volume_ok:
            self._traded_today.add(day_key)
            return Signal(
                direction=SignalDirection.SHORT,
                confidence=0.5,
                stop_loss=range_high,
                target=close - 2 * range_width,
                trigger_level=range_low,
                reasoning=[
                    f"Price {close:.2f} broke below opening range low {range_low:.2f}",
                    f"Volume {current_bar['volume']:.0f} exceeded "
                    f"{self.volume_multiplier}x opening-range average ({avg_volume:.0f})",
                    f"Opening range: {range_low:.2f}-{range_high:.2f} "
                    f"({self.range_minutes}min from market open)",
                ],
            )

        return None

    def check_invalidation(
        self, trade: Trade, multi_tf_data: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> Optional[list[str]]:
        """
        Early-exit hook (see paper_trading/engine.py's on_new_bar()) --
        called on every bar while this strategy has an OPEN position, in
        addition to (and checked BEFORE) the hard stop-loss/target. Not
        part of the Strategy ABC -- called via getattr() duck-typing, so
        strategies without this method simply skip early-exit checks.

        Invalidation for a breakout trade: price closes back through the
        SAME level that was broken to enter (range_high for a long,
        range_low for a short) -- the breakout that justified the trade
        has failed, even though price hasn't reached the stop-loss yet
        (stop-loss sits at the OPPOSITE side of the range, further away).
        Reuses self._range_cache (already populated at signal time, keyed
        the same way _traded_today is) rather than recomputing anything.
        """
        day_key = f"{trade.symbol}_{trade.entry_time.date()}"
        cached = self._range_cache.get(day_key)
        if cached is None:
            return None
        range_high, range_low = cached

        df = multi_tf_data.get(1)
        if df is None or df.empty:
            return None
        current_close = float(df.loc[:current_index]["close"].iloc[-1])

        if trade.direction == SignalDirection.LONG and current_close < range_high:
            return [f"Price closed back below the opening range high ({range_high:.2f}) it broke above to enter -- breakout failed"]
        if trade.direction == SignalDirection.SHORT and current_close > range_low:
            return [f"Price closed back above the opening range low ({range_low:.2f}) it broke below to enter -- breakout failed"]
        return None
