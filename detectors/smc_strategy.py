"""
Smart Money Concepts (SMC) order-block retest strategy: enters when price
returns to retest a previously-identified order block zone (see
detectors/smart_money_concepts.py's find_order_blocks() -- the last
opposing candle before a decisive move) and shows rejection back in the
block's original direction on the SAME bar that touches the zone. LONG on
a bullish order block retest+reject, SHORT on a bearish one.

SCOPE DECISION, flagged explicitly per instruction: liquidity-sweep-based
entries were deliberately left OUT of this strategy. They overlap too
much with detectors/reversal_strategy.py's existing logic to count as
independent: smart_money_concepts.py's own module docstring already says
a liquidity sweep "IS a specific, named case of a failed breakout with a
sharp reversal" -- structurally the same core idea ReversalDetection
already trades (a broken structure/level that reverses), just built on a
different underlying swing-point detector (find_swing_points()'s fractal
method here vs pullback_classifier.py's trend-leg method there). Order
blocks are a genuinely distinct detection primitive (candle-body "last
opposing candle before a big move," not a swing-point/structure-break
concept at all), so this strategy trades ONLY order-block retests. A
backtest result from this strategy should be read as "does trading
order-block retests specifically have edge," NOT "does SMC in general" --
the liquidity-sweep half of SMC is untested here and would need its own
separate validation (almost certainly against ReversalDetection as the
comparison baseline, given the overlap) if ever added.

Single-bar retest+rejection (same "the confirmation IS the entry
condition, not a separate follow-through bar" design as
detectors/vwap_strategy.py): the retest bar's own wick reaching into the
zone plus its own close rejecting back out of it, together, define the
signal -- no 2-step candidate/confirm state machine.

Stop-loss: beyond the order block's own zone (and the actual retest bar's
extreme, if that wicked further than the block itself), widened to an ATR
floor if too tight -- same min_stop_distance_atr_mult convention as every
other strategy here, reusing average_true_range() unchanged. Target:
risk_reward_multiple * risk (default 2.0, externalized like
ValidatedBreakout/VWAPStrategy so backtest/engine.py's gap-risk guard --
which reads risk_reward_multiple via getattr(self.strategy,
"risk_reward_multiple", 2.0) -- stays correct automatically, no extra
code needed here).

No liquidity check inside evaluate() -- same convention as every other
strategy: liquidity gating happens ONLY in detectors/scanner.py's
scan_universe(), never inside a Strategy itself.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import Signal, SignalDirection, Strategy
from detectors.smart_money_concepts import OrderBlock, OrderBlockType, find_order_blocks
from detectors.support_resistance import average_true_range


class SmartMoneyStrategy(Strategy):
    name = "smart_money"

    def __init__(
        self,
        order_block_lookback_bars: int = 50,
        move_atr_mult: float = 1.5,
        move_window: int = 3,
        min_stop_distance_atr_mult: float = 0.5,
        risk_reward_multiple: float = 2.0,
        confidence: float = 0.5,
        fixed_stop_atr_mult: Optional[float] = None,
        entry_time_window: Optional[tuple[str, str]] = None,
    ):
        """
        fixed_stop_atr_mult: None (default) preserves the original stop
            placement described in the module docstring -- beyond the
            order block's own zone (and the retest bar's wick extreme, if
            that wicked further), widened to an ATR floor only if that
            zone-based risk is SMALLER than min_stop_distance_atr_mult*ATR.
            When set (e.g. 0.75), REPLACES that zone-based calculation
            entirely with a fixed stop_distance = fixed_stop_atr_mult*ATR
            from entry -- a tighter, ATR-only stop tested as an
            optimization variant (examples/run_strategy_optimization.py),
            not the default live behavior.
        entry_time_window: None (default) -- no restriction, matching
            original behavior. When set to (start, end) as "HH:MM"
            strings, evaluate() only fires for bars within that window
            (inclusive) -- tested as a variant to see whether avoiding
            late-session entries improves results, not a default filter.
        """
        self.order_block_lookback_bars = order_block_lookback_bars
        self.move_atr_mult = move_atr_mult
        self.move_window = move_window
        self.min_stop_distance_atr_mult = min_stop_distance_atr_mult
        self.risk_reward_multiple = risk_reward_multiple
        self.confidence = confidence
        self.fixed_stop_atr_mult = fixed_stop_atr_mult
        self.entry_time_window = entry_time_window

        self._traded_today: set[str] = set()
        # Same day-boundary caching pattern as every other strategy here
        # (reversal_strategy.py, trend_continuation_strategy.py,
        # vwap_strategy.py) -- avoids repeated expensive
        # df.index.date == trade_date full-index scans.
        self._day_start_cache: dict[str, pd.Timestamp] = {}
        # The order block retested to enter, keyed by symbol -- a local
        # variable in evaluate() with nowhere else to persist to. Safe to
        # key by symbol alone (not day_key): this engine only ever holds
        # one open trade per symbol at a time, so there's no risk of a
        # later day's entry overwriting a still-open earlier one.
        self._entry_order_block: dict[str, OrderBlock] = {}

    def evaluate(
        self, symbol: str, multi_tf_data: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> Optional[Signal]:
        trade_date = current_index.date()
        day_key = f"{symbol}_{trade_date}"
        if day_key in self._traded_today:
            return None

        if self.entry_time_window is not None:
            start, end = self.entry_time_window
            current_time_str = current_index.strftime("%H:%M")
            if not (start <= current_time_str <= end):
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

        if len(df) < self.order_block_lookback_bars + self.move_window + 5:
            return None

        order_blocks = find_order_blocks(
            df, lookback_bars=self.order_block_lookback_bars,
            move_atr_mult=self.move_atr_mult, move_window=self.move_window,
        )
        if not order_blocks:
            return None

        current_bar = df.iloc[-1]
        current_low = float(current_bar["low"])
        current_high = float(current_bar["high"])
        current_close = float(current_bar["close"])

        # find_order_blocks() already guarantees every returned block's
        # candle_time is at least move_window+1 bars before df's last row
        # (df itself is already truncated to current_index, no-lookahead)
        # -- this filter is defensive, not load-bearing.
        candidates = [b for b in order_blocks if b.candle_time < current_index]
        if not candidates:
            return None

        direction: Optional[SignalDirection] = None
        block = None
        for b in candidates:
            if b.block_type == OrderBlockType.BULLISH:
                # Retest: this bar's low reached down into the zone.
                # Rejection: this bar's close is back above the zone.
                if current_low <= b.high and current_close > b.high:
                    direction, block = SignalDirection.LONG, b
                    break
            else:  # BEARISH
                if current_high >= b.low and current_close < b.low:
                    direction, block = SignalDirection.SHORT, b
                    break

        if direction is None or block is None:
            return None

        # Bounded-window ATR: rolling(14) only ever depends on the last
        # ~14-15 rows -- slicing to the last 30 is a pure speed win, not
        # a behavior change, same pattern as every other strategy here.
        atr = average_true_range(df.iloc[-30:])
        min_distance = atr * self.min_stop_distance_atr_mult
        fixed_distance = atr * self.fixed_stop_atr_mult if self.fixed_stop_atr_mult is not None else None

        if direction == SignalDirection.LONG:
            touch_price = current_low
            if fixed_distance is not None:
                # Optimization variant: tight ATR-only stop, replacing the
                # zone-based calculation entirely (see __init__ docstring).
                stop_loss = current_close - fixed_distance
                risk = fixed_distance
                widened = False
            else:
                stop_loss = min(block.low, current_low)
                risk = current_close - stop_loss
                widened = risk < min_distance
                if widened:
                    stop_loss = current_close - min_distance
                    risk = min_distance
            if risk <= 0:
                return None
            target = current_close + self.risk_reward_multiple * risk
        else:
            touch_price = current_high
            if fixed_distance is not None:
                stop_loss = current_close + fixed_distance
                risk = fixed_distance
                widened = False
            else:
                stop_loss = max(block.high, current_high)
                risk = stop_loss - current_close
                widened = risk < min_distance
                if widened:
                    stop_loss = current_close + min_distance
                    risk = min_distance
            if risk <= 0:
                return None
            target = current_close - self.risk_reward_multiple * risk

        self._traded_today.add(day_key)
        self._entry_order_block[symbol] = block
        # Same level check_invalidation() treats as "the zone failed" --
        # block.low for a bullish retest-long, block.high for a bearish
        # retest-short -- reused here as the pre-entry pullback-check level.
        trigger_level = block.low if direction == SignalDirection.LONG else block.high
        return Signal(
            direction=direction,
            confidence=self.confidence,
            stop_loss=stop_loss,
            target=target,
            trigger_level=trigger_level,
            reasoning=[
                f"Retest of {block.block_type.value} order block from "
                f"{block.candle_time.strftime('%H:%M')} ({block.low:.2f}-{block.high:.2f}, "
                f"preceded a {block.move_size_atr:.1f} ATR move)",
                f"Rejection confirmed: bar reached {touch_price:.2f} into the zone, closed at {current_close:.2f}",
                f"Stop at {stop_loss:.2f}" + (
                    f", fixed {self.fixed_stop_atr_mult:.2f}x ATR (optimization variant)" if fixed_distance is not None
                    else ", widened to ATR floor" if widened else " (beyond zone/wick extreme)"
                ),
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

        Invalidation for an order-block retest trade: price closes
        through the OTHER side of the zone that was retested to enter --
        the zone has failed to hold, even though the hard stop
        (min(block.low, wick-extreme) or beyond) may sit further away
        still. This is a genuinely EARLIER warning than the stop in the
        widened-to-ATR-floor case.
        """
        block = self._entry_order_block.get(trade.symbol)
        if block is None:
            return None

        df = multi_tf_data.get(1)
        if df is None or df.empty:
            return None
        current_close = float(df.loc[:current_index]["close"].iloc[-1])

        if trade.direction == SignalDirection.LONG and current_close < block.low:
            return [f"Price closed below the {block.block_type.value} order block zone "
                    f"({block.low:.2f}-{block.high:.2f}) that was retested to enter -- zone failed"]
        if trade.direction == SignalDirection.SHORT and current_close > block.high:
            return [f"Price closed above the {block.block_type.value} order block zone "
                    f"({block.low:.2f}-{block.high:.2f}) that was retested to enter -- zone failed"]
        return None
