"""
Validated Breakout strategy -- wires support_resistance.py and
breakout_validator.py into the Strategy interface so it can run through
BacktestEngine like Opening Range Breakout did.

This is Phase 2 from the build sequence: prove the level-detection +
breakout-validation logic produces a better risk/reward profile than a
naive "price crossed a level" approach would. Compare this strategy's
backtest report against a naive version (not included here, but easy to
build: just skip validate_breakout and enter on any level cross) to see
whether the validation logic is actually earning its complexity -- that's
the "reject features with low ROI" principle from your own brief applied
to this component itself.

Mechanics:
  - Levels are computed from 15-minute data (see support_resistance.py's
    note on why not 1-minute) using data available up to the current bar
    only -- recomputed periodically, not looking at the full dataset.
  - When 1-minute price closes beyond a known level, that's a candidate
    breakout. The strategy waits exactly one more bar (the engine's
    natural fill-at-next-open lag already does this for the trade entry,
    but validate_breakout needs its own confirmation bar first, so entry
    is effectively 2 bars after the raw level cross -- slower than ORB,
    deliberately, because confirmation costs time and that trade-off is
    the whole point of validation).
  - Stop-loss: back on the other side of the level. Target: 2x the risk.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import Signal, SignalDirection, Strategy
from detectors.breakout_validator import BreakoutDirection, validate_breakout
from detectors.multi_timeframe_confirmation import TrendDirection as MtfDirection, check_alignment
from detectors.support_resistance import Level, average_true_range, find_levels, nearest_level


class ValidatedBreakout(Strategy):
    name = "validated_breakout"

    def __init__(
        self,
        level_timeframe: int = 15,
        level_recompute_every_bars: int = 30,
        min_touches: int = 2,
        min_stop_distance_atr_mult: float = 0.5,
        require_mtf_alignment: bool = True,
        mtf_timeframes: tuple[int, ...] = (5, 15),
        risk_reward_multiple: float = 2.0,
        required_confirmation_bars: int = 1,
        min_distance_atr_mult: Optional[float] = None,
        min_volume_zscore: float = 0.5,
    ):
        """
        required_confirmation_bars: 1 (default) preserves the original
            behavior exactly -- validate_breakout() runs on the very next
            bar after the level cross, same as always. Raised (e.g. to 2)
            by examples/run_strategy_optimization.py's confirmation-bars
            variant: the candidate must hold on the CORRECT side of the
            level for every intermediate bar (a reversal before reaching
            required_confirmation_bars drops the candidate immediately, as
            a fakeout) before validate_breakout() finally runs, using the
            LAST of those bars as its confirmation_time -- a stronger
            momentum requirement than the original single-bar check, not
            the default live behavior.
        min_distance_atr_mult: None (default) -- no hard gate beyond
            validate_breakout()'s own existing distance_atr>0.15 SOFT
            factor (one of 4, only 2 of which are required alongside the
            mandatory follow-through check). When set (e.g. 1.5),
            REJECTS the breakout outright if validation.distance_atr
            falls short, on top of (not instead of) the existing scoring
            -- tested as a stricter optimization variant, not live.
        min_volume_zscore: threaded straight through to
            detectors.breakout_validator.validate_breakout()'s own
            parameter of the same name -- default 0.5 matches that
            function's original always-live default exactly.
        """
        self.level_timeframe = level_timeframe
        self.level_recompute_every_bars = level_recompute_every_bars
        self.min_touches = min_touches
        self.min_stop_distance_atr_mult = min_stop_distance_atr_mult
        self.require_mtf_alignment = require_mtf_alignment
        self.mtf_timeframes = list(mtf_timeframes)
        # Target = entry +/- this multiple of risk. Default 2.0 matches the
        # original fixed 1:2 convention every other strategy in this project
        # also uses (see backtest/engine.py's gap-risk guard, which derives
        # intended_risk from |target - stop_loss| assuming a 1:2 signal --
        # that derivation is strategy-agnostic w.r.t. THIS parameter since
        # it recomputes the ratio from target/stop_loss directly, but it's
        # worth noting this strategy is the one place that ratio isn't
        # hardcoded at 3x anymore once risk_reward_multiple != 2.0).
        self.risk_reward_multiple = risk_reward_multiple
        self.required_confirmation_bars = required_confirmation_bars
        self.min_distance_atr_mult = min_distance_atr_mult
        self.min_volume_zscore = min_volume_zscore

        self._traded_today: set[str] = set()
        self._cached_levels: dict[str, list[Level]] = {}  # keyed by symbol
        self._bars_since_recompute: dict[str, int] = {}
        # Track candidate breakouts awaiting their confirmation bar(s):
        # {symbol: (level, breakout_time, bars_confirmed_so_far)}
        self._pending_confirmation: dict[str, tuple[Level, pd.Timestamp, int]] = {}
        # The level that was broken to enter, keyed the same way
        # _traded_today is ("{symbol}_{date}") -- unlike ORB's range, this
        # level was a local variable in evaluate() with nowhere to persist
        # to, so it's tracked here specifically to support check_invalidation().
        self._entry_level: dict[str, float] = {}

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

        df_1min = multi_tf_data[1]
        df_level_tf = multi_tf_data.get(self.level_timeframe)
        if df_1min.empty or df_level_tf is None or df_level_tf.empty:
            return None

        # Recompute levels periodically rather than every single bar --
        # levels don't meaningfully change minute to minute, and this
        # keeps the backtest fast.
        bars_since = self._bars_since_recompute.get(symbol, 999)
        if symbol not in self._cached_levels or bars_since >= self.level_recompute_every_bars:
            self._cached_levels[symbol] = find_levels(df_level_tf, min_touches=self.min_touches)
            self._bars_since_recompute[symbol] = 0
        else:
            self._bars_since_recompute[symbol] = bars_since + 1

        levels = self._cached_levels[symbol]
        if not levels:
            return None

        # Step 1: do we have a pending breakout awaiting confirmation?
        if symbol in self._pending_confirmation:
            level, breakout_time, bars_confirmed = self._pending_confirmation.pop(symbol)

            # Stronger-momentum check (required_confirmation_bars>1 only --
            # a no-op for the default of 1, since bars_confirmed is
            # incremented and compared BEFORE ever reaching validate_breakout,
            # exactly matching the original single-bar-later call below):
            # every intermediate bar must still close on the CORRECT side of
            # the level, or the candidate is dropped immediately as a fakeout
            # rather than waiting for the full confirmation window.
            breakout_bar_close = float(df_1min.loc[breakout_time, "close"])
            current_close_check = float(df_1min.loc[current_index, "close"])
            still_holding = (
                current_close_check > level.price if breakout_bar_close > level.price
                else current_close_check < level.price
            )
            if not still_holding:
                return None  # reversed before confirmation completed -- fakeout

            bars_confirmed += 1
            if bars_confirmed < self.required_confirmation_bars:
                self._pending_confirmation[symbol] = (level, breakout_time, bars_confirmed)
                return None

            validation = validate_breakout(
                df_1min=df_1min,
                level=level,
                breakout_time=breakout_time,
                confirmation_time=current_index,
                min_volume_zscore=self.min_volume_zscore,
            )
            if validation is None:
                return None
            if not validation.is_valid:
                return None  # confirmed fakeout -- exactly what we're trying to avoid trading
            if self.min_distance_atr_mult is not None and validation.distance_atr < self.min_distance_atr_mult:
                return None  # optimization variant: stricter break-distance gate on top of validate_breakout's own scoring

            # Multi-timeframe gate: a validated breakout is still just a
            # local (1-min-level) event. If the 5-min and 15-min trend are
            # actively moving against the breakout direction, taking the
            # trade means fighting the bigger current -- exactly the kind
            # of setup that tends to reverse hard, which is one of the
            # loss patterns this whole project started from.
            if self.require_mtf_alignment:
                alignment = check_alignment(multi_tf_data, self.mtf_timeframes)
                expected_direction = (
                    MtfDirection.UP if validation.direction == BreakoutDirection.BULLISH
                    else MtfDirection.DOWN
                )
                if not (alignment.aligned and alignment.direction == expected_direction):
                    return None

            self._traded_today.add(day_key)
            current_close = float(df_1min.loc[current_index, "close"])
            level_price = level.price
            self._entry_level[day_key] = level_price
            # df_1min is the full multi-day history up to current_index (it
            # grows every bar) -- ATR's rolling(14) only needs the last
            # ~15 rows, so slicing here is a pure speed win, not a
            # behavior change (see support_resistance.py's find_levels()).
            atr = average_true_range(df_1min.iloc[-30:])
            min_distance = atr * self.min_stop_distance_atr_mult

            if validation.direction == BreakoutDirection.BULLISH:
                stop = level_price
                risk_points = current_close - stop
                if risk_points < min_distance:
                    stop = current_close - min_distance
                    risk_points = min_distance
                target = current_close + self.risk_reward_multiple * risk_points
                direction = SignalDirection.LONG
            else:
                stop = level_price
                risk_points = stop - current_close
                if risk_points < min_distance:
                    stop = current_close + min_distance
                    risk_points = min_distance
                target = current_close - self.risk_reward_multiple * risk_points
                direction = SignalDirection.SHORT

            if risk_points <= 0:
                return None  # degenerate case, skip

            reasoning = list(validation.reasoning)
            if self.require_mtf_alignment:
                reasoning.extend(alignment.reasoning)

            return Signal(
                direction=direction,
                confidence=validation.confidence,
                stop_loss=stop,
                target=target,
                reasoning=reasoning,
            )

        # Step 2: no pending confirmation -- look for a fresh level cross
        # on this bar to start the confirmation clock.
        current_close = float(df_1min.loc[current_index, "close"])
        level = nearest_level(levels, current_close, max_distance_pct=2.0)
        if level is None:
            return None

        crossed_up = current_close > level.price
        crossed_down = current_close < level.price

        # Only treat as a fresh breakout if the PREVIOUS bar was still on
        # the other side of the level -- otherwise this is bar 5 of a move
        # that already broke out, not the breakout itself.
        prior_bars = df_1min.loc[:current_index].iloc[:-1]
        if prior_bars.empty:
            return None
        prior_close = float(prior_bars.iloc[-1]["close"])

        just_crossed = (crossed_up and prior_close <= level.price) or (
            crossed_down and prior_close >= level.price
        )
        if just_crossed:
            self._pending_confirmation[symbol] = (level, current_index, 0)

        return None

    def check_invalidation(
        self, trade: Trade, multi_tf_data: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> Optional[list[str]]:
        """
        Early-exit hook (see paper_trading/engine.py's on_new_bar()) --
        called on every bar while this strategy has an OPEN position,
        BEFORE the hard stop-loss/target check. Not part of the Strategy
        ABC -- called via getattr() duck-typing.

        Same invalidation concept as OpeningRangeBreakout: price closes
        back through the level that was broken (and confirmed) to enter,
        even though the hard stop sits further away (stop = level_price
        widened by an ATR floor, so a close-back-through can fire before
        the stop is ever touched).
        """
        day_key = f"{trade.symbol}_{trade.entry_time.date()}"
        level_price = self._entry_level.get(day_key)
        if level_price is None:
            return None

        df = multi_tf_data.get(1)
        if df is None or df.empty:
            return None
        current_close = float(df.loc[:current_index]["close"].iloc[-1])

        if trade.direction == SignalDirection.LONG and current_close < level_price:
            return [f"Price closed back below the level ({level_price:.2f}) it broke above to enter -- breakout failed"]
        if trade.direction == SignalDirection.SHORT and current_close > level_price:
            return [f"Price closed back above the level ({level_price:.2f}) it broke below to enter -- breakout failed"]
        return None
