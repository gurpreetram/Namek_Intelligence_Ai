"""
Fibonacci Retracement strategy: after a clear directional swing leg (see
detectors/fibonacci.py's find_most_recent_leg()), enters when price pulls
back to one of the two most commonly cited retracement levels -- 38.2% or
61.8% -- and shows a same-bar rejection back in the leg's original
direction, confirmed by above-average volume. LONG on an up-leg pullback
that rejects a Fib level to the upside; SHORT on a down-leg pullback that
rejects a Fib level to the downside.

Single-bar retest+rejection (same "the confirmation IS the entry
condition, not a separate follow-through bar" design as detectors/
vwap_strategy.py and detectors/smc_strategy.py): the bar's own wick
reaching into the level's small ATR-based tolerance band, plus its own
close rejecting back out past the level, together define the signal.

Stop-loss: just beyond the NEXT DEEPER Fibonacci level (the adjacent
ratio in fibonacci.RETRACEMENT_RATIOS past the one that triggered entry
-- e.g. a 38.2% entry stops beyond 50%, a 61.8% entry stops beyond 78.6%;
78.6% itself stops beyond the swing leg's own start, i.e. full
invalidation), widened to an ATR floor if that raw distance is too tight
-- same min_stop_distance_atr_mult convention as every other strategy
here, reusing average_true_range() unchanged. Target: risk_reward_multiple
* risk (default 2.0, externalized like every other strategy so backtest/
engine.py's gap-risk guard -- which reads risk_reward_multiple via
getattr(self.strategy, "risk_reward_multiple", 2.0) -- stays correct
automatically, no extra code needed here).

No liquidity or MIN_STOCK_PRICE check inside evaluate() -- same
convention as every other strategy in this project: both gates apply
ONLY in detectors/scanner.py's fast-scan path (Daily Picks), never inside
a Strategy itself, and neither VWAP nor Smart Money Concepts apply them
in their own evaluate() either. This strategy's backtest runs are
therefore comparable to VWAP's/SMC's own initial 400-symbol tests on that
basis -- none of the three filter the backtest universe by liquidity or
price.

HONEST CAVEAT, matching detectors/fibonacci.py's own: Fibonacci
retracement's predictive validity is CONTESTED in academic literature --
popular in retail technical analysis, but not something to assume works
before testing. This strategy exists to let a real backtest answer that
question on this project's own data, not because the concept is presumed
to have edge.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import Signal, SignalDirection, Strategy
from detectors.fibonacci import (
    RETRACEMENT_RATIOS,
    LegDirection,
    compute_retracement_levels,
    find_most_recent_leg,
)
from detectors.support_resistance import average_true_range


class FibonacciStrategy(Strategy):
    name = "fibonacci_retracement"

    def __init__(
        self,
        swing_fractal_window: int = 3,
        swing_lookback_bars: int = 50,
        min_leg_atr_mult: float = 1.5,
        key_ratios: tuple[float, ...] = (0.382, 0.618),
        level_tolerance_atr_mult: float = 0.15,
        volume_lookback_bars: int = 20,
        volume_multiplier: float = 1.3,
        min_stop_distance_atr_mult: float = 0.5,
        risk_reward_multiple: float = 2.0,
        confidence: float = 0.5,
    ):
        self.swing_fractal_window = swing_fractal_window
        self.swing_lookback_bars = swing_lookback_bars
        self.min_leg_atr_mult = min_leg_atr_mult
        self.key_ratios = key_ratios
        self.level_tolerance_atr_mult = level_tolerance_atr_mult
        self.volume_lookback_bars = volume_lookback_bars
        self.volume_multiplier = volume_multiplier
        self.min_stop_distance_atr_mult = min_stop_distance_atr_mult
        self.risk_reward_multiple = risk_reward_multiple
        self.confidence = confidence

        self._traded_today: set[str] = set()
        self._day_start_cache: dict[str, pd.Timestamp] = {}
        # The next-deeper Fib level's price, keyed by symbol -- ONLY set
        # when it's genuinely different from stop_loss (i.e. the ATR floor
        # widened the stop away from it). When NOT widened, deeper_price
        # == stop_loss exactly, so a separate invalidation check would be
        # unreachable (the hard stop always fires first on intrabar
        # touch) -- not tracked in that case, deliberately.
        self._entry_invalidation_price: dict[str, float] = {}

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

        min_bars_needed = max(self.volume_lookback_bars, self.swing_fractal_window * 2 + 1, 30) + 5
        if len(df) < min_bars_needed:
            return None

        swing_window = df.iloc[-self.swing_lookback_bars:]
        leg = find_most_recent_leg(
            swing_window, fractal_window=self.swing_fractal_window, min_leg_atr_mult=self.min_leg_atr_mult,
        )
        if leg is None:
            return None

        fib = compute_retracement_levels(leg)
        atr = average_true_range(df.iloc[-30:])
        if atr <= 0:
            return None
        tolerance = atr * self.level_tolerance_atr_mult

        current_bar = df.iloc[-1]
        current_close = float(current_bar["close"])
        current_low = float(current_bar["low"])
        current_high = float(current_bar["high"])

        avg_volume = float(df["volume"].iloc[-(self.volume_lookback_bars + 1):-1].mean())
        current_volume = float(current_bar["volume"])
        if not (avg_volume > 0 and current_volume >= self.volume_multiplier * avg_volume):
            return None

        direction: Optional[SignalDirection] = None
        triggered_ratio: Optional[float] = None
        for ratio in self.key_ratios:
            level_price = fib.level(ratio)
            if leg.direction == LegDirection.UP:
                # Pullback DOWN toward the level from above the swing
                # high; bullish rejection: wick pierces the level (+
                # tolerance), close rejects back above it.
                if current_low <= level_price + tolerance and current_close > level_price:
                    direction, triggered_ratio = SignalDirection.LONG, ratio
                    break
            else:
                # Pullback UP toward the level from below the swing low;
                # bearish rejection: wick pierces the level (- tolerance),
                # close rejects back below it.
                if current_high >= level_price - tolerance and current_close < level_price:
                    direction, triggered_ratio = SignalDirection.SHORT, ratio
                    break

        if direction is None or triggered_ratio is None:
            return None

        ratio_index = RETRACEMENT_RATIOS.index(triggered_ratio)
        if ratio_index + 1 < len(RETRACEMENT_RATIOS):
            deeper_ratio = RETRACEMENT_RATIOS[ratio_index + 1]
            deeper_price = fib.level(deeper_ratio)
            stop_reference = f"beyond the {deeper_ratio:.1%} level ({deeper_price:.2f})"
        else:
            deeper_price = leg.start_price
            stop_reference = f"beyond the leg's start ({deeper_price:.2f}, full invalidation)"

        min_distance = atr * self.min_stop_distance_atr_mult

        if direction == SignalDirection.LONG:
            raw_risk = current_close - deeper_price
            widened = raw_risk < min_distance
            stop_loss = current_close - min_distance if widened else deeper_price
            risk = current_close - stop_loss
            if risk <= 0:
                return None
            target = current_close + self.risk_reward_multiple * risk
        else:
            raw_risk = deeper_price - current_close
            widened = raw_risk < min_distance
            stop_loss = current_close + min_distance if widened else deeper_price
            risk = stop_loss - current_close
            if risk <= 0:
                return None
            target = current_close - self.risk_reward_multiple * risk

        self._traded_today.add(day_key)
        if widened:
            self._entry_invalidation_price[symbol] = deeper_price
        else:
            self._entry_invalidation_price.pop(symbol, None)
        triggered_level_price = fib.level(triggered_ratio)
        return Signal(
            direction=direction,
            confidence=self.confidence,
            stop_loss=stop_loss,
            target=target,
            trigger_level=triggered_level_price,
            reasoning=[
                f"{leg.direction.value.capitalize()} leg: {leg.start_price:.2f} ({leg.start_time.strftime('%H:%M')}) "
                f"-> {leg.end_price:.2f} ({leg.end_time.strftime('%H:%M')})",
                f"Pullback reached {triggered_ratio:.1%} retracement ({fib.level(triggered_ratio):.2f}), "
                f"rejected back to close at {current_close:.2f}",
                f"Volume confirmed: {current_volume:.0f} vs {avg_volume:.0f} avg ({current_volume / avg_volume:.1f}x)",
                f"Stop at {stop_loss:.2f} ({stop_reference}" + (", widened to ATR floor" if widened else "") + ")",
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

        Only meaningful for trades where the ATR floor widened stop_loss
        AWAY from the next-deeper Fib level (see __init__'s note) -- in
        that case, a close back through the (nearer) Fib level is a
        genuinely earlier warning than the (farther) hard stop. Symbols
        without a tracked invalidation price simply return None here,
        matching every trade where this check would otherwise be
        redundant with the stop.
        """
        invalidation_price = self._entry_invalidation_price.get(trade.symbol)
        if invalidation_price is None:
            return None

        df = multi_tf_data.get(1)
        if df is None or df.empty:
            return None
        current_close = float(df.loc[:current_index]["close"].iloc[-1])

        if trade.direction == SignalDirection.LONG and current_close < invalidation_price:
            return [f"Price closed back below the next Fibonacci level ({invalidation_price:.2f}) -- pullback deepened past the retracement thesis"]
        if trade.direction == SignalDirection.SHORT and current_close > invalidation_price:
            return [f"Price closed back above the next Fibonacci level ({invalidation_price:.2f}) -- pullback deepened past the retracement thesis"]
        return None
