"""
Momentum Entry: a real, tradeable Strategy (unlike detectors/strong_trend.py
and detectors/top_movers.py, both purely informational/display) built by
combining Top Movers' "genuinely moving with volume" idea with Strong
Trend's persistence checks -- meant to catch a stock like Manorama Inds
(+9.87% on 2026-08-17) that was never in the morning watchlist and so
never got a chance to fire ANY of the other 8 strategies, all of which
only evaluate watchlist-tracked symbols.

ALL of these must hold (direction inferred from move sign, not passed in):
  1. Move >= min_move_pct (default 3.0%) from today's open -- a strong
     move, not noise. Deliberately a HIGHER bar than detectors/strong_
     trend.py's own 1.5% (gate 1 there) -- this module is a real entry
     signal, that one is a display list; the two are allowed to disagree
     on threshold, they answer different questions.
  2. Volume z-score >= min_volume_zscore (default 1.5) -- reuses
     detectors/breakout_validator.py's volume_zscore() (same z-score
     convention detectors/pick_verification.py's check_volume_
     confirmation() already uses), over the same VOLUME_LOOKBACK_BARS
     window pick_verification.py uses, not a second, potentially-
     diverging lookback. Deliberately a HIGHER bar than pick_
     verification's own MIN_VOLUME_ZSCORE (1.0) -- again, a real entry
     signal earns a stricter volume bar than the softer post-hoc
     verification layer other strategies get.
  3. Strong Trend's gates 2-4 (persistence, NOT its gate 1 move-threshold
     or gate 5 sector check -- see detectors/strong_trend.py's own
     module docstring for why those two don't apply here): check_trend_
     consistency() (>=3 consecutive HH+HL/LH+LL 5-min bars), check_
     volume_consistency() (last 3 five-min bars each >=70% of today's
     average), check_opening_range_hold() (no close back through the
     broken opening-range level). Reused as-is, not reimplemented.
  4. Freshness: the move must have started holding continuously within
     the last freshness_window_minutes (default 30) -- reuses detectors/
     strong_trend.py's find_trend_start() (same backward-walk that
     produces its "trending since" display value) to find when the
     qualifying move actually began, then rejects if that's more than
     freshness_window_minutes ago. This is what keeps this "fresh
     momentum, not stale" -- a stock up 5% since 9:20 AM that's now flat
     at 2 PM shouldn't still fire at 2 PM.
  5. Liquidity + Rs.100 floor: NOT checked inside evaluate() -- same
     convention every strategy in this project follows (see detectors/
     smc_strategy.py's module docstring: "liquidity gating happens ONLY
     in detectors/scanner.py's scan_universe(), never inside a Strategy
     itself"). detectors/scanner.py's scan_symbol() already applies
     MIN_STOCK_PRICE and analyze_liquidity() to every symbol BEFORE any
     strategy (including this one) is ever evaluated, so this gate is
     satisfied by the existing centralized pipeline, not duplicated here.

NOT added to detectors/pick_verification.py's VERIFIED_STRATEGIES: this
strategy's own gates 1+2 above (3% move, 1.5 volume z-score) are already
stricter versions of exactly what CHANGE 2's post-hoc verification would
otherwise re-check (1.0% move, 1.0 volume z-score) -- running verification
on top would be redundant double-filtering against a signal that already
proved itself, the same reasoning validated_breakout and reversal_
detection are exempt for (see pick_verification.py's own module
docstring: both already gate behind their own multi-step confirmation).

Stop-loss: swing low (long) / swing high (short) over the last
swing_lookback_bars (default 10) 1-min bars, widened to an ATR floor if
too tight -- same min_stop_distance_atr_mult convention (default 0.5)
every other strategy here uses, reusing average_true_range() unchanged
(bounded to the last 30 bars -- same speed-only truncation support_
resistance.py's find_levels() already documents, not a behavior change).
Target: risk_reward_multiple * risk (default 2.0, same fixed 1:2
convention as every other strategy).

Every Signal's reasoning explicitly includes the required experimental
disclaimer -- this strategy has NOT been backtested, its 5 thresholds
above were chosen by combining two already-shipped modules' own defaults,
not validated against historical performance. Treat every signal from
this strategy as experimental until a real backtest exists (examples/
run_universe_backtest.py-style, same as every other strategy went
through before being trusted).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import Signal, SignalDirection, Strategy
from detectors.breakout_validator import volume_zscore
from detectors.pick_verification import VOLUME_LOOKBACK_BARS
from detectors.strong_trend import (
    STRONG_TREND_MIN_CONSECUTIVE,
    STRONG_TREND_OPENING_RANGE_MINUTES,
    STRONG_TREND_VOLUME_MIN_RATIO,
    STRONG_TREND_WINDOW_MINUTES,
    check_opening_range_hold,
    check_trend_consistency,
    check_volume_consistency,
    find_trend_start,
)
from detectors.support_resistance import average_true_range

MOMENTUM_ENTRY_DISCLAIMER = (
    "Momentum-based entry -- no backtested edge validated yet. Treat as experimental until backtested."
)


def compute_swing_stop(
    day_df: pd.DataFrame, direction: SignalDirection, current_price: float,
    swing_lookback_bars: int = 10, min_stop_distance_atr_mult: float = 0.5,
) -> tuple[float, float]:
    """
    Swing low (long) / swing high (short) over the last swing_lookback_bars
    1-min bars, widened to an ATR floor if too tight -- same convention
    every strategy in this project uses for a market-structure-based stop
    (see e.g. detectors/validated_breakout_strategy.py, detectors/
    smc_strategy.py). Factored out here (not left inline in evaluate())
    because detectors/best_picks.py -- an informational display feature,
    not a Strategy -- needs the exact same calculation and must not
    silently drift from what this strategy actually uses.

    Returns (stop, risk) -- risk is always positive (the ATR floor already
    applied), so a caller never needs to re-derive |current_price - stop|.
    """
    recent = day_df.iloc[-swing_lookback_bars:]
    atr = average_true_range(day_df.iloc[-30:])
    min_distance = atr * min_stop_distance_atr_mult

    if direction == SignalDirection.LONG:
        stop = float(recent["low"].min())
        risk = current_price - stop
        if risk < min_distance:
            stop = current_price - min_distance
            risk = min_distance
    else:
        stop = float(recent["high"].max())
        risk = stop - current_price
        if risk < min_distance:
            stop = current_price + min_distance
            risk = min_distance
    return stop, risk


class MomentumEntryStrategy(Strategy):
    name = "momentum_entry"

    def __init__(
        self,
        min_move_pct: float = 3.0,
        min_volume_zscore: float = 1.5,
        freshness_window_minutes: int = 30,
        swing_lookback_bars: int = 10,
        min_stop_distance_atr_mult: float = 0.5,
        risk_reward_multiple: float = 2.0,
        trend_window_minutes: int = STRONG_TREND_WINDOW_MINUTES,
        trend_min_consecutive: int = STRONG_TREND_MIN_CONSECUTIVE,
        volume_min_ratio: float = STRONG_TREND_VOLUME_MIN_RATIO,
        opening_range_minutes: int = STRONG_TREND_OPENING_RANGE_MINUTES,
    ):
        self.min_move_pct = min_move_pct
        self.min_volume_zscore = min_volume_zscore
        self.freshness_window_minutes = freshness_window_minutes
        self.swing_lookback_bars = swing_lookback_bars
        self.min_stop_distance_atr_mult = min_stop_distance_atr_mult
        self.risk_reward_multiple = risk_reward_multiple
        self.trend_window_minutes = trend_window_minutes
        self.trend_min_consecutive = trend_min_consecutive
        self.volume_min_ratio = volume_min_ratio
        self.opening_range_minutes = opening_range_minutes

        self._traded_today: set[str] = set()
        # The level check_invalidation() treats as "the move failed" --
        # the swing stop actually placed, keyed the same way _traded_today
        # is ("{symbol}_{date}"), same pattern examples/opening_range_
        # breakout.py's _range_cache uses for its own invalidation hook.
        self._entry_stop: dict[str, float] = {}

    def evaluate(
        self, symbol: str, multi_tf_data: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> Optional[Signal]:
        trade_date = current_index.date()
        day_key = f"{symbol}_{trade_date}"
        if day_key in self._traded_today:
            return None

        df_1min = multi_tf_data.get(1)
        if df_1min is None or df_1min.empty:
            return None

        today_df = df_1min[df_1min.index.date == trade_date].loc[:current_index]
        if today_df.empty:
            return None

        current_close = float(today_df["close"].iloc[-1])
        open_price = float(today_df["open"].iloc[0])
        if open_price <= 0:
            return None

        # Gate 1: strong move from open.
        raw_move_pct = (current_close - open_price) / open_price * 100.0
        if raw_move_pct >= self.min_move_pct:
            direction = SignalDirection.LONG
        elif raw_move_pct <= -self.min_move_pct:
            direction = SignalDirection.SHORT
        else:
            return None

        # Gate 2: genuinely elevated volume, not a thin-volume pump.
        if len(today_df) < VOLUME_LOOKBACK_BARS + 2:
            return None
        current_volume = float(today_df["volume"].iloc[-1])
        recent_volumes = today_df["volume"].iloc[-(VOLUME_LOOKBACK_BARS + 1):-1]
        vz = volume_zscore(current_volume, recent_volumes)
        if vz < self.min_volume_zscore:
            return None

        # Gate 3: Strong Trend's persistence checks (gates 2-4 there),
        # reused as-is -- see module docstring for why gates 1/5 don't apply.
        consistent, trend_reason = check_trend_consistency(
            today_df, current_index, direction, self.trend_window_minutes, self.trend_min_consecutive,
        )
        if not consistent:
            return None
        volume_steady, volume_reason = check_volume_consistency(today_df, self.volume_min_ratio)
        if not volume_steady:
            return None
        range_ok, range_reason = check_opening_range_hold(today_df, direction, self.opening_range_minutes)
        if not range_ok:
            return None

        # Gate 4: freshness -- the qualifying move must have started
        # holding within the last freshness_window_minutes, not hours ago.
        trend_start = find_trend_start(today_df, direction, open_price, self.min_move_pct)
        age_minutes = (current_index - trend_start).total_seconds() / 60.0
        if age_minutes > self.freshness_window_minutes:
            return None

        # Entry/stop/target -- see compute_swing_stop()'s docstring for why
        # this is a shared function, not inline here.
        stop, risk = compute_swing_stop(
            today_df, direction, current_close, self.swing_lookback_bars, self.min_stop_distance_atr_mult,
        )
        if direction == SignalDirection.LONG:
            target = current_close + self.risk_reward_multiple * risk
        else:
            target = current_close - self.risk_reward_multiple * risk

        if risk <= 0:
            return None

        self._traded_today.add(day_key)
        self._entry_stop[day_key] = stop

        return Signal(
            direction=direction,
            confidence=0.5,  # placeholder single-factor confidence, same convention as OpeningRangeBreakout --
            # not yet calibrated against backtest results (this strategy has no backtest yet at all, see module docstring).
            stop_loss=stop,
            target=target,
            # trigger_level intentionally left None (default) -- that field
            # is for strategies whose confirmation IS a single decisive
            # level (see Signal's own docstring), which this isn't: it's a
            # persistence+volume signal, not a level break. check_
            # invalidation() below tracks its own stop reference directly
            # (self._entry_stop), same as OpeningRangeBreakout's own hook
            # doesn't route through this field either.
            reasoning=[
                f"Momentum entry: moved {raw_move_pct:+.2f}% from today's open ({open_price:.2f}) -- "
                f"passes the {self.min_move_pct:.1f}% minimum",
                f"Volume z-score {vz:.2f} (>= {self.min_volume_zscore:.1f} minimum) -- genuinely elevated, not a thin pump",
                f"Freshness: qualifying move has held continuously since {trend_start.strftime('%H:%M')} "
                f"({age_minutes:.0f} min ago, within the {self.freshness_window_minutes:.0f} min window)",
                trend_reason, volume_reason, range_reason,
                MOMENTUM_ENTRY_DISCLAIMER,
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

        Invalidation: price closes back through the swing stop level that
        justified entry -- same "the move that justified this trade has
        failed" concept examples/opening_range_breakout.py and detectors/
        validated_breakout_strategy.py both use for their own invalidation
        hooks, applied to this strategy's own stop reference.
        """
        day_key = f"{trade.symbol}_{trade.entry_time.date()}"
        stop_level = self._entry_stop.get(day_key)
        if stop_level is None:
            return None

        df = multi_tf_data.get(1)
        if df is None or df.empty:
            return None
        current_close = float(df.loc[:current_index]["close"].iloc[-1])

        if trade.direction == SignalDirection.LONG and current_close < stop_level:
            return [f"Price closed back below the swing-low stop reference ({stop_level:.2f}) -- momentum failed"]
        if trade.direction == SignalDirection.SHORT and current_close > stop_level:
            return [f"Price closed back above the swing-high stop reference ({stop_level:.2f}) -- momentum failed"]
        return None
