"""
Trend Continuation strategy -- enters WITH the trend when a pullback is
classified as likely_continuation, using pullback_classifier.py. Skips
entirely when the classifier says likely_reversal or uncertain -- the
"no trade" case is a deliberate output, not a gap to fill later, because
your notes describe "very few setups go the same direction till market
close" as a symptom of forcing trades in ambiguous conditions.

Entry: next bar after a likely_continuation classification, in the trend
direction. Stop: beyond the pullback's extreme (low of the pullback for
longs, high for shorts) -- if price takes out the pullback low in an
uptrend continuation trade, the continuation thesis is already wrong.
Target: 2x risk, consistent with the other strategies for comparability.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import Signal, SignalDirection, Strategy
from detectors.market_regime import Regime, detect_regime
from detectors.pullback_classifier import (
    TrendDirection,
    PullbackVerdict,
    classify_pullback,
    find_recent_trend_leg,
)
from detectors.support_resistance import average_true_range


class TrendContinuation(Strategy):
    name = "trend_continuation"

    def __init__(
        self,
        trend_lookback_bars: int = 20,
        pullback_bars: int = 2,
        min_confidence: float = 0.75,
        min_stop_distance_atr_mult: float = 0.5,
        require_regime_confirmation: bool = True,
    ):
        self.trend_lookback_bars = trend_lookback_bars
        self.pullback_bars = pullback_bars
        self.min_confidence = min_confidence
        self.min_stop_distance_atr_mult = min_stop_distance_atr_mult
        self.require_regime_confirmation = require_regime_confirmation
        self._traded_today: set[str] = set()
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
        if len(df) < self.trend_lookback_bars + self.pullback_bars + 5:
            return None

        # Restrict to the current trading day only. multi_tf_data[1] is a
        # truncated but still multi-day series (it's "everything up to
        # current_index", not "today"), so without this a 20-bar lookback
        # near market open would silently span the overnight gap between
        # yesterday's close and today's open -- producing a fake "trend leg"
        # out of a price gap that has nothing to do with intraday momentum.
        #
        # df.index.date forces an expensive full-index date conversion --
        # do that once per day (cached below) instead of once per bar; every
        # other bar of the same day reuses a cheap timestamp-range slice
        # that gives the exact same rows, since df is already truncated to
        # current_index.
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

        # Trend leg is computed on data BEFORE the pullback window --
        # pass only that slice so find_recent_trend_leg can't see the
        # pullback bars it's supposed to be evaluated against.
        pre_pullback = df.iloc[:-self.pullback_bars] if self.pullback_bars > 0 else df
        trend_leg = find_recent_trend_leg(pre_pullback, lookback_bars=self.trend_lookback_bars)
        if trend_leg is None:
            return None

        classification = classify_pullback(df, trend_leg, pullback_bars=self.pullback_bars)
        if classification is None:
            return None

        if classification.verdict != PullbackVerdict.LIKELY_CONTINUATION:
            return None
        if classification.confidence < self.min_confidence:
            return None

        # Regime gate: a continuation signal is only as trustworthy as the
        # broader trend context backing it up. This directly targets a
        # real false positive we found -- pure noise data occasionally
        # produced a "valid" continuation signal from the pullback
        # classifier alone (a short 20-bar window can look trend-like by
        # chance). Requiring the slower, 15-min ADX-based regime to agree
        # on direction filters out exactly that kind of coincidental signal,
        # because a regime reading is far less likely to falsely agree in
        # the same direction purely by chance.
        if self.require_regime_confirmation:
            df_15 = multi_tf_data.get(15)
            if df_15 is None or df_15.empty:
                return None
            regime_reading = detect_regime(df_15)
            if regime_reading is None:
                return None
            expected_regime = (
                Regime.TRENDING_UP if trend_leg.direction == TrendDirection.UP
                else Regime.TRENDING_DOWN
            )
            if regime_reading.regime != expected_regime:
                return None

        current_close = float(df["close"].iloc[-1])
        pullback_window = df.iloc[-self.pullback_bars:]

        # A stop this tight (e.g. 15 paise on an ~800 rupee stock) can look
        # fine in R-multiple terms but isn't tradeable in real markets --
        # bid-ask spread and normal slippage would blow through it before
        # an order could even be acknowledged. Floor the stop distance at
        # a fraction of ATR so the backtest only counts setups that are
        # actually executable, not just mathematically valid.
        # df is already day-bounded (see the cross-day fix above), but ATR's
        # rolling(14) only needs the last ~15 rows -- slicing further is a
        # pure speed win, not a behavior change.
        atr = average_true_range(df.iloc[-30:])
        min_distance = atr * self.min_stop_distance_atr_mult

        if trend_leg.direction == TrendDirection.UP:
            stop = float(pullback_window["low"].min())
            risk = current_close - stop
            if risk < min_distance:
                stop = current_close - min_distance
                risk = min_distance
            if risk <= 0:
                return None
            target = current_close + 2 * risk
            direction = SignalDirection.LONG
        else:
            stop = float(pullback_window["high"].max())
            risk = stop - current_close
            if risk < min_distance:
                stop = current_close + min_distance
                risk = min_distance
            if risk <= 0:
                return None
            target = current_close - 2 * risk
            direction = SignalDirection.SHORT

        self._traded_today.add(day_key)
        reasoning = list(classification.reasoning)
        if self.require_regime_confirmation and regime_reading is not None:
            reasoning.extend(regime_reading.reasoning)
        return Signal(
            direction=direction,
            confidence=classification.confidence,
            stop_loss=stop,
            target=target,
            reasoning=reasoning,
        )

    def check_invalidation(
        self, trade: Trade, multi_tf_data: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> Optional[list[str]]:
        """
        Early-exit hook (see paper_trading/engine.py's on_new_bar()) --
        called on every bar while this strategy has an OPEN position,
        BEFORE the hard stop-loss/target check. Not part of the Strategy
        ABC -- called via getattr() duck-typing.

        Invalidation for a trend-continuation trade: the 15-min ADX-based
        regime (same detect_regime() used to gate entry) flips away from
        the direction this trade needs. No new instance state required --
        the expected regime is re-derived from trade.direction itself
        (LONG -> TRENDING_UP, SHORT -> TRENDING_DOWN), the same mapping
        used at entry time.
        """
        df_15 = multi_tf_data.get(15)
        if df_15 is None or df_15.empty:
            return None
        regime_reading = detect_regime(df_15)
        if regime_reading is None:
            return None

        expected_regime = Regime.TRENDING_UP if trade.direction == SignalDirection.LONG else Regime.TRENDING_DOWN
        if regime_reading.regime == expected_regime:
            return None
        return [
            f"Market regime flipped from {expected_regime.value} to {regime_reading.regime.value} "
            f"(ADX {regime_reading.adx:.1f}) -- trend-continuation thesis no longer holds"
        ]
