"""
Sector Confluence Trend -- same entry as TrendContinuation (pullback
classified LIKELY_CONTINUATION, regime-confirmed), gated by ONE additional
requirement: the stock's own SECTOR must also be trending in the same
direction (see detectors/sector_trend.py). A signal only fires when both
agree.

Built as a SEPARATE strategy rather than a flag on TrendContinuation itself,
specifically so the two can be backtested side by side without touching the
original -- see examples/run_sector_confluence_comparison.py.

Implementation note on why this does NOT wrap one persistent
TrendContinuation instance: TrendContinuation marks its own "already traded
today" state the moment it decides to fire, inside evaluate() itself. If we
reused one shared instance and then rejected its signal here (sector
disagreed), that instance would still believe it already traded today and
refuse to offer a second candidate later the same day -- silently making
this strategy MORE conservative than "TrendContinuation AND sector
agreement" actually implies. Instead, a fresh TrendContinuation is
constructed on every evaluate() call (its underlying decision needs only
the current call's data, no cross-bar confirmation state to lose), and this
class owns its OWN _traded_today gate, only marked when a signal actually
clears both checks and is returned.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.strategy_base import Signal, SignalDirection, Strategy
from detectors.sector_trend import compute_sector_trend
from detectors.trend_continuation_strategy import TrendContinuation


class SectorConfluenceTrend(Strategy):
    name = "sector_confluence_trend"

    def __init__(
        self,
        sector_mapping: dict[str, str],
        sector_price_data: dict[str, dict[str, pd.DataFrame]],
        trend_lookback_bars: int = 20,
        pullback_bars: int = 2,
        min_confidence: float = 0.75,
        min_stop_distance_atr_mult: float = 0.5,
        require_regime_confirmation: bool = True,
        sector_trend_lookback_bars: int = 20,
        min_sector_trend_pct: float = 0.0015,
    ):
        self.sector_mapping = sector_mapping
        self.sector_price_data = sector_price_data
        self.sector_trend_lookback_bars = sector_trend_lookback_bars
        # Sector return must clear this magnitude to count as "trending" in
        # either direction -- without a floor, near-zero noise would count
        # as agreeing with whichever direction the stock's own signal picked.
        self.min_sector_trend_pct = min_sector_trend_pct
        self._base_kwargs = dict(
            trend_lookback_bars=trend_lookback_bars,
            pullback_bars=pullback_bars,
            min_confidence=min_confidence,
            min_stop_distance_atr_mult=min_stop_distance_atr_mult,
            require_regime_confirmation=require_regime_confirmation,
        )
        self._traded_today: set[str] = set()

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

        sector = self.sector_mapping.get(symbol)
        if sector is None:
            return None  # no official sector label -- can't confirm confluence, no trade
        sector_data = self.sector_price_data.get(sector)
        if not sector_data:
            return None

        base = TrendContinuation(**self._base_kwargs)
        signal = base.evaluate(symbol, multi_tf_data, current_index)
        if signal is None or signal.direction == SignalDirection.NONE:
            return None

        sector_return = compute_sector_trend(
            current_index, sector_data, lookback_bars=self.sector_trend_lookback_bars,
        )
        if sector_return is None:
            return None

        if sector_return > self.min_sector_trend_pct:
            sector_direction = SignalDirection.LONG
        elif sector_return < -self.min_sector_trend_pct:
            sector_direction = SignalDirection.SHORT
        else:
            sector_direction = None  # sector itself isn't clearly trending either way

        if sector_direction is None or sector_direction != signal.direction:
            return None

        self._traded_today.add(day_key)
        signal.reasoning = list(signal.reasoning) + [
            f"Sector ({sector}) confluence: {sector_return:+.2%} over last "
            f"{self.sector_trend_lookback_bars} bars agrees with {signal.direction.value} direction"
        ]
        return signal
