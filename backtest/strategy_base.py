"""
Every future component -- breakout validator, fakeout detector, scanner
ranking logic -- implements this interface so the backtest engine can run
any of them without knowing their internals. This is the plug point that
lets Phase 2/3/4 slot in without touching the engine itself.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class Signal:
    direction: SignalDirection
    confidence: float  # 0-1, probability estimate -- never a binary guarantee
    stop_loss: float
    target: float
    reasoning: list[str]  # human-readable explanation, required -- this is
    # the "explainable AI" requirement: every signal must say WHY, not just what.
    # The exact price level that was broken/reclaimed/retested to justify
    # this signal (e.g. opening-range high/low, VWAP at cross time, an
    # order block's edge, a Fibonacci level) -- distinct from stop_loss,
    # which may sit further away (widened to an ATR floor). Optional and
    # None by default: only strategies with a genuine "confirmation IS a
    # single decisive level" design populate it (see detectors/
    # pick_verification.py, which uses it to check for an immediate
    # pullback back through that level before treating the setup as
    # confirmed). Purely additive -- backtest/engine.py never reads this
    # field, so strategies that leave it None are completely unaffected.
    trigger_level: Optional[float] = None


class Strategy(ABC):
    """
    A Strategy evaluates one bar at a time across one or more timeframes
    and either returns a Signal or None (no trade). Strategies are stateless
    between calls except for whatever internal state they explicitly manage
    (e.g. "already in a position for this symbol") -- the engine, not the
    strategy, owns position/portfolio state.
    """

    name: str

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        multi_tf_data: dict[int, pd.DataFrame],
        current_index: pd.Timestamp,
    ) -> Optional[Signal]:
        """
        multi_tf_data: {1: df_1min, 5: df_5min, 15: df_15min, ...} -- each
        DataFrame truncated to data available *as of* current_index (the
        engine guarantees no lookahead; see engine.py).
        current_index: the timestamp of the bar being evaluated.
        Returns a Signal to enter a trade, or None.
        """
        raise NotImplementedError
