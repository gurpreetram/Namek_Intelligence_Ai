"""
Structured numeric feature extraction for the ML signal-combination
experiment (examples/train_signal_model.py) -- converts each historical
Trade's entry-time market context into a fixed-width numeric feature
vector, reusing existing detector functions rather than recomputing any
of this logic from scratch. Every feature here is something one of this
project's own detectors already computes as TEXT reasoning somewhere
(breakout_validator.py's confidence scoring, market_regime.py's regime
labels, relative_strength.py's excess-return verdict) -- this module's
only job is extracting the same underlying NUMBERS in a form a model can
consume, not inventing new market-analysis logic.

No-lookahead discipline: every feature is computed strictly from data at
or before the trade's own entry_time. These trades are historical
(already closed), but a "predictive" model trained on features that peek
past entry_time would be silently cheating -- same discipline the
backtest engine itself enforces, just applied here to a feature pipeline
instead of a strategy's evaluate().

RECENT_DAYS_FOR_CONTEXT bounding: regime detection and relative strength
only ever look at a small recent tail (ADX needs ~42 bars of 15-min data,
relative strength needs 20 bars), so feeding them the FULL truncated
history (which grows to ~30 days x 375 bars for a trade late in the
backtest window) would resample enormous, mostly-irrelevant history on
every single trade for an identical answer -- pure waste at the scale
this runs at (tens of thousands of trades). Same bounded-window
optimization already used in detectors/scanner.py's fast-scan path, same
justification: neither computation needs more than a handful of recent
days, and 5 days gives ~3x the ADX minimum, the same "3x period" warm-up
margin already used elsewhere in this project.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.resample import multi_timeframe_view
from detectors.breakout_validator import _close_location_value
from detectors.breakout_validator import volume_zscore as compute_volume_zscore
from detectors.liquidity_analysis import analyze_liquidity
from detectors.market_regime import detect_regime
from detectors.relative_strength import compute_relative_strength
from detectors.support_resistance import average_true_range

MARKET_OPEN_MINUTES = 9 * 60 + 15  # 09:15 IST in minutes-since-midnight
RECENT_DAYS_FOR_CONTEXT = 5

# Every key in TradeFeatures.features, in a fixed order -- the training
# script builds its feature matrix off this list, not by trusting dict
# ordering, so a future reordering here can't silently misalign columns.
FEATURE_NAMES = [
    "avg_traded_value", "dead_bar_pct", "volume_zscore", "clv",
    "risk_atr_multiple", "adx", "plus_di", "minus_di", "regime_available",
    "excess_return_pct", "rs_available", "minutes_since_open", "day_of_week",
    "confidence",
]


@dataclass
class TradeFeatures:
    symbol: str
    entry_time: pd.Timestamp
    strategy_name: str
    features: dict[str, float]
    label_win: int  # 1 if r_multiple > 0 else 0 -- the primary target
    r_multiple: float  # secondary continuous target


def extract_features_for_trade(trade, df_1min: pd.DataFrame, benchmark_df: pd.DataFrame | None) -> TradeFeatures | None:
    """
    trade: a backtest.engine.Trade -- must already be closed (exit_price/
        exit_time set), since the label is derived from its outcome.
    df_1min: the FULL 1-minute history for trade.symbol; this function
        truncates to trade.entry_time itself. Callers processing many
        trades for the same symbol should load this ONCE and reuse it
        (same "share the load" principle as backtest.engine.run_multi()),
        not reload per trade.
    benchmark_df: 1-minute data for the relative-strength benchmark (e.g.
        NIFTYBEES), or None to skip relative-strength features (they
        come back 0.0 with rs_available=0.0 rather than raising).
    Returns None if there isn't enough history before entry_time for the
    core (liquidity/ATR) features -- those need at least ~30 bars.
    """
    if trade.exit_price is None or trade.exit_time is None:
        return None

    truncated = df_1min.loc[:trade.entry_time]
    if len(truncated) < 35:
        return None

    current_bar = truncated.iloc[-1]

    liquidity = analyze_liquidity(truncated)
    avg_traded_value = float(liquidity.avg_traded_value) if liquidity is not None else 0.0
    dead_bar_pct = float(liquidity.dead_bar_pct) if liquidity is not None else 0.0

    recent_volumes = truncated["volume"].iloc[-21:-1]
    volume_zscore = compute_volume_zscore(float(current_bar["volume"]), recent_volumes) if len(recent_volumes) >= 20 else 0.0
    clv = _close_location_value(current_bar)

    atr = average_true_range(truncated.iloc[-30:])
    risk_atr_multiple = abs(trade.entry_price - trade.stop_loss) / atr if atr > 0 else 0.0

    recent_dates = sorted({d for d in truncated.index.date})[-RECENT_DAYS_FOR_CONTEXT:]
    bounded = truncated[truncated.index.date >= recent_dates[0]]

    mtf = multi_timeframe_view(bounded, [1, 15])
    regime = detect_regime(mtf[15])
    if regime is not None:
        adx, plus_di, minus_di, regime_available = regime.adx, regime.plus_di, regime.minus_di, 1.0
    else:
        adx = plus_di = minus_di = 0.0
        regime_available = 0.0

    excess_return_pct, rs_available = 0.0, 0.0
    if benchmark_df is not None:
        bench_truncated = benchmark_df.loc[:trade.entry_time]
        bench_bounded = bench_truncated[bench_truncated.index.date >= recent_dates[0]]
        rs = compute_relative_strength(bounded, bench_bounded)
        if rs is not None:
            excess_return_pct, rs_available = rs.excess_return_pct, 1.0

    minutes_since_open = trade.entry_time.hour * 60 + trade.entry_time.minute - MARKET_OPEN_MINUTES
    day_of_week = float(trade.entry_time.weekday())

    features = {
        "avg_traded_value": avg_traded_value,
        "dead_bar_pct": dead_bar_pct,
        "volume_zscore": float(volume_zscore),
        "clv": float(clv),
        "risk_atr_multiple": float(risk_atr_multiple),
        "adx": float(adx),
        "plus_di": float(plus_di),
        "minus_di": float(minus_di),
        "regime_available": regime_available,
        "excess_return_pct": float(excess_return_pct),
        "rs_available": rs_available,
        "minutes_since_open": float(minutes_since_open),
        "day_of_week": day_of_week,
        "confidence": float(trade.confidence),
    }

    return TradeFeatures(
        symbol=trade.symbol, entry_time=trade.entry_time, strategy_name=trade.strategy_name,
        features=features, label_win=1 if trade.r_multiple > 0 else 0, r_multiple=float(trade.r_multiple),
    )
