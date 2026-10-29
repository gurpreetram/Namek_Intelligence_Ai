"""
Liquidity Analysis: checks whether a stock currently has enough trading
activity to be safely tradeable intraday. This is a pure execution-risk
filter, separate from setup quality -- a technically perfect breakout
signal is still a bad trade if the stock is thin enough that entering or
exiting moves the price against you (slippage) or you simply can't fill
your size.

What this module does NOT do: give a real bid-ask spread or order-book
depth reading -- that data isn't available from OHLCV candles alone (it
needs Level 2 market depth, which is a separate, often paid data feed).
Instead this uses traded VALUE (price x volume) as a practical proxy --
the standard approach when only OHLCV is available -- and volume
consistency (are there dead/near-zero-volume minutes) as a second signal.
Treat this as a reasonable first filter, not a substitute for checking
actual depth/spread once you have that data source.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LiquidityReading:
    is_liquid: bool
    avg_traded_value: float
    avg_volume: float
    dead_bar_pct: float  # fraction of bars with near-zero volume
    reasoning: list[str]


def analyze_liquidity(
    df: pd.DataFrame,
    lookback_bars: int = 30,
    min_avg_traded_value: float = 500_000.0,
    max_dead_bar_pct: float = 0.10,
) -> LiquidityReading | None:
    """
    lookback_bars: window to assess current liquidity over -- recent
        liquidity matters more than the whole day's average, since
        liquidity can dry up intraday even for normally-liquid stocks
        (e.g. around lunch hours).
    min_avg_traded_value: minimum average per-minute traded value (price
        x volume, in rupees) to consider the stock liquid enough for
        intraday entry/exit without excessive slippage. Rs.5,00,000/min
        is a commonly used practical heuristic among intraday traders and
        screeners for "safely liquid" -- NOT a regulatory or universal
        number. Adjust this based on your own position size: a trader
        risking Rs.500/trade needs far less liquidity than one risking
        Rs.50,000/trade. Treat the default as a reasonable starting point
        to tune, not gospel.
    max_dead_bar_pct: if more than this fraction of recent bars have
        near-zero volume, that's a sign of an illiquid stretch (possibly
        a stock that only trades in bursts) even if the average looks
        acceptable -- an average can hide a stock that's dead most
        minutes and spikes occasionally, which is a worse execution
        profile than one that's steadily moderate.
    """
    if len(df) < lookback_bars:
        return None

    window = df.iloc[-lookback_bars:]
    traded_value = window["close"] * window["volume"]
    avg_traded_value = float(traded_value.mean())
    avg_volume = float(window["volume"].mean())

    dead_threshold = avg_volume * 0.05  # bars with <5% of average volume
    dead_bars = (window["volume"] < dead_threshold).sum()
    dead_bar_pct = dead_bars / len(window)

    value_ok = avg_traded_value >= min_avg_traded_value
    consistency_ok = dead_bar_pct <= max_dead_bar_pct
    is_liquid = value_ok and consistency_ok

    reasoning = [
        f"Avg traded value/min over last {lookback_bars} bars: Rs.{avg_traded_value:,.0f} "
        f"({'OK' if value_ok else f'below Rs.{min_avg_traded_value:,.0f} threshold'})",
        f"Dead/thin bars: {dead_bar_pct:.1%} of window "
        f"({'OK' if consistency_ok else f'exceeds {max_dead_bar_pct:.0%} threshold'})",
    ]
    if not is_liquid:
        reasoning.append(
            "NOT recommended for intraday entry/exit at typical position sizes -- "
            "slippage risk is elevated"
        )

    return LiquidityReading(
        is_liquid=is_liquid,
        avg_traded_value=avg_traded_value,
        avg_volume=avg_volume,
        dead_bar_pct=float(dead_bar_pct),
        reasoning=reasoning,
    )
