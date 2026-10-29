"""
Smart Money Concepts (SMC): a family of price-action ideas describing how
larger participants ("smart money") are theorized to leave footprints in
price -- order blocks and liquidity sweeps being the two most concretely
definable ones from OHLCV data alone (concepts like true "institutional
order flow" require order-book/Level 2 data this project doesn't have
access to; what follows are the well-defined, commonly-used PRICE-ACTION
proxies for these ideas, not literal institutional order data).

Order Block: the last opposing candle before a strong, decisive move --
e.g. the last down-candle before a sharp rally is a "bullish order block."
The idea (widely used in SMC-style retail trading communities and
discretionary price-action trading, not an academically proven concept)
is that this zone represents where large buy orders were absorbed before
price moved away, and price often returns to "retest" this zone before
continuing. Definition used here: a candle immediately preceding a move
of at least `move_atr_mult` x ATR in the opposite direction to that
candle's own direction.

Liquidity Sweep: price briefly breaks past an obvious swing high/low
(where stop-losses are assumed to cluster) and then sharply reverses --
the theory being that the break was designed to trigger those stops
("sweep the liquidity") before the real move happens in the other
direction. This is structurally very similar to what breakout_validator.py
already checks (fakeout detection) -- a liquidity sweep IS a specific,
named case of a failed breakout with a sharp reversal, viewed through SMC
terminology. This module doesn't duplicate that logic; it flags the
PATTERN using the same swing-point infrastructure from support_resistance.py.

Honest framing: SMC terminology is popular in retail trading communities,
but much of it (especially "order blocks," "fair value gaps") lacks the
same level of independent statistical validation as, say, ADX or standard
volume analysis. Treat these as ADDITIONAL price-action context to
combine with the other detectors' evidence, not as standalone proof of
anything -- backtest before trusting, same as everything else in this
project.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from detectors.support_resistance import average_true_range


class OrderBlockType(Enum):
    BULLISH = "bullish"  # last down-candle before a strong up move
    BEARISH = "bearish"  # last up-candle before a strong down move


@dataclass
class OrderBlock:
    block_type: OrderBlockType
    candle_time: pd.Timestamp
    high: float
    low: float
    move_size_atr: float  # how many ATRs the subsequent move covered -- a rough
    # strength/significance measure for the block


@dataclass
class LiquiditySweep:
    swept_level: float
    sweep_time: pd.Timestamp
    reversal_direction: str  # "up" or "down" -- the direction price reversed INTO
    penetration_atr: float  # how far beyond the level price briefly went, in ATR terms


def find_order_blocks(
    df: pd.DataFrame,
    lookback_bars: int = 50,
    move_atr_mult: float = 1.5,
    move_window: int = 3,
) -> list[OrderBlock]:
    """
    Scans the last `lookback_bars` for order blocks. move_window: how many
    bars after the candidate candle to measure the "decisive move" over --
    a short window (2-4 bars) is standard for identifying a sharp,
    impulsive move rather than a slow drift that happens to cover the
    same distance over many bars (the latter isn't the same phenomenon).
    """
    if len(df) < lookback_bars:
        return []

    window = df.iloc[-lookback_bars:]
    atr = average_true_range(window)
    if atr <= 0:
        return []

    raw_blocks = []
    for i in range(len(window) - move_window - 1):
        candidate = window.iloc[i]
        candidate_is_down = candidate["close"] < candidate["open"]
        candidate_is_up = candidate["close"] > candidate["open"]
        if not (candidate_is_down or candidate_is_up):
            continue

        move_start_price = float(candidate["close"])
        move_end_price = float(window.iloc[i + move_window]["close"])
        move = move_end_price - move_start_price
        move_atr = abs(move) / atr

        if move_atr < move_atr_mult:
            continue

        # Bullish order block: candidate candle was DOWN, subsequent move UP.
        if candidate_is_down and move > 0:
            raw_blocks.append(OrderBlock(
                block_type=OrderBlockType.BULLISH,
                candle_time=window.index[i],
                high=float(candidate["high"]),
                low=float(candidate["low"]),
                move_size_atr=float(move_atr),
            ))
        # Bearish order block: candidate candle was UP, subsequent move DOWN.
        elif candidate_is_up and move < 0:
            raw_blocks.append(OrderBlock(
                block_type=OrderBlockType.BEARISH,
                candle_time=window.index[i],
                high=float(candidate["high"]),
                low=float(candidate["low"]),
                move_size_atr=float(move_atr),
            ))

    # Deduplicate: within one continuous trending move, many consecutive
    # candles individually satisfy "candle before an X-ATR move" (each
    # sees the trend continuing for `move_window` bars past it). Without
    # collapsing these, one real trend move gets reported as dozens of
    # "order blocks" -- which isn't the concept (an order block is meant
    # to be a specific, singular zone, not every candle inside a trend).
    # Cluster candidates of the same type within `move_window` bars of
    # each other and keep only the strongest (highest move_size_atr).
    raw_blocks.sort(key=lambda b: b.candle_time)
    blocks: list[OrderBlock] = []
    cluster: list[OrderBlock] = []

    def flush_cluster():
        if cluster:
            blocks.append(max(cluster, key=lambda b: b.move_size_atr))

    for block in raw_blocks:
        if cluster and (
            cluster[-1].block_type != block.block_type
            or (block.candle_time - cluster[-1].candle_time) > pd.Timedelta(minutes=move_window)
        ):
            flush_cluster()
            cluster = []
        cluster.append(block)
    flush_cluster()

    return blocks


def find_liquidity_sweeps(
    df: pd.DataFrame,
    lookback_bars: int = 50,
    fractal_window: int = 3,
    min_penetration_atr: float = 0.1,
    reversal_bars: int = 2,
) -> list[LiquiditySweep]:
    """
    Reuses the same swing-point detection as support_resistance.py to
    find obvious swing highs/lows, then checks whether price later broke
    past one and reversed sharply within `reversal_bars` -- the liquidity
    sweep pattern. This deliberately mirrors breakout_validator.py's
    follow-through concept, just checking for the OPPOSITE outcome
    (failure + reversal, rather than confirmation + continuation).
    """
    if len(df) < lookback_bars:
        return []

    window = df.iloc[-lookback_bars:]
    atr = average_true_range(window)
    if atr <= 0:
        return []

    highs = window["high"]
    lows = window["low"]
    is_swing_high = highs == highs.rolling(2 * fractal_window + 1, center=True).max()
    is_swing_low = lows == lows.rolling(2 * fractal_window + 1, center=True).min()

    swing_highs = highs[is_swing_high].dropna()
    swing_lows = lows[is_swing_low].dropna()

    sweeps = []

    for swing_time, swing_price in swing_highs.items():
        idx = window.index.get_loc(swing_time)
        if idx + reversal_bars + 1 >= len(window):
            continue
        after = window.iloc[idx + 1: idx + 1 + reversal_bars + 1]
        max_penetration = (after["high"].max() - swing_price) / atr
        if max_penetration < min_penetration_atr:
            continue
        final_close = float(after["close"].iloc[-1])
        if final_close < swing_price:  # swept above, then closed back below -- reversal down
            sweeps.append(LiquiditySweep(
                swept_level=float(swing_price), sweep_time=swing_time,
                reversal_direction="down", penetration_atr=float(max_penetration),
            ))

    for swing_time, swing_price in swing_lows.items():
        idx = window.index.get_loc(swing_time)
        if idx + reversal_bars + 1 >= len(window):
            continue
        after = window.iloc[idx + 1: idx + 1 + reversal_bars + 1]
        max_penetration = (swing_price - after["low"].min()) / atr
        if max_penetration < min_penetration_atr:
            continue
        final_close = float(after["close"].iloc[-1])
        if final_close > swing_price:  # swept below, then closed back above -- reversal up
            sweeps.append(LiquiditySweep(
                swept_level=float(swing_price), sweep_time=swing_time,
                reversal_direction="up", penetration_atr=float(max_penetration),
            ))

    return sorted(sweeps, key=lambda s: s.sweep_time)
