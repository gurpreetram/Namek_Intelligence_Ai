"""
Identifies significant support/resistance levels from price history.

Approach: find swing highs/lows (fractals -- a bar's high/low that's more
extreme than N bars on either side), then cluster nearby swing points into
levels using ATR-based tolerance (not a fixed price tolerance, since a
0.50 tolerance means something very different for a Rs.50 stock than a
Rs.3000 stock). A level's significance is its touch count -- how many
times price has approached and reacted at that zone. This is deliberately
NOT just "yesterday's high/low" -- your notes describe reactions at levels
that clearly matter to other market participants, which means levels
tested multiple times across sessions, not single-session extremes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Level:
    price: float
    touch_count: int
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp
    touch_timestamps: list[pd.Timestamp] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Level({self.price:.2f}, touches={self.touch_count})"


def average_true_range(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else float(tr.mean())


def find_swing_points(df: pd.DataFrame, fractal_window: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    A swing high at index i: high[i] is the max within [i-window, i+window].
    A swing low at index i: low[i] is the min within [i-window, i+window].
    Returns (swing_highs, swing_lows) as Series of prices indexed by timestamp,
    containing only the confirmed swing points (NaN elsewhere).
    """
    highs, lows = df["high"], df["low"]
    is_swing_high = highs == highs.rolling(2 * fractal_window + 1, center=True).max()
    is_swing_low = lows == lows.rolling(2 * fractal_window + 1, center=True).min()
    return highs[is_swing_high], lows[is_swing_low]


def find_levels(
    df: pd.DataFrame,
    fractal_window: int = 3,
    min_touches: int = 2,
    cluster_tolerance_atr_mult: float = 0.25,
) -> list[Level]:
    """
    df: OHLCV DataFrame (any single timeframe -- typically 5m or 15m;
        running this on 1m data produces noisy, low-significance levels).
    min_touches: levels touched fewer than this many times are discarded --
        a level with only 1 touch isn't demonstrated support/resistance,
        it's just a random high or low.
    cluster_tolerance_atr_mult: swing points within this many ATRs of each
        other are merged into one level.
    """
    if len(df) < 2 * fractal_window + 1:
        return []

    # average_true_range()'s rolling(14) only ever depends on the last ~15
    # rows -- slicing to the last 30 here is a pure speed win (identical
    # ATR value, since df's tail 14-15 rows are unaffected by truncating
    # what came before them), not a behavior change. df itself still holds
    # its full passed-in range for swing-point detection below.
    atr = average_true_range(df.iloc[-30:])
    tolerance = atr * cluster_tolerance_atr_mult
    if tolerance <= 0:
        return []

    swing_highs, swing_lows = find_swing_points(df, fractal_window)
    all_swings = pd.concat([swing_highs.dropna(), swing_lows.dropna()]).sort_index()

    if all_swings.empty:
        return []

    # Greedy clustering: sort swing prices, merge points within tolerance
    # of the running cluster average.
    sorted_swings = all_swings.sort_values()
    clusters: list[list[tuple[pd.Timestamp, float]]] = []
    for ts, price in sorted_swings.items():
        placed = False
        for cluster in clusters:
            cluster_avg = np.mean([p for _, p in cluster])
            if abs(price - cluster_avg) <= tolerance:
                cluster.append((ts, price))
                placed = True
                break
        if not placed:
            clusters.append([(ts, price)])

    levels = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        prices = [p for _, p in cluster]
        timestamps = sorted(ts for ts, _ in cluster)
        levels.append(Level(
            price=float(np.mean(prices)),
            touch_count=len(cluster),
            first_seen=timestamps[0],
            last_seen=timestamps[-1],
            touch_timestamps=timestamps,
        ))

    return sorted(levels, key=lambda l: l.touch_count, reverse=True)


def nearest_level(levels: list[Level], price: float, max_distance_pct: float = 1.5) -> Level | None:
    """Finds the closest level to a given price, within max_distance_pct% -- used
    to check 'is the current price sitting near a known level right now'."""
    if not levels:
        return None
    candidates = [
        lvl for lvl in levels
        if abs(lvl.price - price) / price * 100 <= max_distance_pct
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda lvl: abs(lvl.price - price))
