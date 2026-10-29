"""
Classic chart pattern recognition: Double Top, Double Bottom, Head &
Shoulders, and Inverse Head & Shoulders -- built from swing-point
sequences (reusing support_resistance.py's fractal-based swing detector,
same infrastructure the rest of this project already relies on, not a
separate ad-hoc implementation).

Honest framing up front: these are among the most widely taught chart
patterns in retail technical analysis, but rigorous academic evidence for
their predictive power is genuinely mixed -- some studies (e.g. Lo, Mamaysky
& Wang 2000, "Foundations of Technical Analysis") found certain patterns
carry weak statistical information beyond noise, others find no reliable
edge once transaction costs are accounted for. Treat a detected pattern
as ONE more piece of context to combine with the other detectors' evidence
(volume, structure, multi-timeframe alignment), never as standalone proof.
The backtest engine exists precisely to check whether trading these
patterns, on this specific universe and timeframe, actually has edge --
don't assume it before checking.

Pattern definitions used here (standard textbook definitions):
  - Double Top: two swing highs at approximately the same price, with a
    swing low ("valley") between them. Confirmed (not just "forming") when
    price closes below the valley low -- the "neckline break."
  - Double Bottom: mirror image -- two similar swing lows with a swing
    high ("peak") between them, confirmed on a close above the peak.
  - Head & Shoulders: three swing highs where the middle one (head) is
    higher than the two outer ones (shoulders), which are roughly equal
    to each other. Confirmed on a close below the "neckline" -- the line
    connecting the two swing lows on either side of the head.
  - Inverse Head & Shoulders: mirror image for swing lows, confirmed on a
    close above the neckline connecting the two swing highs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from detectors.support_resistance import find_swing_points


class PatternType(Enum):
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"


@dataclass
class ChartPattern:
    pattern_type: PatternType
    confirmed: bool  # True if the neckline has actually broken, False if
    # the shape exists but hasn't confirmed yet (still "forming")
    neckline_price: float
    key_points: list[tuple[pd.Timestamp, float]]  # the swing points forming the pattern
    reasoning: list[str]


def _similar(a: float, b: float, tolerance_pct: float) -> bool:
    return abs(a - b) / max(abs(a), abs(b), 1e-9) * 100 <= tolerance_pct


def find_double_top_bottom(
    df: pd.DataFrame,
    fractal_window: int = 3,
    peak_similarity_pct: float = 0.5,
    lookback_bars: int = 100,
) -> list[ChartPattern]:
    """
    peak_similarity_pct: how close (in %) the two peaks/troughs must be
    to count as "the same level" -- real double tops rarely hit the exact
    same price twice, so some tolerance is required; too loose and this
    starts matching unrelated swings.
    """
    if len(df) < lookback_bars:
        return []

    window = df.iloc[-lookback_bars:]
    swing_highs, swing_lows = find_swing_points(window, fractal_window)
    swing_highs = swing_highs.dropna()
    swing_lows = swing_lows.dropna()

    patterns = []

    # Double Top: consecutive pairs of swing highs similar in price, with
    # a swing low strictly between them in time.
    high_times = list(swing_highs.index)
    for i in range(len(high_times) - 1):
        t1, t2 = high_times[i], high_times[i + 1]
        p1, p2 = float(swing_highs[t1]), float(swing_highs[t2])
        if not _similar(p1, p2, peak_similarity_pct):
            continue
        between_lows = swing_lows[(swing_lows.index > t1) & (swing_lows.index < t2)]
        if between_lows.empty:
            continue
        valley_time = between_lows.idxmin()
        valley_price = float(between_lows.min())

        after_t2 = window[window.index > t2]
        confirmed = not after_t2.empty and float(after_t2["close"].iloc[-1]) < valley_price

        patterns.append(ChartPattern(
            pattern_type=PatternType.DOUBLE_TOP,
            confirmed=confirmed,
            neckline_price=valley_price,
            key_points=[(t1, p1), (valley_time, valley_price), (t2, p2)],
            reasoning=[
                f"Two peaks near {p1:.2f}/{p2:.2f} ({t1.strftime('%H:%M')}, {t2.strftime('%H:%M')})",
                f"Valley (neckline) at {valley_price:.2f}",
                "Neckline broken -- pattern CONFIRMED" if confirmed
                else "Neckline not yet broken -- pattern still forming, not tradeable yet",
            ],
        ))

    # Double Bottom: mirror.
    low_times = list(swing_lows.index)
    for i in range(len(low_times) - 1):
        t1, t2 = low_times[i], low_times[i + 1]
        p1, p2 = float(swing_lows[t1]), float(swing_lows[t2])
        if not _similar(p1, p2, peak_similarity_pct):
            continue
        between_highs = swing_highs[(swing_highs.index > t1) & (swing_highs.index < t2)]
        if between_highs.empty:
            continue
        peak_time = between_highs.idxmax()
        peak_price = float(between_highs.max())

        after_t2 = window[window.index > t2]
        confirmed = not after_t2.empty and float(after_t2["close"].iloc[-1]) > peak_price

        patterns.append(ChartPattern(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            confirmed=confirmed,
            neckline_price=peak_price,
            key_points=[(t1, p1), (peak_time, peak_price), (t2, p2)],
            reasoning=[
                f"Two troughs near {p1:.2f}/{p2:.2f} ({t1.strftime('%H:%M')}, {t2.strftime('%H:%M')})",
                f"Peak (neckline) at {peak_price:.2f}",
                "Neckline broken -- pattern CONFIRMED" if confirmed
                else "Neckline not yet broken -- pattern still forming, not tradeable yet",
            ],
        ))

    return patterns


def find_head_and_shoulders(
    df: pd.DataFrame,
    fractal_window: int = 3,
    shoulder_similarity_pct: float = 1.0,
    min_head_prominence_pct: float = 0.3,
    lookback_bars: int = 150,
) -> list[ChartPattern]:
    """
    shoulder_similarity_pct: how close the two shoulders must be to each
    other in price to count as "roughly equal" -- looser than the double
    top/bottom tolerance since real shoulders are rarely as symmetric as
    textbook diagrams suggest.
    min_head_prominence_pct: how much higher (for H&S) or lower (for
    inverse H&S) the head must be than the shoulders, as a percentage --
    filters out cases where three swings are all roughly equal (that's
    not a head & shoulders, that's just a range).
    """
    if len(df) < lookback_bars:
        return []

    window = df.iloc[-lookback_bars:]
    swing_highs, swing_lows = find_swing_points(window, fractal_window)
    swing_highs = swing_highs.dropna()
    swing_lows = swing_lows.dropna()

    patterns = []

    # Head & Shoulders: three consecutive swing highs, middle one tallest.
    high_times = list(swing_highs.index)
    for i in range(len(high_times) - 2):
        t_left, t_head, t_right = high_times[i], high_times[i + 1], high_times[i + 2]
        p_left, p_head, p_right = (
            float(swing_highs[t_left]), float(swing_highs[t_head]), float(swing_highs[t_right])
        )
        if not _similar(p_left, p_right, shoulder_similarity_pct):
            continue
        head_prominence = (p_head - max(p_left, p_right)) / p_head * 100
        if head_prominence < min_head_prominence_pct:
            continue

        # Neckline: the two swing lows between left-shoulder/head and head/right-shoulder.
        lows_between_1 = swing_lows[(swing_lows.index > t_left) & (swing_lows.index < t_head)]
        lows_between_2 = swing_lows[(swing_lows.index > t_head) & (swing_lows.index < t_right)]
        if lows_between_1.empty or lows_between_2.empty:
            continue
        neckline = (float(lows_between_1.min()) + float(lows_between_2.min())) / 2

        after = window[window.index > t_right]
        confirmed = not after.empty and float(after["close"].iloc[-1]) < neckline

        patterns.append(ChartPattern(
            pattern_type=PatternType.HEAD_AND_SHOULDERS,
            confirmed=confirmed,
            neckline_price=neckline,
            key_points=[(t_left, p_left), (t_head, p_head), (t_right, p_right)],
            reasoning=[
                f"Left shoulder {p_left:.2f}, head {p_head:.2f} "
                f"(+{head_prominence:.1f}% above shoulders), right shoulder {p_right:.2f}",
                f"Neckline at {neckline:.2f}",
                "Neckline broken -- pattern CONFIRMED" if confirmed
                else "Neckline not yet broken -- pattern still forming, not tradeable yet",
            ],
        ))

    # Inverse Head & Shoulders: mirror, using swing lows.
    low_times = list(swing_lows.index)
    for i in range(len(low_times) - 2):
        t_left, t_head, t_right = low_times[i], low_times[i + 1], low_times[i + 2]
        p_left, p_head, p_right = (
            float(swing_lows[t_left]), float(swing_lows[t_head]), float(swing_lows[t_right])
        )
        if not _similar(p_left, p_right, shoulder_similarity_pct):
            continue
        head_prominence = (min(p_left, p_right) - p_head) / p_head * 100
        if head_prominence < min_head_prominence_pct:
            continue

        highs_between_1 = swing_highs[(swing_highs.index > t_left) & (swing_highs.index < t_head)]
        highs_between_2 = swing_highs[(swing_highs.index > t_head) & (swing_highs.index < t_right)]
        if highs_between_1.empty or highs_between_2.empty:
            continue
        neckline = (float(highs_between_1.max()) + float(highs_between_2.max())) / 2

        after = window[window.index > t_right]
        confirmed = not after.empty and float(after["close"].iloc[-1]) > neckline

        patterns.append(ChartPattern(
            pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
            confirmed=confirmed,
            neckline_price=neckline,
            key_points=[(t_left, p_left), (t_head, p_head), (t_right, p_right)],
            reasoning=[
                f"Left shoulder {p_left:.2f}, head {p_head:.2f} "
                f"(-{head_prominence:.1f}% below shoulders), right shoulder {p_right:.2f}",
                f"Neckline at {neckline:.2f}",
                "Neckline broken -- pattern CONFIRMED" if confirmed
                else "Neckline not yet broken -- pattern still forming, not tradeable yet",
            ],
        ))

    return patterns
