"""
Checks whether the confidence scores our strategies output actually MEAN
anything -- i.e., do trades scored 0.75 confidence actually win around
75% of the time, or is 0.75 just an arbitrary number from "3 of 4 factors
passed" with no real connection to outcome frequency?

This is the difference between a confidence score and a genuine
probability. Right now, every strategy's confidence is an ad-hoc combination
of factor counts (see breakout_validator.py, pullback_classifier.py) --
reasonable as a relative ranking (more factors agreeing = more trustworthy),
but NOT validated as an actual probability until checked against real
outcomes. This module does that check, and produces a calibration mapping
that converts raw ad-hoc confidence into an empirically grounded probability.

Method: bin trades by their confidence score, compute the ACTUAL win rate
within each bin. A well-calibrated system shows bins tracking the diagonal
(0.7-0.8 confidence bin has roughly 70-80% actual win rate). A poorly
calibrated system shows flat or inverted bins (confidence doesn't predict
outcome at all, or even predicts it backwards) -- which is itself an
important, honest finding, not a failure to hide.

CRITICAL LIMITATION: this requires a meaningfully large sample to mean
anything -- a handful of trades per bin produces a "calibration curve"
that's mostly noise. Standard practice wants at least 30+ trades per bin
before trusting the calibrated number; fewer than that, treat the result
as preliminary at best. Given the project's current data situation (small
free-data samples), THIS MODULE IS INFRASTRUCTURE FOR LATER, MOSTLY --
it becomes genuinely useful once Upstox's full historical data is flowing
and backtests produce hundreds of trades, not a handful.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backtest.engine import Trade


@dataclass
class CalibrationBin:
    confidence_range: tuple[float, float]
    trade_count: int
    actual_win_rate: float
    avg_confidence: float
    reliable: bool  # whether trade_count meets the minimum sample threshold


@dataclass
class CalibrationReport:
    bins: list[CalibrationBin]
    brier_score: float  # lower is better; 0 = perfect, 0.25 = no better than always guessing 50%
    overall_trade_count: int
    reasoning: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Calibration report ({self.overall_trade_count} trades total, Brier score {self.brier_score:.3f}):"]
        for b in self.bins:
            reliability_flag = "" if b.reliable else "  [LOW SAMPLE -- treat as noisy]"
            lines.append(
                f"  Confidence {b.confidence_range[0]:.2f}-{b.confidence_range[1]:.2f}: "
                f"{b.trade_count} trades, actual win rate {b.actual_win_rate:.1%}, "
                f"avg confidence {b.avg_confidence:.2f}{reliability_flag}"
            )
        return "\n".join(lines)


class CalibrationMap:
    """
    Stores the bin -> actual-win-rate mapping learned from calibrate(),
    and lets you convert a NEW raw confidence score into the empirically
    grounded probability for future signals -- e.g. "this strategy said
    0.8 confidence, but historically 0.8-confidence signals from this
    strategy actually won 62% of the time, so trust 0.62, not 0.8."
    """

    def __init__(self, bins: list[CalibrationBin]):
        self._bins = sorted(bins, key=lambda b: b.confidence_range[0])

    def calibrate(self, raw_confidence: float) -> float:
        for b in self._bins:
            lo, hi = b.confidence_range
            if lo <= raw_confidence <= hi:
                return b.actual_win_rate if b.reliable else raw_confidence
        # Outside all observed bins -- fall back to the raw score since
        # we have no empirical evidence either way.
        return raw_confidence


def calibrate(trades: list[Trade], n_bins: int = 5, min_reliable_sample: int = 30) -> CalibrationReport:
    """
    trades: closed trades from a backtest run, each carrying the
        confidence score its originating Signal had. Only meaningful
        across MANY trades from the SAME strategy (or a deliberately
        pooled set you've decided is comparable) -- mixing confidence
        scores from strategies with different scoring logic muddies
        the result, since a 0.7 from one strategy's factor-counting
        isn't necessarily comparable to a 0.7 from another's.
    """
    closed = [t for t in trades if not t.is_open]
    if not closed:
        return CalibrationReport([], 0.0, 0, ["No closed trades to calibrate against."])

    confidences = np.array([t.confidence for t in closed])
    outcomes = np.array([1.0 if t.pnl_points > 0 else 0.0 for t in closed])

    brier_score = float(np.mean((confidences - outcomes) ** 2))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        win_rate = float(outcomes[mask].mean())
        avg_conf = float(confidences[mask].mean())
        bins.append(CalibrationBin(
            confidence_range=(float(lo), float(hi)),
            trade_count=count,
            actual_win_rate=win_rate,
            avg_confidence=avg_conf,
            reliable=count >= min_reliable_sample,
        ))

    reasoning = [
        f"Brier score: {brier_score:.3f} (0 = perfect calibration, 0.25 = no better than "
        f"a coin flip guess, closer to 0 is better)",
    ]
    low_sample_bins = [b for b in bins if not b.reliable]
    if low_sample_bins:
        reasoning.append(
            f"{len(low_sample_bins)} of {len(bins)} bins have fewer than "
            f"{min_reliable_sample} trades -- treat those bins' win rates as "
            f"preliminary, not reliable, until more backtest history is available."
        )

    return CalibrationReport(bins=bins, brier_score=brier_score,
                              overall_trade_count=len(closed), reasoning=reasoning)
