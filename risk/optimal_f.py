"""
Ralph Vince's Optimal-F: the fraction of capital, risked per trade, that
maximizes the GEOMETRIC growth rate of equity over a historical sequence
of trade outcomes -- contrasted in risk/position_sizing.py's own
docstring as "more aggressive" than that module's fixed-percentage
approach. Not currently used anywhere in this project; this module exists
purely to let examples/run_optimal_f_smc.py compare position-sizing
METHODS on an already-decided trade sequence, not to generate new signals.

METHODOLOGY NOTE (important, deliberately different from a naive port of
Vince's original paper): Vince's classic formula divides each raw DOLLAR
trade P&L by the single worst dollar loss in the sample, converting
un-normalized P&L into a comparable unit before searching for f. Our
input here is already a sequence of R-MULTIPLES -- each trade already
normalized to ITS OWN initial risk, exactly matching how risk/
position_sizing.py's calculate_position_size() already works (every
trade risks the same fraction of CURRENT capital, regardless of that
trade's own stop distance). Re-dividing by the sample's single worst R
again would double-normalize, and f would stop meaning "fraction of
capital risked per trade" -- breaking any apples-to-apples comparison
against the existing fixed-fractional approach. So the formula used here
is the direct, standard adaptation for a PRE-NORMALIZED R-multiple
sequence: HPR_i(f) = 1 + f*R_i (no division by the worst trade), and
optimal-f is the f that maximizes the Terminal Wealth Relative,
TWR(f) = product(HPR_i(f)), found by a fine grid search rather than a
closed-form solve (Vince's own method is iterative/numerical too -- no
closed form exists for an arbitrary empirical trade distribution).

f is bounded above by 1/|worst_R| (exclusive): the single worst trade in
the sample, sized at f, would cost exactly f*|worst_R| of capital, and at
f = 1/|worst_R| that trade alone wipes out 100% of equity. A sample with
no losing trades at all has no such bound from this formula -- capped
defensively at 0.99 in that case (flagged via f_max returned as 0.99),
since a real trade sequence with truly zero downside is not a
credible assumption to extrapolate an unbounded f from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptimalFResult:
    optimal_f: float
    half_optimal_f: float
    twr_at_optimal: float
    f_max: float
    worst_r: float
    n_trades: int


def compute_optimal_f(r_multiples: np.ndarray, grid_step: float = 0.001) -> OptimalFResult:
    """
    r_multiples: a sequence of trade outcomes already expressed as
    R-multiples (pnl / initial risk), in any order -- TWR at a given f
    IS order-dependent (compounding), but the argmax f itself is not,
    since product() is commutative; callers wanting an order-faithful
    equity curve should still pass trades in their real chronological
    sequence for consistency with how the result gets used downstream.
    """
    r = np.asarray(r_multiples, dtype=float)
    if r.size == 0:
        raise ValueError("r_multiples must be non-empty")

    worst_r = float(r.min())
    if worst_r >= 0:
        f_max = 0.99  # no losing trades in this sample -- see module docstring
    else:
        f_max = max(0.0, min(0.99, (1.0 / abs(worst_r)) - grid_step))

    f_grid = np.arange(0.0, f_max, grid_step) if f_max > 0 else np.array([0.0])

    best_f, best_twr = 0.0, 1.0  # f=0 (never trade) is always a valid baseline: TWR=1.0 (no change)
    for f in f_grid:
        hpr = 1.0 + f * r
        if np.any(hpr <= 0):
            continue
        twr = float(np.prod(hpr))
        if twr > best_twr:
            best_twr, best_f = twr, float(f)

    return OptimalFResult(
        optimal_f=best_f, half_optimal_f=best_f / 2.0, twr_at_optimal=best_twr,
        f_max=f_max, worst_r=worst_r, n_trades=int(r.size),
    )
