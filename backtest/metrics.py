"""
Standard trading performance metrics computed from a list of closed Trades.
These are the numbers your own brief specified as required before anything
is "complete": win rate, profit factor, max drawdown, Sharpe, expectancy,
average R-multiple.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backtest.engine import Trade


@dataclass
class PerformanceReport:
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_points: float
    sharpe_ratio: float
    expectancy_points: float
    avg_r_multiple: float
    avg_win_points: float
    avg_loss_points: float

    def __str__(self) -> str:
        return (
            f"Trades: {self.total_trades}\n"
            f"Win rate: {self.win_rate:.1%}\n"
            f"Profit factor: {self.profit_factor:.2f}\n"
            f"Max drawdown: {self.max_drawdown_points:.2f} pts\n"
            f"Sharpe ratio: {self.sharpe_ratio:.2f}\n"
            f"Expectancy: {self.expectancy_points:.3f} pts/trade\n"
            f"Avg R-multiple: {self.avg_r_multiple:.2f}\n"
            f"Avg win: {self.avg_win_points:.2f} pts | Avg loss: {self.avg_loss_points:.2f} pts"
        )


def compute(trades: list[Trade]) -> PerformanceReport:
    closed = [t for t in trades if not t.is_open]
    if not closed:
        return PerformanceReport(0, 0, 0, 0, 0, 0, 0, 0, 0)

    pnls = np.array([t.pnl_points for t in closed])
    r_multiples = np.array([t.r_multiple for t in closed])

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / len(closed) if closed else 0.0
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity_curve = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0

    # Sharpe on a per-trade basis (not annualized -- annualizing per-trade
    # intraday returns without knowing trade frequency would be misleading).
    sharpe = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0

    expectancy = float(pnls.mean())
    avg_r = float(r_multiples.mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    return PerformanceReport(
        total_trades=len(closed),
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_points=max_drawdown,
        sharpe_ratio=sharpe,
        expectancy_points=expectancy,
        avg_r_multiple=avg_r,
        avg_win_points=avg_win,
        avg_loss_points=avg_loss,
    )


@dataclass
class EquityCurveResult:
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float


def simulate_fixed_fraction_equity_curve(r_multiples: np.ndarray, f: float) -> EquityCurveResult:
    """
    Compounds a starting unit of capital across a trade SEQUENCE (order
    matters -- this is compounding, not a plain average), risking a
    constant fraction f of CURRENT equity on each trade: equity *=
    (1 + f*R_i). Same per-trade-risk-of-current-capital convention risk/
    position_sizing.py's calculate_position_size() already uses (risking
    f% of capital regardless of a given trade's own stop distance, since
    R_i is already normalized to that trade's own risk -- see
    risk/optimal_f.py's module docstring for why this is the correct
    normalization to compare against that existing approach).

    Mirrors compute()'s own cumsum/running-max drawdown pattern above,
    adapted from an additive points-based curve to a multiplicative
    (compounding) percentage one -- position-sizing-by-fraction is
    inherently multiplicative (each trade's win/loss is a % of whatever
    capital exists AT THAT POINT, not a fixed point value), so cumsum
    itself isn't the right primitive here, only the same running-max-vs-
    current drawdown IDEA is reused.
    """
    r = np.asarray(r_multiples, dtype=float)
    if r.size == 0:
        return EquityCurveResult(total_return_pct=0.0, max_drawdown_pct=0.0, sharpe=0.0)

    per_trade_returns = f * r
    equity_curve = np.cumprod(1.0 + per_trade_returns)
    total_return_pct = float((equity_curve[-1] - 1.0) * 100.0)

    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = (running_max - equity_curve) / running_max * 100.0
    max_drawdown_pct = float(drawdown_pct.max())

    sharpe = float(per_trade_returns.mean() / per_trade_returns.std()) if per_trade_returns.std() > 0 else 0.0

    return EquityCurveResult(total_return_pct=total_return_pct, max_drawdown_pct=max_drawdown_pct, sharpe=sharpe)
