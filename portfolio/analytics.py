"""
Portfolio Analytics: aggregates performance across the whole trade
journal (paper_trading/journal.py) -- over time, across symbols, and
across strategies. This is different from backtest/metrics.py, which
reports on a SINGLE backtest run in isolation. Portfolio Analytics
answers a different question: "across everything this system has done
(or paper-traded) so far, what's actually working, and where is the
capital going?"

Three views this module builds:
  - Equity curve: cumulative R-multiple over time -- the shape of this
    curve matters as much as the endpoint. A smooth upward curve is a
    fundamentally different (better) risk profile than a jagged one that
    ends at the same total, even with identical final numbers.
  - Per-symbol breakdown: some symbols may carry a strategy's overall
    numbers while others quietly lose -- aggregate metrics can hide this.
  - Per-strategy breakdown: directly answers "which of the 4 strategies
    built so far (ORB, Validated Breakout, Trend Continuation, Reversal
    Detection) is actually contributing edge" -- this is the practical
    payoff of having added strategy_name to Trade/journal.

Same honest caveat as everywhere else in this project: these views are
only as meaningful as the sample size behind them. A per-symbol or
per-strategy slice with a handful of trades tells you much less than the
same slice with hundreds -- this module doesn't hide small samples, it
flags them (see min_reliable_sample below).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from paper_trading.journal import TradeJournal


@dataclass
class GroupBreakdown:
    group_name: str
    trade_count: int
    win_rate: float
    total_r: float
    avg_r: float
    reliable: bool


@dataclass
class PortfolioReport:
    total_trades: int
    total_r: float
    overall_win_rate: float
    equity_curve: pd.Series  # cumulative R-multiple, indexed by exit_time
    by_symbol: list[GroupBreakdown]
    by_strategy: list[GroupBreakdown]
    max_drawdown_r: float
    reasoning: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Portfolio: {self.total_trades} trades, total {self.total_r:+.2f}R, "
            f"win rate {self.overall_win_rate:.1%}, max drawdown {self.max_drawdown_r:.2f}R",
            "",
            "By strategy:",
        ]
        for g in self.by_strategy:
            flag = "" if g.reliable else "  [LOW SAMPLE]"
            lines.append(f"  {g.group_name}: {g.trade_count} trades, "
                          f"{g.win_rate:.1%} win rate, {g.total_r:+.2f}R total, "
                          f"{g.avg_r:+.2f}R avg{flag}")
        lines.append("")
        lines.append("By symbol:")
        for g in self.by_symbol:
            flag = "" if g.reliable else "  [LOW SAMPLE]"
            lines.append(f"  {g.group_name}: {g.trade_count} trades, "
                          f"{g.win_rate:.1%} win rate, {g.total_r:+.2f}R total{flag}")
        return "\n".join(lines)


def _breakdown_by(df: pd.DataFrame, column: str, min_reliable_sample: int) -> list[GroupBreakdown]:
    groups = []
    for name, group_df in df.groupby(column):
        wins = (group_df["r_multiple"] > 0).sum()
        count = len(group_df)
        groups.append(GroupBreakdown(
            group_name=str(name),
            trade_count=int(count),
            win_rate=float(wins / count) if count > 0 else 0.0,
            total_r=float(group_df["r_multiple"].sum()),
            avg_r=float(group_df["r_multiple"].mean()),
            reliable=count >= min_reliable_sample,
        ))
    return sorted(groups, key=lambda g: g.total_r, reverse=True)


def build_portfolio_report(
    journal: TradeJournal | None = None,
    min_reliable_sample: int = 20,
) -> PortfolioReport | None:
    """
    Builds a full portfolio-level report from the trade journal's history
    of exited trades. Returns None if the journal has no closed trades yet.
    """
    journal = journal or TradeJournal()
    df = journal.load_all()
    if df.empty:
        return None

    exits = df[df["event"] == "trade_exited"].copy()
    if exits.empty:
        return None

    exits["exit_time"] = pd.to_datetime(exits["exit_time"])
    exits = exits.sort_values("exit_time")

    total_trades = len(exits)
    total_r = float(exits["r_multiple"].sum())
    win_rate = float((exits["r_multiple"] > 0).mean())

    equity_curve = exits.set_index("exit_time")["r_multiple"].cumsum()
    running_max = equity_curve.cummax()
    drawdown = running_max - equity_curve
    max_drawdown_r = float(drawdown.max()) if not drawdown.empty else 0.0

    by_symbol = _breakdown_by(exits, "symbol", min_reliable_sample)
    by_strategy = (
        _breakdown_by(exits, "strategy_name", min_reliable_sample)
        if "strategy_name" in exits.columns else []
    )

    reasoning = [
        f"Built from {total_trades} closed trades in the journal.",
    ]
    low_sample_strategies = [g for g in by_strategy if not g.reliable]
    if low_sample_strategies:
        reasoning.append(
            f"{len(low_sample_strategies)} strateg{'y' if len(low_sample_strategies)==1 else 'ies'} "
            f"have fewer than {min_reliable_sample} trades -- their win rate/R numbers "
            f"are preliminary, not yet reliable enough to act on alone."
        )

    return PortfolioReport(
        total_trades=total_trades,
        total_r=total_r,
        overall_win_rate=win_rate,
        equity_curve=equity_curve,
        by_symbol=by_symbol,
        by_strategy=by_strategy,
        max_drawdown_r=max_drawdown_r,
        reasoning=reasoning,
    )
