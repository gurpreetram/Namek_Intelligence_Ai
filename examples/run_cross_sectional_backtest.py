"""
Backtests detectors/cross_sectional_ranking.py as a market-neutral daily
long/short basket strategy: each cached trading day, ranks the full
cached universe by return-from-open to a fixed formation cutoff (default
10:00, 45 min after the 09:15 open), goes LONG the top N ranked symbols
and SHORT the bottom N, entering at the next bar's open after the cutoff
(no lookahead -- same convention backtest/engine.py uses everywhere else)
and exiting at EOD square-off (15:15, same convention as every other
strategy in this project).

RISK-MANAGEMENT CONVENTION CHOSEN, and why: NO per-position stop-loss or
target. This is deliberately different from this project's other,
single-name directional strategies. For a diversified cross-sectional
basket (2N equal-weighted positions per day), the standard convention in
both academic factor-portfolio research and practitioner market-neutral
strategies is that diversification across many simultaneous positions IS
the risk control -- individual per-name stops are for concentrated
single-name bets, not a basket built to be balanced long vs short by
construction. Positions hold to a fixed exit (EOD square-off), matching
"a simple end-of-day exit" -- the alternative the task brief explicitly
offered as standard for this kind of strategy.

A synthetic 1x-ATR stop_loss IS still attached to each position's Trade
record -- NOT as an executed stop (no early exit happens because of it),
but purely so backtest.metrics.compute() can report an avg R-multiple
computed with the EXACT SAME methodology as every other strategy in this
project, for direct comparability. This is clearly labeled in the report.

HONEST SAMPLE-SIZE CAVEAT, restated in the printed report itself (not
just here): only as many distinct trading days exist as Upstox's ~29-30
day rolling 1-minute cache covers. This is a SMALL sample for evaluating
ANY strategy, and doubly so here: with ~2N=40 positions per day but only
~N_DAYS independent days, the DAY-LEVEL statistics (day win rate, Sharpe
on the daily return series) are what actually matter for judging
robustness -- the larger raw position count looks more statistically
solid than it really is, since positions within a day are not
independent observations (they share the same day's market conditions).

--reverse flips the direction assignment: SHORT the top-N ranked (morning
relative winners), LONG the bottom-N (morning relative losers) -- the
opposite of the default momentum-continuation bet. Added to directly test
the mean-reversion hypothesis the default run's result suggested (a
strong, consistent NEGATIVE return going long winners/short losers is
also evidence FOR the reverse trade, not just against the original one --
this flag tests that directly on the exact same universe/cutoff/exit
mechanics, changing nothing else).

TRANSACTION COSTS (added later -- a real, previously-unaddressed gap in
the original finding): each position's return is reported both GROSS
(the raw price move, as before) and NET of an estimated round-trip cost
via --cost-pct, default COST_PCT_PER_ROUND_TRIP below. This is a single
flat assumption covering brokerage + STT + exchange/GST charges +
slippage combined, NOT a itemized cost model -- Indian intraday equity
round-trip costs commonly fall in the 0.05-0.1% of turnover range for a
retail account; 0.075% (the midpoint) is used as the default, deliberately
conservative-but-plausible rather than picked to make the result look
better or worse. Applied identically to every position regardless of
direction (long or short) or --top-n concentration, since it's a
turnover-based cost, not a directional one.

Run: python -m examples.run_cross_sectional_backtest
    python -m examples.run_cross_sectional_backtest --top-n 20 --formation-cutoff 10:00
    python -m examples.run_cross_sectional_backtest --reverse
    python -m examples.run_cross_sectional_backtest --reverse --top-n 1 --cost-pct 0.075
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from backtest.engine import Trade
from backtest.metrics import compute
from backtest.strategy_base import SignalDirection
from config.settings import STORAGE_ROOT
from data import storage
from detectors.cross_sectional_ranking import bar_at_or_before, next_bar_after, rank_symbols_for_day
from detectors.support_resistance import average_true_range

EOD_SQUARE_OFF_TIME = "15:15"
MIN_STOP_DISTANCE_ATR_MULT = 1.0  # synthetic risk reference only -- see module docstring
# Estimated round-trip transaction cost (brokerage + STT + exchange/GST
# charges + slippage combined), as a % of notional -- see module
# docstring for the assumption and why 0.075% (the midpoint of the
# commonly-cited 0.05-0.1% range for Indian retail intraday equity) is
# the default rather than picked to flatter or sink the result.
COST_PCT_PER_ROUND_TRIP = 0.075


def _synthetic_risk(df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> float:
    truncated = df.loc[:cutoff_ts]
    atr = average_true_range(truncated.iloc[-30:]) if len(truncated) >= 2 else 0.0
    return atr * MIN_STOP_DISTANCE_ATR_MULT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--formation-cutoff", type=str, default="10:00")
    parser.add_argument("--limit", type=int, default=None, help="cap universe size, for a fast test run")
    parser.add_argument("--reverse", action="store_true",
                         help="SHORT the top-N morning winners, LONG the bottom-N morning losers "
                              "(mean-reversion bet) instead of the default momentum-continuation direction")
    parser.add_argument("--cost-pct", type=float, default=COST_PCT_PER_ROUND_TRIP,
                         help=f"Estimated round-trip transaction cost as %% of notional, subtracted from every "
                              f"position's gross return to get its net return (default {COST_PCT_PER_ROUND_TRIP}%%, "
                              f"see module docstring for the assumption). Pass 0 to see gross-only.")
    args = parser.parse_args()
    strategy_tag = "cross_sectional_ranking_reversed" if args.reverse else "cross_sectional_ranking"
    print(f"Mode: {'REVERSED (short winners / long losers -- mean-reversion bet)' if args.reverse else 'DEFAULT (long winners / short losers -- momentum-continuation bet)'}")
    print(f"Round-trip cost assumption: {args.cost_pct:.3f}% of notional per position "
          f"(brokerage + STT + exchange/GST + slippage combined -- see module docstring)\n")

    symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    if args.limit is not None:
        symbols = symbols[:args.limit]

    print(f"Loading {len(symbols)} cached symbols into memory...")
    start = time.monotonic()
    symbol_data: dict[str, pd.DataFrame] = {}
    for i, s in enumerate(symbols, 1):
        df = storage.load(s)
        if not df.empty:
            symbol_data[s] = df
        if i % 500 == 0:
            print(f"  loaded {i}/{len(symbols)}...", flush=True)
    print(f"Loaded {len(symbol_data)}/{len(symbols)} symbols with usable data in {time.monotonic() - start:.1f}s\n")

    all_days = sorted({d for df in symbol_data.values() for d in df.index.date})
    print(f"Distinct cached trading days found: {len(all_days)}  ({all_days[0]} .. {all_days[-1]})\n")

    all_trades: list[Trade] = []
    day_returns: list[tuple] = []  # (day, gross_return_pct, net_return_pct, n_positions, coverage)
    skipped_days = 0

    for day in all_days:
        ranked = rank_symbols_for_day(symbol_data, day, formation_cutoff=args.formation_cutoff)
        if len(ranked) < 2 * args.top_n:
            skipped_days += 1
            print(f"  {day}: only {len(ranked)} ranked symbols (< 2*top_n={2*args.top_n}), skipping day")
            continue

        top_ranked = ranked[:args.top_n]      # morning relative winners
        bottom_ranked = ranked[-args.top_n:]  # morning relative losers
        if args.reverse:
            longs, shorts = bottom_ranked, top_ranked  # LONG losers, SHORT winners -- mean-reversion bet
        else:
            longs, shorts = top_ranked, bottom_ranked  # LONG winners, SHORT losers -- momentum-continuation bet
        cutoff_time = pd.Timestamp(args.formation_cutoff).time()

        day_position_returns_gross = []
        day_position_returns_net = []
        for rs, direction in [(r, SignalDirection.LONG) for r in longs] + [(r, SignalDirection.SHORT) for r in shorts]:
            df = symbol_data[rs.symbol]
            sample_ts = df[df.index.date == day].index[0]
            cutoff_ts = pd.Timestamp.combine(day, cutoff_time).tz_localize(sample_ts.tz)

            entry_bar = next_bar_after(df, cutoff_ts)
            eod_ts = pd.Timestamp.combine(day, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(sample_ts.tz)
            exit_bar = bar_at_or_before(df, eod_ts)
            if entry_bar is None or exit_bar is None or entry_bar.name >= exit_bar.name:
                continue

            entry_price = float(entry_bar["open"])
            exit_price = float(exit_bar["close"])
            sign = 1.0 if direction == SignalDirection.LONG else -1.0
            pos_return_pct = sign * (exit_price - entry_price) / entry_price * 100.0
            # NET of the estimated round-trip cost -- a flat drag applied
            # once per position regardless of direction (see module
            # docstring/--cost-pct for the assumption).
            pos_return_pct_net = pos_return_pct - args.cost_pct
            day_position_returns_gross.append(pos_return_pct)
            day_position_returns_net.append(pos_return_pct_net)

            risk = _synthetic_risk(df, cutoff_ts)
            stop_loss = entry_price - sign * risk if risk > 0 else entry_price * 0.995
            target = entry_price + sign * risk * 2.0 if risk > 0 else entry_price * 1.01

            all_trades.append(Trade(
                symbol=rs.symbol, direction=direction, entry_time=entry_bar.name, entry_price=entry_price,
                stop_loss=stop_loss, target=target,
                reasoning=[f"Cross-sectional rank: {rs.return_pct:+.2f}% return-from-open by {args.formation_cutoff}"
                           + (" (reversed: faded, not followed)" if args.reverse else "")],
                confidence=0.5, strategy_name=strategy_tag,
                exit_time=exit_bar.name, exit_price=exit_price, exit_reason="eod_square_off",
            ))

        if day_position_returns_gross:
            day_returns.append((
                day, float(np.mean(day_position_returns_gross)), float(np.mean(day_position_returns_net)),
                len(day_position_returns_gross), len(ranked),
            ))

    print(f"\nDays used: {len(day_returns)}  /  Days skipped (insufficient ranked symbols): {skipped_days}\n")

    # ---- Position-level metrics (reuses backtest.metrics.compute() exactly, same methodology as every other strategy -- GROSS only, R-multiple is a synthetic-risk convention, not a %-cost one) ----
    report = compute(all_trades)
    print("=" * 70)
    print(f"POSITION-LEVEL METRICS (GROSS, R-multiple convention)  ({len(all_trades)} positions across {len(day_returns)} days)")
    print("=" * 70)
    print(report)

    pos_returns_gross = np.array([
        (t.exit_price - t.entry_price) / t.entry_price * 100.0 * (1 if t.direction == SignalDirection.LONG else -1)
        for t in all_trades
    ])
    pos_returns_net = pos_returns_gross - args.cost_pct
    print()
    print(f"POSITION-LEVEL %% RETURN, GROSS vs NET (cost={args.cost_pct:.3f}%%/position):")
    print(f"  GROSS -- win rate: {(pos_returns_gross > 0).mean():.1%}  avg: {pos_returns_gross.mean():+.3f}%  median: {np.median(pos_returns_gross):+.3f}%")
    print(f"  NET   -- win rate: {(pos_returns_net > 0).mean():.1%}  avg: {pos_returns_net.mean():+.3f}%  median: {np.median(pos_returns_net):+.3f}%")

    # ---- Day-level metrics (the statistically honest unit for this strategy shape -- see module docstring) ----
    day_gross_array = np.array([g for _, g, _, _, _ in day_returns])
    day_net_array = np.array([n for _, _, n, _, _ in day_returns])

    def _day_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
        win_rate = (arr > 0).mean() if len(arr) else 0.0
        sharpe = float(arr.mean() / arr.std()) if len(arr) > 1 and arr.std() > 0 else 0.0
        return win_rate, float(arr.mean()), float(arr.std()), sharpe

    gross_win_rate, gross_mean, gross_std, gross_sharpe = _day_stats(day_gross_array)
    net_win_rate, net_mean, net_std, net_sharpe = _day_stats(day_net_array)

    print()
    print("=" * 70)
    print(f"DAY-LEVEL METRICS  ({len(day_returns)} independent trading days -- THE REAL SAMPLE SIZE)")
    print("=" * 70)
    print(f"GROSS -- day win rate: {gross_win_rate:.1%}  avg daily return: {gross_mean:+.3f}%  "
          f"std: {gross_std:.3f}%  Sharpe (not annualized): {gross_sharpe:.2f}")
    print(f"NET   -- day win rate: {net_win_rate:.1%}  avg daily return: {net_mean:+.3f}%  "
          f"std: {net_std:.3f}%  Sharpe (not annualized): {net_sharpe:.2f}")
    print()
    print("Per-day detail (gross / net):")
    for day, gross, net, n_pos, coverage in day_returns:
        print(f"  {day}  gross={gross:+.3f}%  net={net:+.3f}%  positions={n_pos:2d}  ranked_universe={coverage}")

    print()
    print("=" * 70)
    print(f"SAMPLE-SIZE CAVEAT: results above are from {len(day_returns)} trading days of cached "
          f"1-minute data (Upstox's ~29-30 day rolling window). This is a SMALL sample -- treat "
          f"all findings here as preliminary, not validated conclusions.")
    print("=" * 70)


if __name__ == "__main__":
    main()
