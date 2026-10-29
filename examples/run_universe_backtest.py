"""
Runs all four strategies against every symbol that currently has cached
1-minute data (data/storage.py's per-symbol Parquet files) -- the "scale
up from 5 hardcoded stocks" step. Each SYMBOL's backtest is independent of
every other symbol's, so this parallelizes trivially across CPU cores via
multiprocessing.Pool rather than looping serially like
examples/run_all_detectors_upstox.py does.

Task granularity is per-SYMBOL, not per-(symbol, strategy): all 4 strategies
share the same 1-minute data and the same 5-min/15-min resampling for a given
symbol (via backtest.engine.run_multi), computed once and reused, instead of
each strategy reloading the symbol from disk and rebuilding the resample from
scratch. 3 of the 4 strategies use overlapping timeframe sets ([1,5,15],
[1,5,15], [1,15]), so without this the 15-min resample alone was being
computed independently 3 times per symbol for identical output -- a real,
measured cost, not a hypothetical one (see backtest/engine.py's run_multi
docstring).

The whole point of scaling up isn't just a bigger blended number -- it's
finding out whether a strategy has real edge on SOME stocks even if the
universe-wide average is unimpressive (or vice versa: a decent-looking
blended number hiding the fact that it's carried by 3 symbols and a loss
everywhere else). That's why this prints a per-symbol breakdown sorted by
total R-multiple, not just the aggregate PerformanceReport.

Honest caveat carried over from run_all_detectors_upstox.py: even a wide
symbol universe over Upstox's ~30-day 1-minute data window is still a
modest sample per symbol (and per strategy-per-symbol, once you slice
that finely) -- treat the per-symbol breakdown as "where to look closer,"
not a final verdict on any individual stock.

Run: python -m examples.run_universe_backtest
    python -m examples.run_universe_backtest --limit 400   # bounded batch (alphabetical)
(assumes data/bulk_fetch.py has already populated storage/candles/)
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from datetime import date, timedelta

from backtest.engine import Trade, run_multi
from backtest.metrics import compute
from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage
from detectors.fibonacci_strategy import FibonacciStrategy
from detectors.reversal_strategy import ReversalDetection
from detectors.scanner import scan_universe
from detectors.smc_strategy import SmartMoneyStrategy
from detectors.trend_continuation_strategy import TrendContinuation
from detectors.validated_breakout_strategy import ValidatedBreakout
from detectors.vwap_strategy import VWAPStrategy
from examples.opening_range_breakout import OpeningRangeBreakout
from reporting.html_report import SymbolBreakdown, generate_report

# Same construction as examples/run_all_detectors_upstox.py -- these are
# the actually-backtested parameters, not loosened demo values. Builders
# (not instances) so each worker process constructs its own fresh strategy
# rather than one instance's state being (incorrectly) shared/pickled
# across processes.
STRATEGY_BUILDERS = {
    "opening_range_breakout": (
        lambda: OpeningRangeBreakout(range_minutes=15, volume_multiplier=1.5), [1, 5, 15]),
    "validated_breakout": (
        lambda: ValidatedBreakout(level_timeframe=15, min_touches=2), [1, 5, 15]),
    "trend_continuation": (
        lambda: TrendContinuation(trend_lookback_bars=20, pullback_bars=2, min_confidence=0.75), [1, 15]),
    "reversal_detection": (
        lambda: ReversalDetection(trend_lookback_bars=20, pullback_bars=2, min_confidence=0.6), [1]),
    "vwap_reclaim": (
        lambda: VWAPStrategy(volume_lookback_bars=20, volume_multiplier=1.3, stop_lookback_bars=15), [1]),
    "smart_money": (
        lambda: SmartMoneyStrategy(order_block_lookback_bars=50, move_atr_mult=1.5), [1]),
    "fibonacci_retracement": (
        lambda: FibonacciStrategy(swing_lookback_bars=50, min_leg_atr_mult=1.5), [1]),
}

STRATEGY_DISPLAY_NAMES = {
    "opening_range_breakout": "Opening Range Breakout (baseline)",
    "validated_breakout": "Validated Breakout",
    "trend_continuation": "Trend Continuation",
    "reversal_detection": "Reversal Detection",
    "vwap_reclaim": "VWAP Reclaim (new, untested)",
    "smart_money": "Smart Money Concepts -- order block retest (new, untested)",
    "fibonacci_retracement": "Fibonacci Retracement (new, untested)",
}


def _cached_symbols() -> list[str]:
    return sorted(
        p.name.removesuffix("_1min.parquet")
        for p in STORAGE_ROOT.glob("*_1min.parquet")
    )


def _run_one(task: tuple[str, date, date]) -> list[tuple[str, str, list[Trade]]]:
    """
    Worker function: runs ALL 4 strategies against ONE symbol via
    run_multi(), sharing that symbol's one-time data load and resampling
    across them. Must stay a plain module-level function (not a closure)
    so multiprocessing can pickle/import it into worker processes.

    Returns one (symbol, strategy_name, trades) tuple per strategy -- same
    shape the caller previously got one-task-at-a-time, just batched 4-per-
    symbol now so downstream aggregation code doesn't need to change.
    """
    symbol, start_date, end_date = task
    specs = {name: (build(), tfs) for name, (build, tfs) in STRATEGY_BUILDERS.items()}
    try:
        results_by_strategy = run_multi(symbol, specs, start_date, end_date)
    except Exception as e:
        print(f"  {symbol}: {e}")
        return [(symbol, name, []) for name in STRATEGY_BUILDERS]
    return [(symbol, name, result.trades) for name, result in results_by_strategy.items()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest all 4 strategies against cached symbols."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only backtest the first N cached symbols (alphabetical). "
             "Omit to run every cached symbol.",
    )
    args = parser.parse_args()

    symbols = _cached_symbols()
    if not symbols:
        print("No cached data found in storage/candles/ -- run "
              "`python -m data.bulk_fetch` first.")
        return
    if args.limit is not None:
        symbols = symbols[:args.limit]

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)  # Upstox's actual 1-min data limit

    cores = os.cpu_count() or 1
    n_strategies = len(STRATEGY_BUILDERS)
    # One task per SYMBOL now (not per symbol-strategy pair) -- run_multi()
    # backtests all 4 strategies against that symbol's data in one shared
    # load+resample pass. See module docstring for why this replaced the
    # old one-task-per-(symbol,strategy) design.
    tasks = [(symbol, start_date, end_date) for symbol in symbols]
    total_units = len(tasks) * n_strategies  # for progress reporting/ETA only
    print(f"Backtesting {len(symbols)} cached symbols x {n_strategies} strategies "
          f"= {total_units} strategy-runs across {len(tasks)} per-symbol tasks, "
          f"{cores} CPU cores...\n")

    # imap_unordered (not map): map() blocks silently until EVERY task is
    # done before returning anything, which gives zero visibility into a
    # run that can take tens of minutes -- no way to tell "still working"
    # from "stalled" short of checking process/CPU state from outside.
    # imap_unordered yields each result as soon as ANY worker finishes it
    # (order doesn't matter here -- results are keyed by symbol/strategy
    # below, not position), so progress can be reported as it happens.
    # Progress is reported in strategy-run units (e.g. "300/1600 done") even
    # though the underlying pool only yields once per SYMBOL (4 units at a
    # time), so the reported total still matches what you'd see per-strategy.
    raw_results = []
    units_done = 0
    progress_every = 100
    last_reported_bucket = -1
    start_time = time.monotonic()

    with multiprocessing.Pool(processes=cores) as pool:
        for per_symbol_results in pool.imap_unordered(_run_one, tasks):
            raw_results.extend(per_symbol_results)
            units_done += len(per_symbol_results)
            bucket = units_done // progress_every
            if bucket != last_reported_bucket or units_done >= total_units:
                last_reported_bucket = bucket
                elapsed = time.monotonic() - start_time
                eta_min = (elapsed / units_done) * (total_units - units_done) / 60
                print(f"{units_done}/{total_units} done, ETA {eta_min:.1f} min", flush=True)

    trades_by_strategy: dict[str, list[Trade]] = {name: [] for name in STRATEGY_BUILDERS}
    trades_by_strategy_symbol: dict[str, dict[str, list[Trade]]] = {name: {} for name in STRATEGY_BUILDERS}
    for symbol, strategy_name, trades in raw_results:
        trades_by_strategy[strategy_name].extend(trades)
        trades_by_strategy_symbol[strategy_name][symbol] = trades

    strategy_results = {}
    per_symbol_breakdown = {}
    comparison_rows = []

    for strategy_name, display_name in STRATEGY_DISPLAY_NAMES.items():
        all_trades = trades_by_strategy[strategy_name]
        report = compute(all_trades)
        strategy_results[display_name] = (all_trades, report)

        breakdown: list[SymbolBreakdown] = []
        for symbol, trades in trades_by_strategy_symbol[strategy_name].items():
            closed = [t for t in trades if not t.is_open]
            if not closed:
                continue
            sym_report = compute(closed)
            total_r = float(sum(t.r_multiple for t in closed))
            breakdown.append(SymbolBreakdown(
                symbol=symbol, trades=sym_report.total_trades,
                win_rate=sym_report.win_rate, total_r=total_r, avg_r=sym_report.avg_r_multiple,
            ))
        breakdown.sort(key=lambda b: b.total_r, reverse=True)
        per_symbol_breakdown[display_name] = breakdown

        print(f"\n{'=' * 60}\n{display_name}\n{'=' * 60}")
        if all_trades:
            print(report)
            print("\nPer-symbol (top 10 by total R):")
            for b in breakdown[:10]:
                print(f"  {b.symbol:12s} trades={b.trades:3d}  win_rate={b.win_rate:.1%}  "
                      f"total_R={b.total_r:+.2f}  avg_R={b.avg_r:+.2f}")
            if len(breakdown) > 10:
                print("  ...")
                print("  Bottom 5 by total R:")
                for b in breakdown[-5:]:
                    print(f"  {b.symbol:12s} trades={b.trades:3d}  win_rate={b.win_rate:.1%}  "
                          f"total_R={b.total_r:+.2f}  avg_R={b.avg_r:+.2f}")
        else:
            print("No trades generated across the universe for this strategy.")

        comparison_rows.append((display_name, report))

    print(f"\n{'=' * 60}\nCOMPARISON SUMMARY\n{'=' * 60}")
    print(f"{'Strategy':32s} {'Trades':>8s} {'WinRate':>8s} {'PF':>7s} {'AvgR':>7s} {'Sharpe':>7s}")
    for display_name, report in comparison_rows:
        pf_str = f"{report.profit_factor:.2f}" if report.profit_factor != float("inf") else "inf"
        print(f"{display_name:32s} {report.total_trades:8d} {report.win_rate:7.1%}  "
              f"{pf_str:>6s} {report.avg_r_multiple:7.2f} {report.sharpe_ratio:7.2f}")

    print(f"\n{'=' * 60}\nScanner (live-style scan as of the most recent cached bar)\n{'=' * 60}")
    scan_data = {symbol: storage.load(symbol) for symbol in symbols}
    scan_data = {s: df for s, df in scan_data.items() if not df.empty}
    scanner_results: list = []
    if scan_data:
        most_recent = max(df.index.max() for df in scan_data.values())
        scanner_results = scan_universe(scan_data, most_recent, min_score=0.5)
        if not scanner_results:
            print(f"No qualifying setups as of {most_recent}.")
        for r in scanner_results:
            print(f"{r.symbol:10s} score={r.score:.2f}  setup={r.setup_type:20s} direction={r.direction}")

    report_path = PROJECT_ROOT / "reporting" / "latest_report.html"
    generate_report(
        report_path, strategy_results, scanner_results,
        data_source_note=f"Real Upstox data, {start_date} to {end_date} "
                          f"({(end_date - start_date).days} days, {len(symbols)} symbols -- "
                          f"full cached universe, run in parallel across {cores} cores).",
        per_symbol_breakdown=per_symbol_breakdown,
    )
    print(f"\nHTML report written to: {report_path}")


if __name__ == "__main__":
    main()
