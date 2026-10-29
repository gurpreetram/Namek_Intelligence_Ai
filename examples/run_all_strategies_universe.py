"""
Full-universe backtest of all 8 strategies -- the original 7 from
examples/run_universe_backtest.py PLUS rolling_range_breakout (new,
2026-08-17, see examples/rolling_range_breakout.py), its first full-
universe test. Two things distinguish this from the earlier baseline runs
whose numbers are quoted in this module's own comparison table below:

1. Uses data.storage.load_combined() (archive + rolling cache), not the
   ~29-day rolling-cache-only window every earlier full-universe run used
   -- the permanent archive now holds 45+ trading days (see examples/
   daily_archive.py), so this is a materially larger, more reliable
   sample per symbol, not just a re-run of the same data.
2. Universe is price-filtered (>= detectors.scanner.MIN_STOCK_PRICE, same
   floor the live scanner/Daily Picks use), same as examples/
   run_smart_money_verification_universe.py -- not literally "every
   cached symbol" the way the ORIGINAL run_universe_backtest.py was,
   since MIN_STOCK_PRICE was introduced after that baseline was recorded.

No verification_config is applied (matching run_universe_backtest.py's
original, on-record numbers) -- this measures each strategy's raw,
unverified edge, the same thing the baselines below measured.

CHECKPOINT/RESUME: same architecture as examples/run_smc_vwap_universe.py
(built after a real ~5-hour run died at 88.5% complete with nothing
persisted) -- every symbol's 8-strategy result is appended to
logs/all_strategies_universe_partial_results.jsonl (one JSON line per
symbol per strategy) IMMEDIATELY after that symbol finishes. On startup,
a symbol is only skipped if ALL 8 strategy lines are already present --
one missing line (a process killed mid-write) means the whole symbol is
redone, since run_multi() computes all 8 in one shared-resample pass
anyway. The final report always reads the FULL partial-results file, not
this process's in-memory results, so a run interrupted and resumed any
number of times still produces one correct final report.

Progress logs to logs/universe_run.log (same shared file every other
full-universe validation run in this project uses).

Run: python -m examples.run_all_strategies_universe
    python -m examples.run_all_strategies_universe --limit 400   # bounded test
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from datetime import date, timedelta

import pandas as pd

from backtest.engine import Trade, run_multi
from backtest.metrics import compute
from backtest.strategy_base import SignalDirection
from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage
from detectors.fibonacci_strategy import FibonacciStrategy
from detectors.reversal_strategy import ReversalDetection
from detectors.scanner import MIN_STOCK_PRICE
from detectors.smc_strategy import SmartMoneyStrategy
from detectors.trend_continuation_strategy import TrendContinuation
from detectors.validated_breakout_strategy import ValidatedBreakout
from detectors.vwap_strategy import VWAPStrategy
from examples.opening_range_breakout import OpeningRangeBreakout
from examples.rolling_range_breakout import RollingRangeBreakout

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "all_strategies_universe_partial_results.jsonl"

# Same construction as examples/run_universe_backtest.py's STRATEGY_BUILDERS
# -- identical parameters for the original 7, so this is a genuine "same
# strategy, bigger/different data" test, not a different config being
# compared. rolling_range_breakout uses its class defaults (range_minutes=30,
# volume_multiplier=1.5, see examples/rolling_range_breakout.py).
STRATEGY_BUILDERS = {
    "opening_range_breakout": (
        lambda: OpeningRangeBreakout(range_minutes=15, volume_multiplier=1.5), [1, 5, 15]),
    "rolling_range_breakout": (
        lambda: RollingRangeBreakout(range_minutes=30, volume_multiplier=1.5), [1, 5, 15]),
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

# Previously known full-universe baselines (rolling-cache-only window,
# recorded before the permanent archive existed / before it held this much
# history) -- quoted here so the final report can show old-vs-new directly,
# not left as a number only in a chat transcript.
KNOWN_BASELINES = {
    "opening_range_breakout": {"trades": 14468, "win_rate": 0.453, "profit_factor": 0.96, "avg_r": -0.02},
    "validated_breakout": {"trades": 7735, "win_rate": 0.329, "profit_factor": 1.03, "avg_r": -0.04},
    "trend_continuation": {"trades": 4003, "win_rate": 0.348, "profit_factor": 0.98, "avg_r": -0.04},
    "reversal_detection": {"trades": 253, "win_rate": 0.308, "profit_factor": 0.87, "avg_r": -0.09},
    "vwap_reclaim": {"trades": 10011, "win_rate": 0.374, "profit_factor": 0.99, "avg_r": 0.01},
    "smart_money": {"trades": 12609, "win_rate": 0.330, "profit_factor": 1.07, "avg_r": -0.03},
    "fibonacci_retracement": None,  # no specific prior number given, only "similar to SMC range"
    "rolling_range_breakout": None,  # brand new -- no prior full-universe baseline exists
}

# Full available history -- data.storage.load_combined() supplies archive +
# rolling cache, so unlike every earlier full-universe run here there's no
# ~29-day Upstox-window reason to clip the start date. 2020-01-01 is just a
# safely-early sentinel (the real earliest archived date is 2026-06-08 as of
# this writing) -- run_multi()'s own date filter does the real narrowing.
START_DATE = date(2020, 1, 1)


def _price_filtered_universe(log: logging.Logger) -> list[str]:
    """Every cached symbol whose last known close is >= MIN_STOCK_PRICE --
    reused from detectors.scanner, not a second threshold. Checks the
    ROLLING cache's latest bar (storage.load(), not load_combined()) for
    this filter specifically -- the archive's latest bar can lag behind
    the rolling cache's, and "is this symbol currently liquid enough to
    matter" should reflect the most recent price, not a stale archived one."""
    all_symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    kept = []
    excluded = 0
    for symbol in all_symbols:
        df = storage.load(symbol)
        if df.empty:
            kept.append(symbol)  # no rolling-cache price to judge by -- don't exclude blind
            continue
        if float(df["close"].iloc[-1]) < MIN_STOCK_PRICE:
            excluded += 1
        else:
            kept.append(symbol)
    log.info(f"Price filter (MIN_STOCK_PRICE={MIN_STOCK_PRICE:.2f}): {excluded} of {len(all_symbols)} "
              f"cached symbols excluded, {len(kept)} remain in the test universe.")
    return kept


def _run_one(task: tuple[str, date, date]) -> list[tuple[str, str, list[Trade]]]:
    """Worker: all 8 strategies against ONE symbol, sharing one
    load_combined() + resample pass via run_multi(). Must stay a plain
    module-level function so multiprocessing can pickle it into workers."""
    symbol, start_date, end_date = task
    specs = {name: (build(), tfs) for name, (build, tfs) in STRATEGY_BUILDERS.items()}
    try:
        df_1min = storage.load_combined(symbol)
        results_by_strategy = run_multi(symbol, specs, start_date, end_date, df_1min=df_1min)
    except Exception as e:
        print(f"  {symbol}: {e}")
        # Still returns all 8 strategy names (empty trade lists) so this
        # symbol is marked COMPLETE on resume -- a permanently-broken
        # symbol would otherwise be retried forever on every resume.
        return [(symbol, name, []) for name in STRATEGY_BUILDERS]
    return [(symbol, name, result.trades) for name, result in results_by_strategy.items()]


def _trade_to_dict(trade: Trade) -> dict:
    return {
        "symbol": trade.symbol, "direction": trade.direction.value,
        "entry_time": trade.entry_time.isoformat(), "entry_price": trade.entry_price,
        "stop_loss": trade.stop_loss, "target": trade.target, "reasoning": trade.reasoning,
        "confidence": trade.confidence, "strategy_name": trade.strategy_name,
        "exit_time": trade.exit_time.isoformat() if trade.exit_time is not None else None,
        "exit_price": trade.exit_price, "exit_reason": trade.exit_reason,
        "initial_stop_loss": trade.initial_stop_loss, "peak_price": trade.peak_price,
        "breakeven_moved": trade.breakeven_moved, "trailing_active": trade.trailing_active,
    }


def _trade_from_dict(d: dict) -> Trade:
    return Trade(
        symbol=d["symbol"], direction=SignalDirection(d["direction"]),
        entry_time=pd.Timestamp(d["entry_time"]), entry_price=d["entry_price"],
        stop_loss=d["stop_loss"], target=d["target"], reasoning=d["reasoning"],
        confidence=d["confidence"], strategy_name=d["strategy_name"],
        exit_time=pd.Timestamp(d["exit_time"]) if d["exit_time"] is not None else None,
        exit_price=d["exit_price"], exit_reason=d["exit_reason"],
        initial_stop_loss=d.get("initial_stop_loss"), peak_price=d.get("peak_price"),
        breakeven_moved=d.get("breakeven_moved", False), trailing_active=d.get("trailing_active", False),
    )


def _append_partial_results(symbol: str, results_by_strategy: dict[str, list[Trade]]) -> None:
    """One write() call for all 8 of this symbol's strategy lines -- keeps
    a crash from ever leaving a symbol half-written."""
    lines = []
    for strategy_name, trades in results_by_strategy.items():
        record = {"symbol": symbol, "strategy": strategy_name, "trades": [_trade_to_dict(t) for t in trades]}
        lines.append(json.dumps(record))
    PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL_RESULTS_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")


def _load_completed_symbols() -> dict[str, dict[str, list[Trade]]]:
    """{symbol: {strategy_name: trades}} for symbols with ALL 8 strategies
    already recorded. A symbol with fewer than 8 present (process died
    mid-write, or a corrupted line) is treated as NOT done and
    recomputed from scratch -- run_multi() computes all 8 in one call, so
    there's no way to redo just the missing ones anyway."""
    if not PARTIAL_RESULTS_PATH.exists():
        return {}
    by_symbol: dict[str, dict[str, list[Trade]]] = {}
    with open(PARTIAL_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                trades = [_trade_from_dict(t) for t in record["trades"]]
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            by_symbol.setdefault(record["symbol"], {})[record["strategy"]] = trades
    return {
        symbol: strategies for symbol, strategies in by_symbol.items()
        if set(strategies.keys()) == set(STRATEGY_BUILDERS.keys())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only test the first N price-filtered symbols (alphabetical). Omit for the full universe.")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()], force=True,
    )
    log = logging.getLogger("all_strategies_universe")
    log.info("=" * 70)
    log.info("ALL 8 STRATEGIES x FULL UNIVERSE (load_combined -- archive+rolling) -- RUN START (checkpoint/resume enabled)")
    log.info("=" * 70)

    all_symbols = _price_filtered_universe(log)
    if not all_symbols:
        log.info("No cached data found -- run data/bulk_fetch.py first.")
        return
    if args.limit is not None:
        all_symbols = all_symbols[:args.limit]

    end_date = date.today() - timedelta(days=1)

    # This host has 7.8GB total RAM (2.5GB free at the time this was
    # written) -- os.cpu_count() workers (8 here) all spawning near-
    # simultaneously (Windows multiprocessing uses "spawn", a full fresh
    # interpreter + pandas/numpy/OpenBLAS import per worker, not "fork")
    # hit a real "paging file too small" / OpenBLAS allocation failure
    # during pool startup in testing on this machine. It self-recovered
    # that one time, but this run is meant to survive UNATTENDED for
    # hours, so trading some speed for not risking repeated worker
    # crashes is the right call here -- capped at 4 regardless of core
    # count. Override via MAX_WORKERS env var if run on a machine with
    # more headroom.
    cores = min(os.cpu_count() or 1, int(os.environ.get("MAX_WORKERS", 4)))
    n_strategies = len(STRATEGY_BUILDERS)

    completed = _load_completed_symbols()
    already_done = [s for s in all_symbols if s in completed]
    remaining = [s for s in all_symbols if s not in completed]

    total_units = len(all_symbols) * n_strategies
    units_done = len(already_done) * n_strategies
    log.info(f"Resume check ({PARTIAL_RESULTS_PATH.name}): {len(already_done)}/{len(all_symbols)} "
             f"symbols already complete, skipping. {len(remaining)} remaining to process.")
    log.info(f"Backtesting {len(remaining)} remaining symbols x {n_strategies} strategies "
             f"= {len(remaining) * n_strategies} runs this session, {cores} CPU cores. "
             f"Overall progress starts at {units_done}/{total_units} units. "
             f"Date range: {START_DATE} to {end_date} (full load_combined() history).")

    if remaining:
        tasks = [(symbol, START_DATE, end_date) for symbol in remaining]
        progress_every = 80  # 80 symbols x 8 strategies = 640 units -- frequent enough for a multi-hour run
        last_bucket = units_done // progress_every
        start_time = time.monotonic()
        units_done_at_start = units_done

        with multiprocessing.Pool(processes=cores) as pool:
            for per_symbol_results in pool.imap_unordered(_run_one, tasks):
                symbol = per_symbol_results[0][0]
                results_by_strategy = {name: trades for _, name, trades in per_symbol_results}
                _append_partial_results(symbol, results_by_strategy)

                units_done += len(per_symbol_results)
                bucket = units_done // progress_every
                if bucket != last_bucket or units_done >= total_units:
                    last_bucket = bucket
                    elapsed = time.monotonic() - start_time
                    processed_this_run = units_done - units_done_at_start
                    remaining_units = total_units - units_done
                    eta_min = (elapsed / processed_this_run) * remaining_units / 60 if processed_this_run > 0 else float("nan")
                    log.info(f"{units_done}/{total_units} units done "
                             f"({units_done // n_strategies}/{len(all_symbols)} symbols), ETA {eta_min:.1f} min")

        elapsed = time.monotonic() - start_time
        log.info(f"This session's batch completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    else:
        log.info("All symbols already complete -- skipping straight to aggregation.")

    log.info("Aggregating final results from partial-results file...")
    all_completed = _load_completed_symbols()
    trades_by_strategy: dict[str, list[Trade]] = {name: [] for name in STRATEGY_BUILDERS}
    missing = []
    for symbol in all_symbols:
        strategies = all_completed.get(symbol)
        if strategies is None:
            missing.append(symbol)
            continue
        for strategy_name, trades in strategies.items():
            trades_by_strategy[strategy_name].extend(trades)
    if missing:
        log.warning(f"{len(missing)} symbols missing from partial results after this run "
                    f"(unexpected -- investigate): {missing[:20]}{'...' if len(missing) > 20 else ''}")

    log.info("=" * 70)
    log.info("ALL 8 STRATEGIES: FULL-UNIVERSE RESULTS (load_combined -- archive+rolling cache)")
    log.info("=" * 70)
    reports = {}
    for name in STRATEGY_BUILDERS:
        report = compute(trades_by_strategy[name])
        reports[name] = report
        log.info(f"--- {name} ---")
        for line in str(report).split("\n"):
            log.info(line)

    log.info("=" * 70)
    log.info("SIDE BY SIDE -- ALL 8 STRATEGIES")
    log.info("=" * 70)
    header = f"{'Metric':22s} " + " ".join(f"{name[:16]:>17s}" for name in STRATEGY_BUILDERS)
    log.info(header)
    rows = [
        ("Trades", lambda r: f"{r.total_trades}"),
        ("Win rate", lambda r: f"{r.win_rate:.1%}"),
        ("Profit factor", lambda r: f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"),
        ("Avg R-multiple", lambda r: f"{r.avg_r_multiple:+.2f}"),
        ("Sharpe", lambda r: f"{r.sharpe_ratio:.2f}"),
        ("Max drawdown", lambda r: f"{r.max_drawdown_points:.1f}"),
    ]
    for label, fmt in rows:
        vals = " ".join(f"{fmt(reports[name]):>17s}" for name in STRATEGY_BUILDERS)
        log.info(f"{label:22s} {vals}")

    log.info("=" * 70)
    log.info("OLD (rolling-cache-only window) vs NEW (full archive+rolling, load_combined())")
    log.info("=" * 70)
    for name in STRATEGY_BUILDERS:
        baseline = KNOWN_BASELINES.get(name)
        r = reports[name]
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"
        if baseline is None:
            log.info(f"{name}: NO PRIOR FULL-UNIVERSE BASELINE -- new trades={r.total_trades} "
                      f"WR={r.win_rate:.1%} PF={pf_str} avgR={r.avg_r_multiple:+.2f} Sharpe={r.sharpe_ratio:.2f}")
        else:
            log.info(f"{name}:")
            log.info(f"    OLD: trades={baseline['trades']} WR={baseline['win_rate']:.1%} "
                      f"PF={baseline['profit_factor']:.2f} avgR={baseline['avg_r']:+.2f}")
            log.info(f"    NEW: trades={r.total_trades} WR={r.win_rate:.1%} "
                      f"PF={pf_str} avgR={r.avg_r_multiple:+.2f} Sharpe={r.sharpe_ratio:.2f}")
    log.info("RUN COMPLETE")


if __name__ == "__main__":
    main()
