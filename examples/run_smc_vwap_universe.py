"""
Full-universe validation for the two newest strategies -- Smart Money
Concepts (order-block retest, detectors/smc_strategy.py) and VWAP Reclaim
(detectors/vwap_strategy.py) -- both already backtested on a 400-symbol
subset via examples/run_universe_backtest.py:
    SMC:          2189 trades, PF 1.23, avg R -0.01
    VWAP Reclaim: 1742 trades, PF 0.92,  avg R +0.04
This script checks whether those numbers hold at the FULL 2373-symbol
scale, or flip the way TrendContinuation's did (avg R +0.06 -> -0.03 going
from a 400-symbol alphabetical subset to the full universe) -- that
confirmed alphabetical-subset bias is why NEITHER number above should be
assumed to hold without actually re-running at scale.

Both strategies share ONE load+resample pass per symbol via
backtest.engine.run_multi() (same reasoning as run_universe_backtest.py
and run_validated_breakout_variants.py: both only need timeframe [1], so
loading twice per symbol would be pure waste). Progress uses
imap_unordered + a running counter, not blocking pool.map().

CHECKPOINT/RESUME: same architecture as run_validated_breakout_variants.py
(built after a real ~5-hour run died at 88.5% complete with nothing
persisted). Every symbol's 2-strategy result is appended to
logs/smc_vwap_partial_results.jsonl (one JSON line per symbol per
strategy) IMMEDIATELY after that symbol finishes. On startup, any symbol
with BOTH strategies already present in that file is skipped. The final
report is always computed by reading the FULL partial-results file, not
this process's own in-memory results, so a run that dies and gets resumed
any number of times still produces one correct final report.

Progress logs to logs/universe_run.log (same file the earlier trailing-
stop 3-variant run used) per explicit request to keep both runs' progress
history in one place.

Run: python -m examples.run_smc_vwap_universe
    python -m examples.run_smc_vwap_universe --limit 400
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
from detectors.smc_strategy import SmartMoneyStrategy
from detectors.vwap_strategy import VWAPStrategy

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "smc_vwap_partial_results.jsonl"

# Same construction as run_universe_backtest.py's STRATEGY_BUILDERS for
# these two entries -- identical parameters, so this is a genuine "same
# strategy, bigger universe" test, not a different config being compared.
STRATEGY_BUILDERS = {
    "smart_money": lambda: SmartMoneyStrategy(order_block_lookback_bars=50, move_atr_mult=1.5),
    "vwap_reclaim": lambda: VWAPStrategy(volume_lookback_bars=20, volume_multiplier=1.3, stop_lookback_bars=15),
}
TIMEFRAMES = [1]  # both strategies only need 1-minute data


def _cached_symbols() -> list[str]:
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _run_one(task: tuple[str, date, date]) -> list[tuple[str, str, list[Trade]]]:
    symbol, start_date, end_date = task
    specs = {name: (build(), TIMEFRAMES) for name, build in STRATEGY_BUILDERS.items()}
    try:
        results_by_strategy = run_multi(symbol, specs, start_date, end_date)
    except Exception as e:
        print(f"  {symbol}: {e}")
        # Still returns both strategy names (with empty trade lists) so
        # this symbol gets marked COMPLETE on resume -- a permanently-
        # broken symbol (bad data, etc.) would otherwise be retried
        # forever on every resume, same reasoning as bulk_fetch.py.
        return [(symbol, name, []) for name in STRATEGY_BUILDERS]
    return [(symbol, name, result.trades) for name, result in results_by_strategy.items()]


def _trade_to_dict(trade: Trade) -> dict:
    return {
        "symbol": trade.symbol,
        "direction": trade.direction.value,
        "entry_time": trade.entry_time.isoformat(),
        "entry_price": trade.entry_price,
        "stop_loss": trade.stop_loss,
        "target": trade.target,
        "reasoning": trade.reasoning,
        "confidence": trade.confidence,
        "strategy_name": trade.strategy_name,
        "exit_time": trade.exit_time.isoformat() if trade.exit_time is not None else None,
        "exit_price": trade.exit_price,
        "exit_reason": trade.exit_reason,
        "initial_stop_loss": trade.initial_stop_loss,
        "peak_price": trade.peak_price,
        "breakeven_moved": trade.breakeven_moved,
        "trailing_active": trade.trailing_active,
    }


def _trade_from_dict(d: dict) -> Trade:
    return Trade(
        symbol=d["symbol"],
        direction=SignalDirection(d["direction"]),
        entry_time=pd.Timestamp(d["entry_time"]),
        entry_price=d["entry_price"],
        stop_loss=d["stop_loss"],
        target=d["target"],
        reasoning=d["reasoning"],
        confidence=d["confidence"],
        strategy_name=d["strategy_name"],
        exit_time=pd.Timestamp(d["exit_time"]) if d["exit_time"] is not None else None,
        exit_price=d["exit_price"],
        exit_reason=d["exit_reason"],
        initial_stop_loss=d.get("initial_stop_loss"),
        peak_price=d.get("peak_price"),
        breakeven_moved=d.get("breakeven_moved", False),
        trailing_active=d.get("trailing_active", False),
    )


def _append_partial_results(symbol: str, results_by_strategy: dict[str, list[Trade]]) -> None:
    """One write() call for both of this symbol's strategy lines -- keeps
    a crash from ever leaving a symbol half-written, same reasoning as
    run_validated_breakout_variants.py's identical helper."""
    lines = []
    for strategy_name, trades in results_by_strategy.items():
        record = {"symbol": symbol, "strategy": strategy_name, "trades": [_trade_to_dict(t) for t in trades]}
        lines.append(json.dumps(record))
    PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL_RESULTS_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")


def _load_completed_symbols() -> dict[str, dict[str, list[Trade]]]:
    """Returns {symbol: {strategy_name: trades}} for symbols that have
    BOTH strategies already recorded. A symbol with only one strategy
    present (process died mid-write, or a corrupted line) is treated as
    NOT done and recomputed from scratch -- run_multi() computes both in
    one call, so there's no way to redo just the missing one anyway."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N cached symbols (alphabetical). Omit for the full universe.")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("smc_vwap")
    log.info("=" * 70)
    log.info("SMC + VWAP RECLAIM FULL-UNIVERSE VALIDATION -- RUN START (checkpoint/resume enabled)")
    log.info("=" * 70)

    all_symbols = _cached_symbols()
    if not all_symbols:
        log.info("No cached data found -- run data/bulk_fetch.py first.")
        return
    if args.limit is not None:
        all_symbols = all_symbols[:args.limit]

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)

    cores = os.cpu_count() or 1
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
             f"Overall progress starts at {units_done}/{total_units} units.")

    if remaining:
        tasks = [(symbol, start_date, end_date) for symbol in remaining]
        progress_every = 100
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
                    log.info(f"{units_done}/{total_units} done, ETA {eta_min:.1f} min")

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
    log.info("SMC + VWAP RECLAIM: FULL-UNIVERSE RESULTS")
    log.info("=" * 70)
    reports = {}
    for name in STRATEGY_BUILDERS:
        report = compute(trades_by_strategy[name])
        reports[name] = report
        log.info(f"--- {name} ---")
        for line in str(report).split("\n"):
            log.info(line)

    log.info("=" * 70)
    log.info("SIDE BY SIDE")
    log.info("=" * 70)
    log.info(f"{'Metric':22s} {'smart_money':>14s} {'vwap_reclaim':>14s}")
    rows = [
        ("Trades", lambda r: f"{r.total_trades}"),
        ("Win rate", lambda r: f"{r.win_rate:.1%}"),
        ("Profit factor", lambda r: f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"),
        ("Avg R-multiple", lambda r: f"{r.avg_r_multiple:+.2f}"),
        ("Sharpe", lambda r: f"{r.sharpe_ratio:.2f}"),
        ("Max drawdown", lambda r: f"{r.max_drawdown_points:.1f}"),
        ("Expectancy (pts)", lambda r: f"{r.expectancy_points:+.3f}"),
    ]
    for label, fmt in rows:
        vals = [fmt(reports[name]) for name in STRATEGY_BUILDERS]
        log.info(f"{label:22s} {vals[0]:>14s} {vals[1]:>14s}")
    log.info("RUN COMPLETE")


if __name__ == "__main__":
    main()
