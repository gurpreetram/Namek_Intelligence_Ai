"""
Compares three ValidatedBreakout variants across the cached universe:
  1. baseline_1_2  -- fixed 1:2 risk-reward target (the current default,
     already-backtested behavior).
  2. fixed_1_3     -- fixed 1:3 risk-reward target (wider target, same
     fixed-exit logic, via ValidatedBreakout's risk_reward_multiple param).
  3. trailing      -- no fixed target at all; stop moves to breakeven once
     profit reaches 1x initial risk, then trails at 0.5x initial risk
     behind the peak once profit reaches 1.5x initial risk (see
     backtest/engine.py's _check_exit_trailing).

All three share ONE load+resample pass per symbol via backtest.engine.run_multi()
(same reasoning as examples/run_universe_backtest.py: they all use the same
[1,5,15] timeframes, so loading/resampling 3x independently would be pure
waste). Progress uses imap_unordered + a running counter, not blocking
pool.map() -- same pattern as run_universe_backtest.py, for the same
reason: zero visibility into a run that can take many minutes otherwise.

CHECKPOINT/RESUME (added after this exact failure mode happened for real:
a laptop shutdown killed a ~5-hour run at 88.5% complete with NOTHING
persisted, since the only output was one aggregate report computed at the
very end). Every symbol's 3-variant result is appended to
logs/trailing_stop_partial_results.jsonl (one JSON line per symbol per
variant) IMMEDIATELY after that symbol finishes, not batched. On startup,
any symbol with all 3 variants already present in that file is skipped --
same "don't redo cached work" principle as data/bulk_fetch.py's
skip_if_cached. The final report is always computed by reading the FULL
partial-results file (every symbol ever completed, across however many
resumed runs it took), not from this process's own in-memory results --
so a run that dies and gets resumed 3 times still produces one correct
final report at the end, same as an uninterrupted run would have.

The question this answers: does letting winners run (wider fixed target,
or a trailing stop) improve avg R-multiple over the fixed 1:2 baseline,
or does it just increase variance without improving the average? Report
honestly either way -- this is a real test, not a foregone conclusion.

Run: python -m examples.run_validated_breakout_variants
    python -m examples.run_validated_breakout_variants --limit 400
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
from detectors.validated_breakout_strategy import ValidatedBreakout

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "trailing_stop_partial_results.jsonl"

VARIANT_BUILDERS = {
    "baseline_1_2": lambda: ValidatedBreakout(level_timeframe=15, min_touches=2),
    "fixed_1_3": lambda: ValidatedBreakout(level_timeframe=15, min_touches=2, risk_reward_multiple=3.0),
    "trailing": lambda: ValidatedBreakout(level_timeframe=15, min_touches=2),  # RR=2.0 default for the underlying
    # signal's stop/target convention (still needed by the gap-risk guard),
    # but the ENGINE ignores target entirely once use_trailing_stop=True.
}
TRAILING_CONFIG = {
    "trailing": {
        "use_trailing_stop": True,
        "breakeven_trigger_r": 1.0,
        "trail_activate_r": 1.5,
        "trail_distance_r": 0.5,
    },
}
TIMEFRAMES = [1, 5, 15]


def _cached_symbols() -> list[str]:
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _run_one(task: tuple[str, date, date]) -> list[tuple[str, str, list[Trade]]]:
    symbol, start_date, end_date = task
    specs = {name: (build(), TIMEFRAMES) for name, build in VARIANT_BUILDERS.items()}
    try:
        results_by_variant = run_multi(symbol, specs, start_date, end_date, trailing_config=TRAILING_CONFIG)
    except Exception as e:
        print(f"  {symbol}: {e}")
        # Still returns all 3 variant names (with empty trade lists) so this
        # symbol gets marked COMPLETE on resume -- a permanently-broken
        # symbol (bad data, etc.) would otherwise be retried forever on
        # every resume, same reasoning as bulk_fetch.py not endlessly
        # re-attempting a symbol that will never succeed.
        return [(symbol, name, []) for name in VARIANT_BUILDERS]
    return [(symbol, name, result.trades) for name, result in results_by_variant.items()]


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


def _append_partial_results(symbol: str, results_by_variant: dict[str, list[Trade]]) -> None:
    """One write() call for all 3 of this symbol's variant lines -- keeps
    a crash from ever leaving a symbol half-written (1 or 2 of 3 variant
    lines present), which would otherwise need extra logic to detect and
    redo. Not a formal fsync-level atomicity guarantee, but matches the
    same "atomic enough for a local single-writer file" reasoning already
    used elsewhere in this project (paper_trading/live_poller.py's
    latest_prices.json, via temp-file+rename -- that pattern doesn't
    translate directly to an APPEND-only file, so a single grouped write()
    is the equivalent here)."""
    lines = []
    for variant_name, trades in results_by_variant.items():
        record = {"symbol": symbol, "variant": variant_name, "trades": [_trade_to_dict(t) for t in trades]}
        lines.append(json.dumps(record))
    PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL_RESULTS_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")


def _load_completed_symbols() -> dict[str, dict[str, list[Trade]]]:
    """Returns {symbol: {variant_name: trades}} for symbols that have ALL
    3 variants already recorded in the partial-results file. A symbol
    with fewer than 3 variant lines (e.g. the process died mid-write, or
    a line got corrupted) is treated as NOT done and will be recomputed
    from scratch on the next run -- run_multi() computes all 3 variants
    in one call, so there's no way to redo just the missing one anyway."""
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
                continue  # corrupted/partial line -- ignore, symbol gets redone
            by_symbol.setdefault(record["symbol"], {})[record["variant"]] = trades
    return {
        symbol: variants for symbol, variants in by_symbol.items()
        if set(variants.keys()) == set(VARIANT_BUILDERS.keys())
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
    log = logging.getLogger("vb_variants")
    log.info("=" * 70)
    log.info("VALIDATED BREAKOUT 3-VARIANT COMPARISON -- RUN START (checkpoint/resume enabled)")
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
    n_variants = len(VARIANT_BUILDERS)

    completed = _load_completed_symbols()
    already_done = [s for s in all_symbols if s in completed]
    remaining = [s for s in all_symbols if s not in completed]

    total_units = len(all_symbols) * n_variants
    units_done = len(already_done) * n_variants
    log.info(f"Resume check ({PARTIAL_RESULTS_PATH.name}): {len(already_done)}/{len(all_symbols)} "
             f"symbols already complete, skipping. {len(remaining)} remaining to process.")
    log.info(f"Backtesting {len(remaining)} remaining symbols x {n_variants} ValidatedBreakout variants "
             f"= {len(remaining) * n_variants} runs this session, {cores} CPU cores. "
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
                results_by_variant = {name: trades for _, name, trades in per_symbol_results}
                _append_partial_results(symbol, results_by_variant)

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

    # Final report is ALWAYS built from the full partial-results file, not
    # from this process's own in-memory results -- correct whether this
    # run did all the work, none of it (fully resumed), or something in
    # between.
    log.info("Aggregating final results from partial-results file...")
    all_completed = _load_completed_symbols()
    trades_by_variant: dict[str, list[Trade]] = {name: [] for name in VARIANT_BUILDERS}
    missing = []
    for symbol in all_symbols:
        variants = all_completed.get(symbol)
        if variants is None:
            missing.append(symbol)
            continue
        for variant_name, trades in variants.items():
            trades_by_variant[variant_name].extend(trades)
    if missing:
        log.warning(f"{len(missing)} symbols missing from partial results after this run "
                    f"(unexpected -- investigate): {missing[:20]}{'...' if len(missing) > 20 else ''}")

    log.info("=" * 70)
    log.info("VALIDATED BREAKOUT: 3-VARIANT COMPARISON")
    log.info("=" * 70)
    reports = {}
    for name in VARIANT_BUILDERS:
        report = compute(trades_by_variant[name])
        reports[name] = report
        log.info(f"--- {name} ---")
        for line in str(report).split("\n"):
            log.info(line)

    log.info("=" * 70)
    log.info("SIDE BY SIDE")
    log.info("=" * 70)
    log.info(f"{'Metric':22s} {'baseline_1_2':>14s} {'fixed_1_3':>14s} {'trailing':>14s}")
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
        vals = [fmt(reports[name]) for name in VARIANT_BUILDERS]
        log.info(f"{label:22s} {vals[0]:>14s} {vals[1]:>14s} {vals[2]:>14s}")
    log.info("RUN COMPLETE")


if __name__ == "__main__":
    main()
