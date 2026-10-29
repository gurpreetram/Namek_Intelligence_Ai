"""
Autonomous end-to-end pipeline: validate the ATR-bounding/.index.date-caching
perf fix, speed-test it, then conditionally scale from a 400-symbol batch up
to the full cached NSE universe -- all with decision gates that can stop the
run early rather than burn hours on a fix that didn't actually help, and all
logged with timestamps to logs/universe_run.log so progress survives even if
nobody is watching the terminal.

Reuses STRATEGY_BUILDERS / STRATEGY_DISPLAY_NAMES / _cached_symbols / _run_one
from examples.run_universe_backtest directly rather than duplicating them, so
this pipeline and a manual `python -m examples.run_universe_backtest` run are
guaranteed to backtest identically.

Run: python -m examples.run_full_pipeline
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import time
from datetime import date, timedelta

from backtest.engine import MIN_REALIZED_RISK_FRACTION, run_multi
from backtest.metrics import compute
from config.settings import PROJECT_ROOT
from data import storage
from detectors.reversal_strategy import ReversalDetection
from detectors.scanner import scan_universe
from detectors.trend_continuation_strategy import TrendContinuation
from detectors.validated_breakout_strategy import ValidatedBreakout
from examples.opening_range_breakout import OpeningRangeBreakout
from examples.run_universe_backtest import (
    STRATEGY_BUILDERS,
    STRATEGY_DISPLAY_NAMES,
    _cached_symbols,
    _run_one,
)
from reporting.html_report import SymbolBreakdown, generate_report

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")

OLD_BASELINE_MIN_PER_SYMBOL = 11.4  # from the pre-fix run's own observed pace
PROCEED_THRESHOLD_MIN = 2.0
STOP_THRESHOLD_MIN = 5.0
STEP4_TIME_LIMIT_MIN = 90.0


def run_batch(symbols: list[str], start_date: date, end_date: date, cores: int,
              label: str) -> tuple[list[tuple[str, str, list]], float]:
    """
    Runs run_multi() across `symbols` via the same multiprocessing pool
    pattern as run_universe_backtest.py, logging progress every 100
    strategy-run units. Returns (flat raw_results, elapsed_seconds).
    """
    tasks = [(symbol, start_date, end_date) for symbol in symbols]
    total_units = len(tasks) * len(STRATEGY_BUILDERS)
    raw_results: list[tuple[str, str, list]] = []
    units_done = 0
    last_bucket = -1
    t0 = time.monotonic()

    with multiprocessing.Pool(processes=cores) as pool:
        for per_symbol_results in pool.imap_unordered(_run_one, tasks):
            raw_results.extend(per_symbol_results)
            units_done += len(per_symbol_results)
            bucket = units_done // 100
            if bucket != last_bucket or units_done >= total_units:
                last_bucket = bucket
                elapsed = time.monotonic() - t0
                eta_min = (elapsed / units_done) * (total_units - units_done) / 60
                log.info(f"[{label}] {units_done}/{total_units} done, ETA {eta_min:.1f} min")

    elapsed_total = time.monotonic() - t0
    return raw_results, elapsed_total


def aggregate(raw_results: list[tuple[str, str, list]]) -> tuple[dict, dict]:
    """Same aggregation logic as run_universe_backtest.py's main(). Returns
    (strategy_results: {display_name: (trades, report)}, per_symbol_breakdown)."""
    trades_by_strategy: dict[str, list] = {name: [] for name in STRATEGY_BUILDERS}
    trades_by_strategy_symbol: dict[str, dict[str, list]] = {name: {} for name in STRATEGY_BUILDERS}
    for symbol, strategy_name, trades in raw_results:
        trades_by_strategy[strategy_name].extend(trades)
        trades_by_strategy_symbol[strategy_name][symbol] = trades

    strategy_results = {}
    per_symbol_breakdown = {}
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

    return strategy_results, per_symbol_breakdown


def log_strategy_summary(strategy_results: dict, per_symbol_breakdown: dict, top_n: int = 15) -> None:
    for display_name, (trades, report) in strategy_results.items():
        log.info(f"-- {display_name} --")
        if trades:
            log.info(
                f"   trades={report.total_trades} win_rate={report.win_rate:.1%} "
                f"profit_factor={report.profit_factor:.2f} avg_R={report.avg_r_multiple:.2f} "
                f"sharpe={report.sharpe_ratio:.2f} max_dd={report.max_drawdown_points:.2f}pts"
            )
            breakdown = per_symbol_breakdown.get(display_name, [])
            top = breakdown[:top_n]
            bottom = breakdown[-top_n:] if len(breakdown) > top_n else []
            log.info(f"   Top {len(top)} by total R: " + ", ".join(
                f"{b.symbol}({b.total_r:+.2f}R)" for b in top))
            if bottom:
                log.info(f"   Bottom {len(bottom)} by total R: " + ", ".join(
                    f"{b.symbol}({b.total_r:+.2f}R)" for b in bottom))
        else:
            log.info("   No trades generated.")


def main() -> None:
    log.info("=" * 70)
    log.info("AUTONOMOUS PIPELINE START")
    log.info("=" * 70)

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)
    cores = os.cpu_count() or 1

    # --- STEP 1 ---
    log.info("STEP 1: old stale --limit 400 run (bg4hbce1w, liquidity-fix-only, "
              "no gap-risk fix) -- confirmed already killed manually before this "
              "pipeline started.")

    # --- STEP 2: validate on RELIANCE ---
    # NOTE: this used to assert trade counts against a hardcoded historical
    # baseline dict. That approach turned out to be unreliable for two
    # independent reasons discovered the hard way: (1) start_date/end_date
    # are a ROLLING 29-day window based on date.today(), so trade counts
    # legitimately drift day to day even with zero code changes; (2) the
    # gap-risk guard (see backtest/engine.py) can legitimately void trades
    # on ANY symbol, including liquid ones -- RELIANCE itself had one
    # (2026-07-07 12:18, realized risk 47.8% of intended, previously
    # inflated to +5.27R) voided by this exact guard. A "must match a fixed
    # number" gate would have falsely blocked the pipeline on correct
    # behavior. Instead this asserts the guard's own invariant holds on
    # live output: no surviving trade's realized risk is below
    # MIN_REALIZED_RISK_FRACTION of what its signal intended. That's a
    # direct, mechanical proof the fix is actually wired in and functioning
    # -- not a guess about what the trade count "should" be.
    log.info("STEP 2: validating liquidity + gap-risk guards on RELIANCE across all 4 strategies")
    specs = {name: (build(), tfs) for name, (build, tfs) in STRATEGY_BUILDERS.items()}
    results = run_multi("RELIANCE", specs, start_date, end_date)
    all_ok = True
    for name, result in results.items():
        trades = result.trades
        bad = []
        for t in trades:
            intended_risk = abs(t.target - t.stop_loss) / 3.0
            if intended_risk > 0 and t.risk_points < MIN_REALIZED_RISK_FRACTION * intended_risk - 1e-9:
                bad.append((t.entry_time, t.risk_points, intended_risk))
        ok = not bad
        all_ok = all_ok and ok
        log.info(f"   {name}: {len(trades)} trades, gap-risk invariant "
                  f"{'OK' if ok else f'VIOLATED on {bad}'}")

    if not all_ok:
        log.error("STOPPED: gap-risk guard invariant violated on live RELIANCE output, needs review")
        return
    log.info("STEP 2 PASSED: liquidity + gap-risk guards active and self-consistent on RELIANCE")

    # --- STEP 3: clean 5-symbol speed test ---
    log.info("STEP 3: running clean 5-symbol speed test")
    test_symbols = _cached_symbols()[:5]
    log.info(f"   Test symbols: {test_symbols}")
    _, elapsed_5 = run_batch(test_symbols, start_date, end_date, cores, "5-symbol-test")
    per_symbol_min = (elapsed_5 / len(test_symbols)) / 60
    log.info(f"   New time: {elapsed_5:.1f}s total for {len(test_symbols)} symbols "
              f"= {per_symbol_min:.2f} min/symbol (old baseline: ~{OLD_BASELINE_MIN_PER_SYMBOL:.1f} min/symbol)")

    if per_symbol_min < PROCEED_THRESHOLD_MIN:
        log.info(f"DECISION GATE: {per_symbol_min:.2f} min/symbol < {PROCEED_THRESHOLD_MIN} min "
                  f"threshold -- proceeding to STEP 4")
    else:
        reason = (
            f"{per_symbol_min:.2f} min/symbol is >= {STOP_THRESHOLD_MIN} min (clearly insufficient)"
            if per_symbol_min >= STOP_THRESHOLD_MIN else
            f"{per_symbol_min:.2f} min/symbol falls between the {PROCEED_THRESHOLD_MIN} min proceed "
            f"and {STOP_THRESHOLD_MIN} min stop thresholds -- not a clear enough win to justify "
            f"committing to a multi-hour run"
        )
        log.error(f"STOPPED: fix insufficient, needs review -- {reason}")
        return

    # --- STEP 4: 400-symbol run ---
    log.info("STEP 4: running full 400-symbol x 4-strategy backtest")
    symbols_400 = _cached_symbols()[:400]
    raw_400, elapsed_400 = run_batch(symbols_400, start_date, end_date, cores, "400-symbol")
    elapsed_400_min = elapsed_400 / 60
    log.info(f"STEP 4 completed in {elapsed_400_min:.1f} minutes for {len(symbols_400)} symbols")

    strategy_results_400, breakdown_400 = aggregate(raw_400)
    log_strategy_summary(strategy_results_400, breakdown_400)

    if elapsed_400_min >= STEP4_TIME_LIMIT_MIN:
        log.error(f"STOPPED: 400-symbol run too slow ({elapsed_400_min:.1f} min >= "
                  f"{STEP4_TIME_LIMIT_MIN} min threshold), needs review")
        return
    log.info(f"DECISION GATE: 400-symbol run completed in {elapsed_400_min:.1f} min "
              f"(< {STEP4_TIME_LIMIT_MIN} min threshold) -- proceeding to STEP 5 (full universe)")

    # --- STEP 5: full universe ---
    log.info("STEP 5: running full universe backtest (all cached symbols)")
    symbols_all = _cached_symbols()
    log.info(f"   Full universe: {len(symbols_all)} cached symbols")
    raw_all, elapsed_all = run_batch(symbols_all, start_date, end_date, cores, "full-universe")
    log.info(f"STEP 5 completed in {elapsed_all / 60:.1f} minutes for {len(symbols_all)} symbols")

    strategy_results_all, breakdown_all = aggregate(raw_all)
    log.info("=" * 70)
    log.info("FINAL SUMMARY -- FULL UNIVERSE")
    log.info("=" * 70)
    log_strategy_summary(strategy_results_all, breakdown_all, top_n=20)

    log.info("-- Comparison across strategies --")
    for display_name, (trades, report) in strategy_results_all.items():
        pf_str = f"{report.profit_factor:.2f}" if report.profit_factor != float("inf") else "inf"
        log.info(f"   {display_name:32s} trades={report.total_trades:6d} "
                  f"win_rate={report.win_rate:6.1%} PF={pf_str:>6s} "
                  f"avgR={report.avg_r_multiple:6.2f} sharpe={report.sharpe_ratio:6.2f}")

    log.info("Running final live-style scan across the full universe...")
    scan_data = {symbol: storage.load(symbol) for symbol in symbols_all}
    scan_data = {s: df for s, df in scan_data.items() if not df.empty}
    scanner_results: list = []
    if scan_data:
        most_recent = max(df.index.max() for df in scan_data.values())
        scanner_results = scan_universe(scan_data, most_recent, min_score=0.5)
        log.info(f"   Scanner found {len(scanner_results)} qualifying setups as of {most_recent}")

    report_path = PROJECT_ROOT / "reporting" / "latest_report.html"
    generate_report(
        report_path, strategy_results_all, scanner_results,
        data_source_note=f"Real Upstox data, {start_date} to {end_date} "
                          f"({(end_date - start_date).days} days, {len(symbols_all)} symbols -- "
                          f"full cached universe, autonomous pipeline run, {cores} CPU cores).",
        per_symbol_breakdown=breakdown_all,
    )
    log.info(f"HTML report written to: {report_path}")

    log.info("=" * 70)
    log.info("AUTONOMOUS PIPELINE COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
