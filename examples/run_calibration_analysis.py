"""
Runs risk/probability_calibration.py's calibrate() against real
ValidatedBreakout backtest trades -- the actual check of whether this
strategy's confidence score means anything: do 0.75-confidence signals
really win around 75% of the time, or is confidence just an arbitrary
number from "3 of 4 factors agreed" with no real connection to outcomes?

Uses the SAME baseline configuration (risk_reward_multiple=2.0, no
trailing) as examples/run_validated_breakout_variants.py's "baseline_1_2"
variant and the original 4-strategy examples/run_universe_backtest.py --
not a new/different strategy setup, so this result is directly comparable
to those. Backtests 400 cached symbols (parallelized, imap_unordered
progress -- same pattern as run_universe_backtest.py), collects every
closed Trade (each already carries its originating Signal's confidence),
and calibrates against them.

Caches the result to reporting/calibration_report_validated_breakout.json
so api/main.py's GET /calibration-report can serve it without re-running
a 400-symbol backtest on every dashboard request -- this script is the
one-time (or periodically re-run) analysis step, not something the API
recomputes live.

Run: python -m examples.run_calibration_analysis
    python -m examples.run_calibration_analysis --limit 2373   # full universe instead
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from dataclasses import asdict
from datetime import date, timedelta

from config.settings import PROJECT_ROOT, STORAGE_ROOT
from detectors.validated_breakout_strategy import ValidatedBreakout
from risk.probability_calibration import calibrate


def _cached_symbols() -> list[str]:
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _run_one(task: tuple[str, date, date]) -> list:
    """Worker: baseline ValidatedBreakout (RR=2.0, no trailing) on ONE
    symbol. Plain BacktestEngine.run(), not run_multi() -- only one
    strategy variant needed here, no sharing to do."""
    from backtest.engine import BacktestEngine
    symbol, start_date, end_date = task
    engine = BacktestEngine(strategy=ValidatedBreakout(level_timeframe=15, min_touches=2), timeframes=[1, 5, 15])
    try:
        result = engine.run(symbol, start_date, end_date)
    except Exception as e:
        print(f"  {symbol}: {e}")
        return []
    return result.trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=400,
                         help="How many cached symbols (alphabetical) to backtest. Default 400, "
                              "matching the reference run this script's docstring compares against.")
    args = parser.parse_args()

    symbols = _cached_symbols()
    if not symbols:
        print("No cached data found -- run data/bulk_fetch.py first.")
        return
    symbols = symbols[:args.limit]

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)

    cores = os.cpu_count() or 1
    tasks = [(symbol, start_date, end_date) for symbol in symbols]
    print(f"Backtesting ValidatedBreakout (baseline, RR=2.0, no trailing) across "
          f"{len(symbols)} symbols, {cores} CPU cores...\n")

    all_trades = []
    units_done = 0
    progress_every = 50
    last_bucket = -1
    start_time = time.monotonic()

    with multiprocessing.Pool(processes=cores) as pool:
        for trades in pool.imap_unordered(_run_one, tasks):
            all_trades.extend(trades)
            units_done += 1
            bucket = units_done // progress_every
            if bucket != last_bucket or units_done >= len(tasks):
                last_bucket = bucket
                elapsed = time.monotonic() - start_time
                eta_min = (elapsed / units_done) * (len(tasks) - units_done) / 60
                print(f"{units_done}/{len(tasks)} symbols done, {len(all_trades)} trades so far, "
                      f"ETA {eta_min:.1f} min", flush=True)

    elapsed = time.monotonic() - start_time
    print(f"\nBacktest completed in {elapsed:.1f}s ({elapsed / 60:.1f} min) -- "
          f"{len(all_trades)} total trades.\n")

    report = calibrate(all_trades, n_bins=5, min_reliable_sample=30)

    print("=" * 70)
    print("CALIBRATION REPORT -- ValidatedBreakout (baseline)")
    print("=" * 70)
    print(report)
    print()
    for r in report.reasoning:
        print(f"- {r}")

    # Honest verdict, computed here (not asserted) -- correlation between
    # bin midpoint confidence and actual win rate across RELIABLE bins
    # only, since unreliable (low-sample) bins are noise by the module's
    # own definition.
    reliable_bins = [b for b in report.bins if b.reliable]
    print(f"\nReliable bins (>=30 trades): {len(reliable_bins)} of {len(report.bins)}")
    if len(reliable_bins) >= 2:
        import numpy as np
        avg_confs = np.array([b.avg_confidence for b in reliable_bins])
        win_rates = np.array([b.actual_win_rate for b in reliable_bins])
        correlation = float(np.corrcoef(avg_confs, win_rates)[0, 1]) if len(set(avg_confs)) > 1 else float("nan")
        print(f"Correlation between avg confidence and actual win rate across reliable bins: {correlation:+.3f}")
        print("(+1.0 = perfectly calibrated, 0 = confidence has no relationship to outcome, "
              "negative = confidence is backwards)")
    else:
        print("Fewer than 2 reliable bins -- cannot meaningfully assess a calibration trend, "
              "only report the raw numbers above.")

    output_path = PROJECT_ROOT / "reporting" / "calibration_report_validated_breakout.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "strategy": "validated_breakout",
        "symbols_tested": len(symbols),
        "brier_score": report.brier_score,
        "overall_trade_count": report.overall_trade_count,
        "reasoning": report.reasoning,
        "bins": [
            {
                "confidence_range_low": b.confidence_range[0],
                "confidence_range_high": b.confidence_range[1],
                "trade_count": b.trade_count,
                "actual_win_rate": b.actual_win_rate,
                "avg_confidence": b.avg_confidence,
                "reliable": b.reliable,
            }
            for b in report.bins
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
