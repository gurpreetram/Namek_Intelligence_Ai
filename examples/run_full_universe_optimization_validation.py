"""
TEST 2 follow-up to examples/run_strategy_optimization.py: full-universe
validation of the two best SINGLE-change variants found there --
  smc_quality_filter    -- SmartMoneyStrategy(move_atr_mult=2.0), alone
  vb_confirmation_bars  -- ValidatedBreakout(required_confirmation_bars=2), alone
against the full cached universe (unfiltered -- same raw symbol list
convention run_strategy_optimization.py's 400-symbol test used, so this
is a true scale-up of the exact same methodology, not a differently-
scoped comparison), same 30-day window.

CHECKPOINT/RESUME: same pattern as examples/run_smart_money_verification_universe.py
(built after a real ~5-hour run died mid-way with nothing persisted) --
every symbol's result (BOTH variants' trades) is appended to
logs/optimization_validation_partial_results.jsonl immediately after
that symbol finishes. On startup, any symbol already present is skipped.
The final report is always built by reading the FULL partial-results
file, not this process's own in-memory results, so a run that dies and
gets resumed N times still produces one correct final report.

Run: python -m examples.run_full_universe_optimization_validation
    python -m examples.run_full_universe_optimization_validation --limit 400   # bounded test
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
from detectors.validated_breakout_strategy import ValidatedBreakout

LOG_PATH = PROJECT_ROOT / "logs" / "optimization_validation_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "optimization_validation_partial_results.jsonl"

SMC_KWARGS = dict(order_block_lookback_bars=50, move_atr_mult=2.0)
VB_KWARGS = dict(required_confirmation_bars=2)


def _specs() -> dict:
    return {
        "smc_quality_filter": (SmartMoneyStrategy(**SMC_KWARGS), [1]),
        "vb_confirmation_bars": (ValidatedBreakout(**VB_KWARGS), [1, 5, 15]),
    }


def _cached_symbols() -> list[str]:
    return sorted(p.name.removesuffix("_1min.parquet") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _trade_to_dict(trade: Trade) -> dict:
    return {
        "symbol": trade.symbol, "direction": trade.direction.value,
        "entry_time": trade.entry_time.isoformat(), "entry_price": trade.entry_price,
        "stop_loss": trade.stop_loss, "target": trade.target, "reasoning": trade.reasoning,
        "confidence": trade.confidence, "strategy_name": trade.strategy_name,
        "exit_time": trade.exit_time.isoformat() if trade.exit_time is not None else None,
        "exit_price": trade.exit_price, "exit_reason": trade.exit_reason,
    }


def _trade_from_dict(d: dict) -> Trade:
    return Trade(
        symbol=d["symbol"], direction=SignalDirection(d["direction"]),
        entry_time=pd.Timestamp(d["entry_time"]), entry_price=d["entry_price"],
        stop_loss=d["stop_loss"], target=d["target"], reasoning=d["reasoning"],
        confidence=d["confidence"], strategy_name=d["strategy_name"],
        exit_time=pd.Timestamp(d["exit_time"]) if d["exit_time"] is not None else None,
        exit_price=d["exit_price"], exit_reason=d["exit_reason"],
    )


def _run_one(task: tuple[str, date, date]) -> dict:
    symbol, start_date, end_date = task
    try:
        result = run_multi(symbol, _specs(), start_date, end_date)
        return {
            "symbol": symbol,
            "smc_trades": result["smc_quality_filter"].trades,
            "vb_trades": result["vb_confirmation_bars"].trades,
            "errored": False,
        }
    except Exception as e:
        print(f"  {symbol}: {e}")
        return {"symbol": symbol, "smc_trades": [], "vb_trades": [], "errored": True}


def _append_partial_result(r: dict) -> None:
    record = {
        "symbol": r["symbol"],
        "smc_trades": [_trade_to_dict(t) for t in r["smc_trades"]],
        "vb_trades": [_trade_to_dict(t) for t in r["vb_trades"]],
    }
    PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL_RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_completed_symbols() -> dict[str, dict]:
    """{symbol: {"smc_trades": [...], "vb_trades": [...]}} for every symbol
    with a valid recorded line. A corrupted/partial line (process died
    mid-write) is ignored -- that symbol gets redone on the next run."""
    if not PARTIAL_RESULTS_PATH.exists():
        return {}
    by_symbol: dict[str, dict] = {}
    with open(PARTIAL_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                smc_trades = [_trade_from_dict(t) for t in record["smc_trades"]]
                vb_trades = [_trade_from_dict(t) for t in record["vb_trades"]]
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            by_symbol[record["symbol"]] = {"smc_trades": smc_trades, "vb_trades": vb_trades}
    return by_symbol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only test the first N cached symbols (alphabetical). Omit for the full universe.")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()], force=True,
    )
    log = logging.getLogger("optimization_validation")
    log.info("=" * 70)
    log.info("FULL-UNIVERSE VALIDATION: smc_quality_filter + vb_confirmation_bars (checkpoint/resume enabled)")
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
    completed = _load_completed_symbols()
    already_done = [s for s in all_symbols if s in completed]
    remaining = [s for s in all_symbols if s not in completed]

    log.info(f"Resume check ({PARTIAL_RESULTS_PATH.name}): {len(already_done)}/{len(all_symbols)} "
             f"symbols already complete, skipping. {len(remaining)} remaining to process.")
    log.info(f"Backtesting {len(remaining)} remaining symbols, {cores} CPU cores. "
             f"Overall progress starts at {len(already_done)}/{len(all_symbols)} symbols.")

    if remaining:
        tasks = [(symbol, start_date, end_date) for symbol in remaining]
        progress_every = 50
        symbols_done = len(already_done)
        last_bucket = symbols_done // progress_every
        start_time = time.monotonic()
        symbols_done_at_start = symbols_done

        with multiprocessing.Pool(processes=cores) as pool:
            for r in pool.imap_unordered(_run_one, tasks):
                _append_partial_result(r)
                symbols_done += 1
                bucket = symbols_done // progress_every
                if bucket != last_bucket or symbols_done >= len(all_symbols):
                    last_bucket = bucket
                    elapsed = time.monotonic() - start_time
                    processed_this_run = symbols_done - symbols_done_at_start
                    remaining_symbols = len(all_symbols) - symbols_done
                    eta_min = (elapsed / processed_this_run) * remaining_symbols / 60 if processed_this_run > 0 else float("nan")
                    log.info(f"{symbols_done}/{len(all_symbols)} symbols done, ETA {eta_min:.1f} min")

        elapsed = time.monotonic() - start_time
        log.info(f"This session's batch completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    else:
        log.info("All symbols already complete -- skipping straight to aggregation.")

    log.info("Aggregating final results from partial-results file...")
    all_completed = _load_completed_symbols()
    all_smc_trades: list[Trade] = []
    all_vb_trades: list[Trade] = []
    for data in all_completed.values():
        all_smc_trades.extend(data["smc_trades"])
        all_vb_trades.extend(data["vb_trades"])

    smc_report = compute(all_smc_trades)
    vb_report = compute(all_vb_trades)

    def _pf(r) -> str:
        return f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"

    log.info("=" * 100)
    log.info(f"FULL-UNIVERSE RESULTS ({len(all_completed)}/{len(all_symbols)} symbols)")
    log.info("=" * 100)
    log.info(f"{'Variant':26s} {'Trades':>7s} {'WinRate':>8s} {'PF':>7s} {'AvgR':>7s} {'Sharpe':>7s} {'MaxDD':>9s}")
    log.info("-" * 100)
    log.info(f"{'smc_quality_filter':26s} {smc_report.total_trades:7d} {smc_report.win_rate:7.1%}  "
              f"{_pf(smc_report):>6s} {smc_report.avg_r_multiple:7.2f} {smc_report.sharpe_ratio:7.2f} "
              f"{smc_report.max_drawdown_points:9.1f}")
    log.info(f"{'vb_confirmation_bars':26s} {vb_report.total_trades:7d} {vb_report.win_rate:7.1%}  "
              f"{_pf(vb_report):>6s} {vb_report.avg_r_multiple:7.2f} {vb_report.sharpe_ratio:7.2f} "
              f"{vb_report.max_drawdown_points:9.1f}")
    log.info("=" * 100)

    log.info("\nCOMPARISON AGAINST KNOWN FULL-UNIVERSE BASELINES (as given):")
    log.info(f"  SMC baseline (full universe): PF 1.07, avg R -0.03  -->  quality_filter: "
              f"PF {_pf(smc_report)}, avg R {smc_report.avg_r_multiple:.2f}")
    log.info(f"  VB baseline (full universe):  PF 1.03, avg R -0.04  -->  confirmation_bars: "
              f"PF {_pf(vb_report)}, avg R {vb_report.avg_r_multiple:.2f}")


if __name__ == "__main__":
    main()
