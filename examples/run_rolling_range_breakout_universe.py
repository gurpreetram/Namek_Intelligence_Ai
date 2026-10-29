"""
Full-universe RollingRangeBreakout trade generation -- first genuine
full-scale test of this strategy (added 2026-08-17 as one of the "three
fixes for big movers"; it had previously only run on small ad-hoc
subsets). Same checkpoint/resume architecture as examples/run_orb_
universe.py (built after a real ~5-hour run died at 88.5% complete with
nothing persisted) -- see that module's docstring for the full rationale.

TWO deliberate differences from run_orb_universe.py, not oversights:
  1. Uses data.storage.load_combined() (permanent archive + rolling
     cache merged) for MAXIMUM available history per symbol, not
     storage.load()'s ~30-day rolling window alone -- BacktestEngine.run()
     accepts a pre-loaded df_1min for exactly this reason (see its own
     docstring). start_date/end_date below span the full combined range
     this project has for ANY symbol, not a fixed 29-day lookback.
  2. Worker count is capped via MAX_WORKERS (default 4), not the ORB
     script's uncapped os.cpu_count() -- this project hit a real "paging
     file too small" crash on this exact 7.8GB-RAM machine running 8
     workers of a similar multiprocessing.Pool; 4 was the number
     established afterward as safe for a normal, otherwise-idle machine
     (examples/run_all_strategies_universe.py and the momentum-entry
     historical simulation both use the same env var). Set even lower
     via MAX_WORKERS=N if running alongside other memory-heavy processes
     (e.g. a live trading session) -- this run was deliberately launched
     with MAX_WORKERS=2 on 2026-08-20 for exactly that reason (a real
     live session with real open positions was active, plus real system
     RAM already at ~94% -- see this project's own notes from that day).

Run: python -m examples.run_rolling_range_breakout_universe
    python -m examples.run_rolling_range_breakout_universe --limit 400
    MAX_WORKERS=2 python -m examples.run_rolling_range_breakout_universe
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

from backtest.engine import BacktestEngine, Trade
from backtest.metrics import compute
from backtest.strategy_base import SignalDirection
from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage
from examples.rolling_range_breakout import RollingRangeBreakout

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "rolling_range_breakout_partial_results.jsonl"
TIMEFRAMES = [1, 5, 15]
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))


def _cached_symbols() -> list[str]:
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _run_one(task: tuple[str, date, date]) -> tuple[str, list[Trade]]:
    symbol, start_date, end_date = task
    engine = BacktestEngine(strategy=RollingRangeBreakout(), timeframes=TIMEFRAMES)
    try:
        df_1min = storage.load_combined(symbol)
        result = engine.run(symbol, start_date, end_date, df_1min=df_1min)
    except Exception as e:
        print(f"  {symbol}: {e}")
        return symbol, []  # empty list still marks this symbol COMPLETE on resume
    return symbol, result.trades


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


def _append_partial_result(symbol: str, trades: list[Trade]) -> None:
    record = {"symbol": symbol, "trades": [_trade_to_dict(t) for t in trades]}
    PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL_RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_completed_symbols() -> dict[str, list[Trade]]:
    if not PARTIAL_RESULTS_PATH.exists():
        return {}
    by_symbol: dict[str, list[Trade]] = {}
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
            by_symbol[record["symbol"]] = trades
    return by_symbol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()], force=True,
    )
    log = logging.getLogger("rolling_range_breakout_universe")
    log.info("=" * 70)
    log.info(f"ROLLING RANGE BREAKOUT FULL-UNIVERSE TRADE GENERATION -- RUN START "
             f"(checkpoint/resume enabled, MAX_WORKERS={MAX_WORKERS})")
    log.info("=" * 70)

    all_symbols = _cached_symbols()
    if not all_symbols:
        log.info("No cached data found.")
        return
    if args.limit is not None:
        all_symbols = all_symbols[:args.limit]

    # Maximum available history, not a fixed 29-day lookback -- see module
    # docstring. 2026-06-01 comfortably predates the earliest archived data
    # this project has (confirmed 2026-06-08 is the earliest real date
    # across the archive); BacktestEngine.run() clips to whatever's
    # actually present per-symbol regardless.
    start_date = date(2026, 6, 1)
    end_date = date.today() - timedelta(days=1)

    completed = _load_completed_symbols()
    already_done = [s for s in all_symbols if s in completed]
    remaining = [s for s in all_symbols if s not in completed]

    total_units = len(all_symbols)
    units_done = len(already_done)
    log.info(f"Resume check ({PARTIAL_RESULTS_PATH.name}): {len(already_done)}/{len(all_symbols)} "
             f"symbols already complete, skipping. {len(remaining)} remaining to process.")

    if remaining:
        tasks = [(symbol, start_date, end_date) for symbol in remaining]
        progress_every = 100
        last_bucket = units_done // progress_every
        start_time = time.monotonic()
        units_done_at_start = units_done

        with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
            for symbol, trades in pool.imap_unordered(_run_one, tasks):
                _append_partial_result(symbol, trades)
                units_done += 1
                bucket = units_done // progress_every
                if bucket != last_bucket or units_done >= total_units:
                    last_bucket = bucket
                    elapsed = time.monotonic() - start_time
                    processed = units_done - units_done_at_start
                    remaining_units = total_units - units_done
                    eta_min = (elapsed / processed) * remaining_units / 60 if processed > 0 else float("nan")
                    log.info(f"{units_done}/{total_units} symbols done, ETA {eta_min:.1f} min")

        elapsed = time.monotonic() - start_time
        log.info(f"This session's batch completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    else:
        log.info("All symbols already complete.")

    all_completed = _load_completed_symbols()
    all_trades: list[Trade] = []
    missing = []
    for symbol in all_symbols:
        trades = all_completed.get(symbol)
        if trades is None:
            missing.append(symbol)
            continue
        all_trades.extend(trades)
    if missing:
        log.warning(f"{len(missing)} symbols missing from partial results: {missing[:20]}")

    report = compute(all_trades)
    log.info("=" * 70)
    log.info("ROLLING RANGE BREAKOUT: FULL-UNIVERSE RESULT")
    log.info("=" * 70)
    for line in str(report).split("\n"):
        log.info(line)
    log.info("RUN COMPLETE")


if __name__ == "__main__":
    main()
