"""
Full-universe Opening Range Breakout trade generation -- produces the same
full-trade-detail dataset examples/run_smc_vwap_universe.py already built
for smart_money/vwap_reclaim, but for opening_range_breakout, needed as
input for the ML signal-combination experiment (examples/train_signal_model.py):
feature extraction needs REAL trades with REAL entry_time/symbol/outcome,
and this project doesn't otherwise persist full per-trade records anywhere
(run_universe_backtest.py only ever printed aggregate summaries).

Same checkpoint/resume architecture as run_smc_vwap_universe.py (built
after a real ~5-hour run died at 88.5% complete with nothing persisted) --
see that module's docstring for the full rationale. Single strategy here,
so the "all N strategies present" resume check simplifies to "this symbol
has a line at all".

Run: python -m examples.run_orb_universe
    python -m examples.run_orb_universe --limit 400
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
from examples.opening_range_breakout import OpeningRangeBreakout

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "orb_partial_results.jsonl"
TIMEFRAMES = [1, 5, 15]


def _cached_symbols() -> list[str]:
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _run_one(task: tuple[str, date, date]) -> tuple[str, list[Trade]]:
    symbol, start_date, end_date = task
    engine = BacktestEngine(strategy=OpeningRangeBreakout(), timeframes=TIMEFRAMES)
    try:
        result = engine.run(symbol, start_date, end_date)
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
            by_symbol[record["symbol"]] = trades  # last line for a symbol wins if duplicated
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
    log = logging.getLogger("orb_universe")
    log.info("=" * 70)
    log.info("OPENING RANGE BREAKOUT FULL-UNIVERSE TRADE GENERATION -- RUN START (checkpoint/resume enabled)")
    log.info("=" * 70)

    all_symbols = _cached_symbols()
    if not all_symbols:
        log.info("No cached data found.")
        return
    if args.limit is not None:
        all_symbols = all_symbols[:args.limit]

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)
    cores = os.cpu_count() or 1

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

        with multiprocessing.Pool(processes=cores) as pool:
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
    log.info("OPENING RANGE BREAKOUT: FULL-UNIVERSE RESULT")
    log.info("=" * 70)
    for line in str(report).split("\n"):
        log.info(line)
    log.info("RUN COMPLETE")


if __name__ == "__main__":
    main()
