"""
Full-universe test of ONE variant: smart_money + CHANGE 2's pre-entry
verification layer (1.0% min-move + volume z-score >=1.0 + 3-bar
no-pullback, detectors/pick_verification.py) + the strategy's ORIGINAL
2x-risk-reward target (CHANGE 1's fixed 12-point target was tested and
reverted -- see detectors/scanner.py's FIXED_TARGET_POINTS docstring).
This is exactly the combination that scored 60 trades / 50.0% win rate /
PF 1.47 / avg R +0.01 / Sharpe 0.11 on a 400-symbol sample -- the question
here is whether that holds up at full scale, per this project's
established "400-symbol optimism doesn't always survive scaling up" rule.

Universe: every cached symbol whose LAST KNOWN cached close is >=
detectors.scanner.MIN_STOCK_PRICE (Rs.100) -- same price floor the live
scanner/Daily Picks apply, recomputed fresh from the current cache each
run (not a static list), consistent with examples/daily_universe_refresh.py's
_filter_min_price(). ~748 of 2,392 cached symbols excluded as of this
writing, ~1,644 remain.

CHECKPOINT/RESUME (same pattern as examples/run_validated_breakout_variants.py,
built after a real ~5-hour run died at 88.5% complete with nothing
persisted): every symbol's result (trades + rejected_signal_count) is
appended to logs/smart_money_verification_partial_results.jsonl
immediately after that symbol finishes. On startup, any symbol already
present is skipped. The final report is always built by reading the FULL
partial-results file, not this process's own in-memory results, so a run
that dies and gets resumed N times still produces one correct final
report -- laptop sleep has been a recurring interruption risk in this
project, so this matters here as much as it did for the trailing-stop test.

Progress logged to logs/universe_run.log (same shared log path
run_validated_breakout_variants.py uses -- both are "one-off full-universe
validation run" tools, sharing one log makes historical runs easy to find
in one place).

Run: python -m examples.run_smart_money_verification_universe
    python -m examples.run_smart_money_verification_universe --limit 400   # bounded test
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
from detectors.scanner import MIN_STOCK_PRICE
from detectors.smc_strategy import SmartMoneyStrategy

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "smart_money_verification_partial_results.jsonl"
VARIANT_NAME = "smart_money"
TIMEFRAMES = [1]


def _price_filtered_universe(log: logging.Logger) -> list[str]:
    """Every cached symbol whose last known close is >= MIN_STOCK_PRICE --
    reused from detectors.scanner, not a second threshold. Recomputed
    fresh from the current cache every run, same as
    examples/daily_universe_refresh.py's _filter_min_price()."""
    all_symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    kept = []
    excluded = 0
    for symbol in all_symbols:
        df = storage.load(symbol)
        if df.empty:
            kept.append(symbol)  # no price data to judge by -- don't exclude blind
            continue
        if float(df["close"].iloc[-1]) < MIN_STOCK_PRICE:
            excluded += 1
        else:
            kept.append(symbol)
    log.info(f"Price filter (MIN_STOCK_PRICE={MIN_STOCK_PRICE:.2f}): {excluded} of {len(all_symbols)} "
              f"cached symbols excluded, {len(kept)} remain in the test universe.")
    return kept


def _run_one(task: tuple[str, date, date]) -> dict:
    symbol, start_date, end_date = task
    specs = {VARIANT_NAME: (SmartMoneyStrategy(order_block_lookback_bars=50, move_atr_mult=1.5), TIMEFRAMES)}
    try:
        result = run_multi(symbol, specs, start_date, end_date,
                            verification_config={VARIANT_NAME: True})[VARIANT_NAME]
    except Exception as e:
        print(f"  {symbol}: {e}")
        return {"symbol": symbol, "trades": [], "rejected": 0, "errored": True}
    return {"symbol": symbol, "trades": result.trades, "rejected": result.rejected_signal_count, "errored": False}


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


def _append_partial_result(symbol: str, trades: list[Trade], rejected: int) -> None:
    record = {"symbol": symbol, "trades": [_trade_to_dict(t) for t in trades], "rejected": rejected}
    PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL_RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_completed_symbols() -> dict[str, dict]:
    """{symbol: {"trades": [...], "rejected": N}} for every symbol with a
    valid recorded line. A corrupted/partial line (process died mid-write)
    is simply ignored -- that symbol gets redone on the next run, same
    "treat incomplete as not-done" principle as run_validated_breakout_variants.py."""
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
                trades = [_trade_from_dict(t) for t in record["trades"]]
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            by_symbol[record["symbol"]] = {"trades": trades, "rejected": record.get("rejected", 0)}
    return by_symbol


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
    log = logging.getLogger("smart_money_verification_universe")
    log.info("=" * 70)
    log.info("SMART_MONEY + VERIFICATION + ORIGINAL 2x-RR TARGET -- FULL UNIVERSE RUN (checkpoint/resume enabled)")
    log.info("=" * 70)

    all_symbols = _price_filtered_universe(log)
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
                _append_partial_result(r["symbol"], r["trades"], r["rejected"])
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
    all_trades: list[Trade] = []
    total_rejected = 0
    missing = []
    for symbol in all_symbols:
        entry = all_completed.get(symbol)
        if entry is None:
            missing.append(symbol)
            continue
        all_trades.extend(entry["trades"])
        total_rejected += entry["rejected"]
    if missing:
        log.warning(f"{len(missing)} symbols missing from partial results after this run "
                    f"(unexpected -- investigate): {missing[:20]}{'...' if len(missing) > 20 else ''}")

    log.info("=" * 70)
    log.info("SMART_MONEY + VERIFICATION + ORIGINAL 2x-RR TARGET -- FULL UNIVERSE RESULT")
    log.info("=" * 70)
    report = compute(all_trades)
    for line in str(report).split("\n"):
        log.info(line)
    log.info(f"Rejected by verification: {total_rejected}")

    log.info("=" * 70)
    log.info("COMPARISON vs the 400-symbol sample (Trades 60, WR 50.0%, PF 1.47, Avg R +0.01, Sharpe 0.11)")
    log.info("=" * 70)
    pf_str = f"{report.profit_factor:.2f}" if report.profit_factor != float("inf") else "inf"
    log.info(f"{'Metric':22s} {'400-symbol':>12s} {'Full universe':>14s}")
    log.info(f"{'Trades':22s} {'60':>12s} {report.total_trades:>14d}")
    log.info(f"{'Win rate':22s} {'50.0%':>12s} {report.win_rate:>13.1%} ")
    log.info(f"{'Profit factor':22s} {'1.47':>12s} {pf_str:>14s}")
    log.info(f"{'Avg R-multiple':22s} {'+0.01':>12s} {report.avg_r_multiple:>+14.2f}")
    log.info(f"{'Sharpe':22s} {'0.11':>12s} {report.sharpe_ratio:>14.2f}")
    log.info(f"{'Max drawdown':22s} {'n/a':>12s} {report.max_drawdown_points:>14.1f}")
    log.info("RUN COMPLETE")


if __name__ == "__main__":
    main()
