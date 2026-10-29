"""
Validates the early-exit warning system (see paper_trading/engine.py's
_check_early_exit()/_check_momentum_reversal()) via HISTORICAL REPLAY
BEFORE trusting it in a real live session -- per explicit instruction,
this must happen first, with the false-positive rate reported honestly.

Methodology: runs a normal bar-by-bar historical replay -- one
PaperTradingEngine per (symbol, strategy) pair, fed via on_new_bar()
exactly as paper_trading/live_poller.py's LivePoller.feed_bar() does live
(same code path, not a parallel/simplified one) -- across all 7
registered strategies, and collects every trade that exited via one of
the two early-exit reasons.

For each early exit, this looks FORWARD in the ALREADY-KNOWN historical
data (this is backtesting, not live -- the "future" from the early-exit
point onward already happened and is sitting in the cache) from the
early-exit bar to EOD square-off (15:15), and classifies the outcome
using the SAME intrabar high/low touch logic PaperTradingEngine.
_check_exit() uses for the ORIGINAL stop_loss/target (the ones the
position would have kept if the early exit hadn't fired):

  CORRECT  -- price later touches the original stop_loss (at or before
              touching target) -- the early exit avoided a loss that
              would have happened anyway.
  WASTED   -- price later touches the original target WITHOUT ever
              touching the original stop_loss first -- the position
              would have gone on to hit its full profit target; the
              early exit gave that up. This is the "false positive" the
              task brief asked about.
  NEUTRAL  -- neither the original stop nor target is touched before EOD
              -- ambiguous, reported separately, not folded into either
              bucket (a position drifting sideways near where it already
              was isn't really a correct call OR a false positive).

False-positive rate = WASTED / (CORRECT + WASTED), among RESOLVED cases
only (NEUTRAL excluded from the ratio, reported alongside it).

Parallelized across symbols via multiprocessing.Pool -- same pattern as
examples/run_universe_backtest.py/run_morning_watchlist.py (each symbol's
replay is fully independent of every other symbol's, so this parallelizes
trivially). Needed in practice, not just as an optimization: the
15-symbol sequential run took 40 minutes (~23s/symbol/strategy); 150-200
symbols sequentially would be 7-9+ HOURS. Each worker loads its own
symbol's data (not passed through IPC -- pickling large DataFrames across
process boundaries is itself slow) and writes to its own throwaway,
per-symbol journal file (never the real paper_trading/journal.jsonl,
and never a SHARED file across workers, which concurrent processes
appending to at once could corrupt).

Run: python -m examples.validate_early_exit_replay
    python -m examples.validate_early_exit_replay --limit 200
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from collections import defaultdict

import pandas as pd

from backtest.strategy_base import SignalDirection
from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage
from detectors.fibonacci_strategy import FibonacciStrategy
from detectors.reversal_strategy import ReversalDetection
from detectors.smc_strategy import SmartMoneyStrategy
from detectors.trend_continuation_strategy import TrendContinuation
from detectors.validated_breakout_strategy import ValidatedBreakout
from detectors.vwap_strategy import VWAPStrategy
from examples.opening_range_breakout import OpeningRangeBreakout
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import TradeJournal

STRATEGY_BUILDERS = {
    "opening_range_breakout": (lambda: OpeningRangeBreakout(), [1, 5, 15]),
    "validated_breakout": (lambda: ValidatedBreakout(), [1, 5, 15]),
    "trend_continuation": (lambda: TrendContinuation(), [1, 15]),
    "reversal_detection": (lambda: ReversalDetection(), [1]),
    "vwap_reclaim": (lambda: VWAPStrategy(), [1]),
    "smart_money": (lambda: SmartMoneyStrategy(), [1]),
    "fibonacci_retracement": (lambda: FibonacciStrategy(), [1]),
}

EARLY_EXIT_REASONS = ("early_warning_momentum_reversal", "early_warning_setup_invalidated")
EXAMPLES_PER_SYMBOL_CAP = 2  # keep each worker's IPC payload small


def _classify_outcome(day_df: pd.DataFrame, direction: SignalDirection, stop_loss: float,
                       target: float, after_ts: pd.Timestamp) -> str:
    """Looks forward from after_ts (the early-exit bar) to EOD, bar by
    bar, using the same intrabar high/low touch rule _check_exit() uses,
    to see which of the ORIGINAL stop_loss/target would have been hit
    first had the early exit not fired."""
    future = day_df[day_df.index > after_ts]
    for _, bar in future.iterrows():
        high, low = float(bar["high"]), float(bar["low"])
        if direction == SignalDirection.LONG:
            stop_hit = low <= stop_loss
            target_hit = high >= target
        else:
            stop_hit = high >= stop_loss
            target_hit = low <= target
        if stop_hit:
            return "CORRECT"
        if target_hit:
            return "WASTED"
    return "NEUTRAL"


def _replay_one_symbol(symbol: str) -> dict:
    """Worker function -- one symbol, all 7 strategies, full cached
    history. Returns a compact, JSON-picklable summary rather than raw
    Trade objects (keeps IPC payload small across ~150-200 worker calls)."""
    df = storage.load(symbol)
    if df.empty:
        return {"symbol": symbol, "outcomes": {}, "examples": [], "reason_counts": {}}

    journal_path = PROJECT_ROOT / "logs" / f"_early_exit_validation_tmp_{symbol}_{os.getpid()}.jsonl"
    journal = TradeJournal(path=journal_path)

    engines = {
        name: PaperTradingEngine(build(), tfs, journal=journal, notifier=None)
        for name, (build, tfs) in STRATEGY_BUILDERS.items()
    }
    outcomes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    examples: list[dict] = []
    reason_counts: dict[str, int] = defaultdict(int)

    for current_index in df.index:
        # Pass the FULL (fixed) df directly, not a fresh per-bar slice --
        # on_new_bar() already internally truncates every read to
        # `:current_index` on its own (confirmed by inspection).
        for strategy_name, engine in engines.items():
            open_before = engine.open_positions().get(symbol)
            engine.on_new_bar(symbol, df, current_index)
            open_after = engine.open_positions().get(symbol)

            if open_before is not None and (open_after is None or open_after is not open_before):
                if open_before.exit_reason in EARLY_EXIT_REASONS:
                    reason_counts[open_before.exit_reason] = reason_counts.get(open_before.exit_reason, 0) + 1
                    day_df = df[df.index.date == open_before.entry_time.date()]
                    outcome = _classify_outcome(
                        day_df, open_before.direction, open_before.stop_loss,
                        open_before.target, open_before.exit_time,
                    )
                    outcomes[strategy_name][outcome] += 1
                    outcomes[strategy_name]["_reason_" + open_before.exit_reason] += 1
                    if len(examples) < EXAMPLES_PER_SYMBOL_CAP:
                        examples.append({
                            "symbol": symbol, "strategy": strategy_name,
                            "direction": open_before.direction.value,
                            "exit_reason": open_before.exit_reason,
                            "exit_reasoning": open_before.exit_reasoning,
                            "entry_price": open_before.entry_price,
                            "exit_price": open_before.exit_price,
                            "stop_loss": open_before.stop_loss, "target": open_before.target,
                            "outcome": outcome,
                        })

    try:
        journal_path.unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "symbol": symbol,
        "outcomes": {k: dict(v) for k, v in outcomes.items()},
        "examples": examples,
        "reason_counts": dict(reason_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="number of symbols to replay (default 200)")
    parser.add_argument("--workers", type=int, default=None, help="defaults to os.cpu_count()")
    args = parser.parse_args()

    symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))[:args.limit]
    cores = args.workers or os.cpu_count() or 1
    print(f"Replaying {len(symbols)} symbols x {len(STRATEGY_BUILDERS)} strategies "
          f"across {cores} CPU cores (each worker handles one symbol, all 7 strategies, "
          f"full cached history)...\n")

    outcomes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    early_exit_examples: list[dict] = []
    total_trades_by_reason: dict[str, int] = defaultdict(int)
    symbols_with_data = 0
    replay_start = time.monotonic()

    with multiprocessing.Pool(processes=cores) as pool:
        for done, result in enumerate(pool.imap_unordered(_replay_one_symbol, symbols), 1):
            if result["outcomes"] or result["reason_counts"]:
                symbols_with_data += 1
            for strategy_name, o in result["outcomes"].items():
                for k, v in o.items():
                    outcomes[strategy_name][k] += v
            for reason, count in result["reason_counts"].items():
                total_trades_by_reason[reason] += count
            early_exit_examples.extend(result["examples"])

            progress_every = max(1, min(10, len(symbols) // 10 or 1))
            if done % progress_every == 0 or done == len(symbols):
                elapsed = time.monotonic() - replay_start
                eta_min = (elapsed / done) * (len(symbols) - done) / 60
                print(f"  {done}/{len(symbols)} symbols replayed, ETA {eta_min:.1f} min", flush=True)

    elapsed = time.monotonic() - replay_start
    print(f"\nReplay completed in {elapsed:.1f}s ({elapsed / 60:.2f} min)\n")

    print("=" * 70)
    print("EARLY EXIT WARNING -- REPLAY VALIDATION RESULT")
    print("=" * 70)
    print(f"Symbols replayed: {len(symbols)}  ({symbols_with_data} had usable cached data)")
    total_early_exits = sum(total_trades_by_reason.values())
    print(f"Total early exits fired: {total_early_exits}")
    for reason, count in total_trades_by_reason.items():
        print(f"  {reason}: {count}")
    print()

    grand_correct = grand_wasted = grand_neutral = 0
    for strategy_name in STRATEGY_BUILDERS:
        o = outcomes.get(strategy_name, {})
        correct, wasted, neutral = o.get("CORRECT", 0), o.get("WASTED", 0), o.get("NEUTRAL", 0)
        total = correct + wasted + neutral
        if total == 0:
            print(f"{strategy_name}: no early exits fired")
            continue
        resolved = correct + wasted
        fp_rate = wasted / resolved if resolved > 0 else float("nan")
        print(f"{strategy_name}: {total} early exits "
              f"(momentum_reversal={o.get('_reason_early_warning_momentum_reversal', 0)}, "
              f"setup_invalidated={o.get('_reason_early_warning_setup_invalidated', 0)})")
        print(f"  CORRECT (avoided real stop-loss): {correct}")
        print(f"  WASTED  (would have hit target):  {wasted}")
        print(f"  NEUTRAL (neither hit by EOD):      {neutral}")
        print(f"  False-positive rate (WASTED / resolved): {fp_rate:.1%}" if resolved > 0 else "  False-positive rate: n/a (no resolved cases)")
        print()
        grand_correct += correct
        grand_wasted += wasted
        grand_neutral += neutral

    grand_resolved = grand_correct + grand_wasted
    print("=" * 70)
    print("OVERALL (all strategies combined)")
    print("=" * 70)
    print(f"CORRECT: {grand_correct}   WASTED: {grand_wasted}   NEUTRAL: {grand_neutral}")
    if grand_resolved > 0:
        print(f"Overall false-positive rate (WASTED / resolved): {grand_wasted / grand_resolved:.1%}")
        print(f"Overall correct rate (CORRECT / resolved): {grand_correct / grand_resolved:.1%}")
    else:
        print("No resolved early exits to compute a rate from.")
    print(f"({grand_neutral} additional NEUTRAL cases excluded from the rate above -- "
          f"neither the original stop nor target was touched before EOD)")

    print()
    print("=" * 70)
    print(f"EXAMPLE EARLY EXITS (up to {EXAMPLES_PER_SYMBOL_CAP}/symbol collected, showing first 20)")
    print("=" * 70)
    for ex in early_exit_examples[:20]:
        print(f"  {ex['symbol']:12s} {ex['strategy']:24s} {ex['direction']:5s} "
              f"[{ex['outcome']}]  entry={ex['entry_price']:.2f} exit={ex['exit_price']:.2f} "
              f"stop={ex['stop_loss']:.2f} target={ex['target']:.2f}")
        for r in ex["exit_reasoning"]:
            print(f"      - {r}")


if __name__ == "__main__":
    main()
