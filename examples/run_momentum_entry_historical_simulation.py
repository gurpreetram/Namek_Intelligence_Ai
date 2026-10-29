"""
Historical simulation of detectors/momentum_entry_strategy.py's
MomentumEntryStrategy across the full archived history -- built to answer
a specific question before deciding whether to relax or remove the
strategy's freshness gate: a live fine-grained scan (17 checkpoints across
2026-08-14) found ZERO real signals, and gate-by-gate tracing on real big
movers showed each individual gate computing sensibly -- but one real day
isn't enough evidence to judge whether the 5-gate combination is
calibrated correctly or simply too strict. This replays the SAME gate
logic across every archived trading day at 5 intraday checkpoints each.

Reuses the exact same gate functions MomentumEntryStrategy.evaluate()
calls (detectors/breakout_validator.py's volume_zscore(), detectors/
strong_trend.py's check_trend_consistency()/check_volume_consistency()/
check_opening_range_hold()/find_trend_start(), detectors/support_
resistance.py's average_true_range(), detectors/liquidity_analysis.py's
analyze_liquidity()) -- NOT a second, reimplemented copy of the strategy's
logic, so a "signal fired here" always means the real strategy would also
have fired it. Only the FREESHNESS gate's threshold is varied (Variant A:
<=30min, matching the live strategy's current default; B: <=90min; C: no
limit) -- computed by evaluating gates 1-3c ONCE per (symbol, day,
checkpoint) and classifying the same age_minutes value against 3
thresholds, not re-running the (expensive) persistence/volume/range gates
3x.

Per (symbol, day), only the FIRST chronologically-qualifying checkpoint
counts as a signal, matching the real strategy's _traded_today one-shot-
per-day state -- a symbol whose setup remains valid across multiple
checkpoints in the same day would only ever fire once live, so counting
every checkpoint it stays valid would overstate real trading opportunities.

BACKTEST SIMULATION for any signal found: entry at the NEXT 1-min bar's
open after the qualifying checkpoint (no lookahead, same convention every
other strategy/backtest in this project uses), exit at stop-loss/target
(checked bar-by-bar, stop checked before target, same order backtest/
engine.py's _check_exit() uses) or 15:15 EOD square-off close, whichever
comes first.

CHECKPOINT/RESUME: same architecture as examples/run_smc_vwap_universe.py
-- every symbol's full result (funnel records across all days/checkpoints,
backtest trades for all 3 variants) is appended to logs/momentum_entry_
historical_partial_results.jsonl immediately after that symbol finishes.
On startup, an already-complete symbol is skipped. The final report always
reads the FULL partial-results file, not this process's in-memory state.

Progress logs to logs/universe_run.log (same shared file every other
full-universe validation run in this project uses).

Run: python -m examples.run_momentum_entry_historical_simulation
    python -m examples.run_momentum_entry_historical_simulation --limit 200
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from datetime import date

import pandas as pd

from backtest.strategy_base import SignalDirection
from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage
from detectors.breakout_validator import volume_zscore
from detectors.liquidity_analysis import analyze_liquidity
from detectors.pick_verification import VOLUME_LOOKBACK_BARS
from detectors.scanner import MIN_STOCK_PRICE
from detectors.strong_trend import (
    STRONG_TREND_MIN_CONSECUTIVE,
    STRONG_TREND_OPENING_RANGE_MINUTES,
    STRONG_TREND_VOLUME_MIN_RATIO,
    STRONG_TREND_WINDOW_MINUTES,
    check_opening_range_hold,
    check_trend_consistency,
    check_volume_consistency,
    find_trend_start,
)
from detectors.support_resistance import average_true_range

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "momentum_entry_historical_partial_results.jsonl"

CHECKPOINTS = ["10:00", "11:00", "12:00", "13:00", "14:00"]
VARIANTS = {"A_30min": 30.0, "B_90min": 90.0, "C_no_limit": None}
MIN_MOVE_PCT = 3.0
MIN_VOLUME_ZSCORE = 1.5
SWING_LOOKBACK_BARS = 10
MIN_STOP_DISTANCE_ATR_MULT = 0.5
RISK_REWARD_MULTIPLE = 2.0
EOD_SQUARE_OFF_TIME = "15:15"


def _cached_symbols() -> list[str]:
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _simulate_trade_forward(day_df: pd.DataFrame, entry_pos: int, direction: SignalDirection,
                             stop: float, target: float) -> dict | None:
    """entry_pos: integer position in day_df of the bar to enter AT its open
    (the bar immediately after the qualifying checkpoint) -- None if that
    bar doesn't exist (signal too close to EOD to actually fill)."""
    if entry_pos >= len(day_df):
        return None
    entry_price = float(day_df["open"].iloc[entry_pos])
    risk = abs(entry_price - stop)
    if risk <= 0:
        return None

    exit_price = None
    exit_reason = None
    for i in range(entry_pos, len(day_df)):
        bar = day_df.iloc[i]
        ts = day_df.index[i]
        high, low = float(bar["high"]), float(bar["low"])
        if direction == SignalDirection.LONG:
            if low <= stop:
                exit_price, exit_reason = stop, "stop_loss"
                break
            if high >= target:
                exit_price, exit_reason = target, "target"
                break
        else:
            if high >= stop:
                exit_price, exit_reason = stop, "stop_loss"
                break
            if low <= target:
                exit_price, exit_reason = target, "target"
                break
        if ts.strftime("%H:%M") >= EOD_SQUARE_OFF_TIME:
            exit_price, exit_reason = float(bar["close"]), "eod_square_off"
            break
    if exit_price is None:
        exit_price, exit_reason = float(day_df["close"].iloc[-1]), "eod_square_off"

    sign = 1.0 if direction == SignalDirection.LONG else -1.0
    pnl_points = sign * (exit_price - entry_price)
    r_multiple = pnl_points / risk
    return {
        "entry_price": entry_price, "exit_price": exit_price, "exit_reason": exit_reason,
        "pnl_points": pnl_points, "r_multiple": r_multiple,
    }


def _process_symbol(symbol: str) -> dict:
    empty = {"symbol": symbol, "checkpoint_records": [], "trades_by_variant": {v: [] for v in VARIANTS}}
    try:
        df = storage.load_combined(symbol)
    except Exception as e:
        print(f"  {symbol}: load error -- {e}")
        return empty
    if df.empty:
        return empty
    # Same price-floor convention as every other full-universe script here
    # (examples/run_smart_money_verification_universe.py's _price_filtered_
    # universe()) -- latest cached close, not re-evaluated per historical day.
    if float(df["close"].iloc[-1]) < MIN_STOCK_PRICE:
        return empty

    trading_days = sorted(set(df.index.date))
    checkpoint_records: list[dict] = []
    trades_by_variant: dict[str, list[dict]] = {v: [] for v in VARIANTS}

    for day in trading_days:
        day_df_full = df[df.index.date == day]
        if day_df_full.empty:
            continue
        open_price = float(day_df_full["open"].iloc[0])
        if open_price <= 0:
            continue
        tz = day_df_full.index.tz

        day_gate_results = []  # chronological (checkpoint, as_of, all_gates_ok, age_minutes, direction, close, stop, target)
        for cp in CHECKPOINTS:
            as_of = pd.Timestamp(f"{day} {cp}:00").tz_localize(tz) if tz is not None else pd.Timestamp(f"{day} {cp}:00")
            today_df = day_df_full.loc[:as_of]
            if len(today_df) < VOLUME_LOOKBACK_BARS + 2:
                continue

            # Same centralized liquidity gate scan_symbol() applies BEFORE
            # any strategy runs -- checked here too so this simulation
            # doesn't overcount signals the real pipeline would never reach.
            liquidity = analyze_liquidity(today_df)
            if liquidity is not None and not liquidity.is_liquid:
                continue

            current_close = float(today_df["close"].iloc[-1])
            raw_move_pct = (current_close - open_price) / open_price * 100.0
            if raw_move_pct >= MIN_MOVE_PCT:
                direction = SignalDirection.LONG
            elif raw_move_pct <= -MIN_MOVE_PCT:
                direction = SignalDirection.SHORT
            else:
                direction = None

            gate1 = direction is not None
            gate2 = gate3a = gate3b = gate3c = False
            age_minutes = None
            stop = target = None

            if gate1:
                current_volume = float(today_df["volume"].iloc[-1])
                recent_volumes = today_df["volume"].iloc[-(VOLUME_LOOKBACK_BARS + 1):-1]
                vz = volume_zscore(current_volume, recent_volumes)
                gate2 = vz >= MIN_VOLUME_ZSCORE

            if gate2:
                consistent, _ = check_trend_consistency(
                    today_df, as_of, direction, STRONG_TREND_WINDOW_MINUTES, STRONG_TREND_MIN_CONSECUTIVE,
                )
                gate3a = consistent

            if gate3a:
                volsteady, _ = check_volume_consistency(today_df, STRONG_TREND_VOLUME_MIN_RATIO)
                gate3b = volsteady

            if gate3b:
                range_ok, _ = check_opening_range_hold(today_df, direction, STRONG_TREND_OPENING_RANGE_MINUTES)
                gate3c = range_ok

            if gate3c:
                trend_start = find_trend_start(today_df, direction, open_price, MIN_MOVE_PCT)
                age_minutes = (as_of - trend_start).total_seconds() / 60.0

                recent = today_df.iloc[-SWING_LOOKBACK_BARS:]
                atr = average_true_range(today_df.iloc[-30:])
                min_distance = atr * MIN_STOP_DISTANCE_ATR_MULT
                if direction == SignalDirection.LONG:
                    stop = float(recent["low"].min())
                    risk = current_close - stop
                    if risk < min_distance:
                        stop = current_close - min_distance
                        risk = min_distance
                    target = current_close + RISK_REWARD_MULTIPLE * risk
                else:
                    stop = float(recent["high"].max())
                    risk = stop - current_close
                    if risk < min_distance:
                        stop = current_close + min_distance
                        risk = min_distance
                    target = current_close - RISK_REWARD_MULTIPLE * risk

            checkpoint_records.append({
                "day": str(day), "checkpoint": cp, "gate1_move": gate1, "gate2_volume": gate2,
                "gate3a_persistence": gate3a, "gate3b_volsteady": gate3b, "gate3c_range": gate3c,
                "age_minutes": age_minutes,
            })
            day_gate_results.append((cp, as_of, gate3c, age_minutes, direction, len(today_df), stop, target))

        for variant_name, threshold in VARIANTS.items():
            for cp, as_of, all_gates_ok, age_minutes, direction, entry_pos, stop, target in day_gate_results:
                if not all_gates_ok:
                    continue
                if threshold is not None and (age_minutes is None or age_minutes > threshold):
                    continue
                outcome = _simulate_trade_forward(day_df_full, entry_pos, direction, stop, target)
                if outcome is not None:
                    trades_by_variant[variant_name].append({
                        "symbol": symbol, "day": str(day), "checkpoint": cp, "direction": direction.value,
                        **outcome,
                    })
                break  # first qualifying checkpoint only -- matches _traded_today's one-shot-per-day gate

    return {"symbol": symbol, "checkpoint_records": checkpoint_records, "trades_by_variant": trades_by_variant}


def _append_partial_result(result: dict) -> None:
    PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL_RESULTS_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")


def _load_completed_symbols() -> dict[str, dict]:
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
            except json.JSONDecodeError:
                continue
            by_symbol[record["symbol"]] = record
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
    log = logging.getLogger("momentum_entry_historical")
    log.info("=" * 70)
    log.info("MOMENTUM ENTRY HISTORICAL SIMULATION -- RUN START (checkpoint/resume enabled)")
    log.info("=" * 70)

    all_symbols = _cached_symbols()
    if not all_symbols:
        log.info("No cached data found -- run data/bulk_fetch.py first.")
        return
    if args.limit is not None:
        all_symbols = all_symbols[:args.limit]

    cores = min(os.cpu_count() or 1, int(os.environ.get("MAX_WORKERS", 4)))

    completed = _load_completed_symbols()
    already_done = [s for s in all_symbols if s in completed]
    remaining = [s for s in all_symbols if s not in completed]

    log.info(f"Resume check ({PARTIAL_RESULTS_PATH.name}): {len(already_done)}/{len(all_symbols)} "
             f"symbols already complete, skipping. {len(remaining)} remaining to process.")
    log.info(f"Simulating {len(remaining)} remaining symbols x {len(CHECKPOINTS)} checkpoints/day x "
             f"~45 archived days, {cores} CPU cores.")

    if remaining:
        progress_every = 100
        symbols_done = len(already_done)
        last_bucket = symbols_done // progress_every
        start_time = time.monotonic()
        symbols_done_at_start = symbols_done

        with multiprocessing.Pool(processes=cores) as pool:
            for result in pool.imap_unordered(_process_symbol, remaining):
                _append_partial_result(result)
                symbols_done += 1
                bucket = symbols_done // progress_every
                if bucket != last_bucket or symbols_done >= len(all_symbols):
                    last_bucket = bucket
                    elapsed = time.monotonic() - start_time
                    processed = symbols_done - symbols_done_at_start
                    remaining_n = len(all_symbols) - symbols_done
                    eta_min = (elapsed / processed) * remaining_n / 60 if processed > 0 else float("nan")
                    log.info(f"{symbols_done}/{len(all_symbols)} symbols done, ETA {eta_min:.1f} min")

        elapsed = time.monotonic() - start_time
        log.info(f"This session's batch completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    else:
        log.info("All symbols already complete -- skipping straight to aggregation.")

    log.info("Aggregating final results from partial-results file...")
    all_completed = _load_completed_symbols()

    # --- Funnel: which gate rejects the most, among candidates that at least moved 3% ---
    funnel = {"gate1_move": 0, "gate2_volume": 0, "gate3a_persistence": 0, "gate3b_volsteady": 0, "gate3c_range": 0}
    total_records = 0
    checkpoint_first_fire_counts: dict[str, int] = {cp: 0 for cp in CHECKPOINTS}
    day_signal_counts: dict[str, int] = {}

    for symbol in all_symbols:
        entry = all_completed.get(symbol)
        if entry is None:
            continue
        for rec in entry["checkpoint_records"]:
            total_records += 1
            if rec["gate1_move"]:
                funnel["gate1_move"] += 1
            if rec["gate2_volume"]:
                funnel["gate2_volume"] += 1
            if rec["gate3a_persistence"]:
                funnel["gate3a_persistence"] += 1
            if rec["gate3b_volsteady"]:
                funnel["gate3b_volsteady"] += 1
            if rec["gate3c_range"]:
                funnel["gate3c_range"] += 1

    log.info("=" * 70)
    log.info("STEP 1: FUNNEL -- which gate rejects the most (sequential AND, among the prior gate's passers)")
    log.info("=" * 70)
    log.info(f"Total (symbol, day, checkpoint) evaluations with enough bars + liquid: {total_records}")
    log.info(f"  Pass gate 1 (move >= {MIN_MOVE_PCT}% from open):min          {funnel['gate1_move']}")
    log.info(f"  Pass gate 2 (volume z-score >= {MIN_VOLUME_ZSCORE}):              {funnel['gate2_volume']}")
    log.info(f"  Pass gate 3a (trend persistence, >=3 consec 5m bars): {funnel['gate3a_persistence']}")
    log.info(f"  Pass gate 3b (volume not drying up):                  {funnel['gate3b_volsteady']}")
    log.info(f"  Pass gate 3c (no reversal through opening range):     {funnel['gate3c_range']}")

    # --- Which checkpoint produces the most FIRST-fires (variant A, the current default) ---
    for symbol in all_symbols:
        entry = all_completed.get(symbol)
        if entry is None:
            continue
        for t in entry["trades_by_variant"].get("A_30min", []):
            checkpoint_first_fire_counts[t["checkpoint"]] = checkpoint_first_fire_counts.get(t["checkpoint"], 0) + 1
            day_signal_counts[t["day"]] = day_signal_counts.get(t["day"], 0) + 1

    log.info("=" * 70)
    log.info("Which checkpoint produces the most first-fires (Variant A, <=30min freshness)")
    log.info("=" * 70)
    for cp in CHECKPOINTS:
        log.info(f"  {cp}: {checkpoint_first_fire_counts.get(cp, 0)}")

    total_days = len(day_signal_counts) if day_signal_counts else len({r["day"] for e in all_completed.values() for r in e["checkpoint_records"]})
    log.info(f"Distinct archived trading days seen: {total_days}")

    # --- STEP 2/3: backtest each variant ---
    log.info("=" * 70)
    log.info("STEP 2/3: BACKTEST RESULTS PER VARIANT")
    log.info("=" * 70)
    variant_stats = {}
    for variant_name in VARIANTS:
        trades = []
        for symbol in all_symbols:
            entry = all_completed.get(symbol)
            if entry is None:
                continue
            trades.extend(entry["trades_by_variant"].get(variant_name, []))

        n = len(trades)
        wins = [t for t in trades if t["pnl_points"] > 0]
        losses = [t for t in trades if t["pnl_points"] <= 0]
        win_rate = len(wins) / n if n else 0.0
        gross_profit = sum(t["pnl_points"] for t in wins)
        gross_loss = abs(sum(t["pnl_points"] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        avg_r = sum(t["r_multiple"] for t in trades) / n if n else 0.0
        avg_return_pct = sum(t["pnl_points"] / t["entry_price"] * 100 for t in trades) / n if n else 0.0
        signals_per_day = n / total_days if total_days else 0.0

        variant_stats[variant_name] = {
            "trades": n, "win_rate": win_rate, "profit_factor": profit_factor,
            "avg_r": avg_r, "avg_return_pct": avg_return_pct, "signals_per_day": signals_per_day,
        }
        pf_str = f"{profit_factor:.2f}" if profit_factor != float("inf") else "inf"
        log.info(f"--- Variant {variant_name} ---")
        log.info(f"  Trades: {n}  (signals/day: {signals_per_day:.2f})")
        log.info(f"  Win rate: {win_rate:.1%}")
        log.info(f"  Profit factor: {pf_str}")
        log.info(f"  Avg R-multiple: {avg_r:+.2f}")
        log.info(f"  Avg return/trade: {avg_return_pct:+.2f}%")

    log.info("=" * 70)
    log.info("STEP 4: SIDE-BY-SIDE COMPARISON")
    log.info("=" * 70)
    log.info(f"{'Variant':14s} {'Trades':>8s} {'Sig/day':>9s} {'WinRate':>9s} {'PF':>7s} {'AvgR':>7s}")
    for variant_name, s in variant_stats.items():
        pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float("inf") else "inf"
        log.info(f"{variant_name:14s} {s['trades']:8d} {s['signals_per_day']:9.2f} {s['win_rate']:8.1%}  {pf_str:>6s} {s['avg_r']:+7.2f}")
    log.info("RUN COMPLETE")


if __name__ == "__main__":
    main()
