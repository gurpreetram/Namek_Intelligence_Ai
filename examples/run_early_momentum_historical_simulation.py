"""
Historical simulation of detectors/early_momentum.py's find_early_momentum_
picks() across the full archived history -- same purpose and methodology
as examples/run_best_picks_historical_simulation.py (get honest performance
numbers before treating the F8 "Early Momentum" section as reliable), just
against the opposite-end-of-the-move signal. See that script's own
docstring for the full architecture rationale (single-threaded, not
parallelized -- find_early_momentum_picks() ranks across the WHOLE liquid
universe at each checkpoint, same non-independence-per-symbol reasoning
applies here unchanged).

METHODOLOGY (identical to the Best Picks script, only the picking function
differs): for each archived trading day, at 5 checkpoints (10:00, 11:00,
12:00, 13:00, 14:00 IST), call find_early_momentum_picks() against the
full liquid universe truncated to that moment -- exactly the real
function, not a reimplementation. Per (symbol, day), only the FIRST
checkpoint where that symbol appears in the top 3 counts as an entry.
Two-stage partial-exit simulation, checkpoint/resume, and the report
sections are all byte-for-byte the same simulation code as the Best Picks
script -- see that module for the full explanation of the partial-exit
rules and the honest "win = target_1 hit" definition (not blended P&L
positive).

Run: python -m examples.run_early_momentum_historical_simulation
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from backtest.engine import Trade
from backtest.metrics import compute
from backtest.strategy_base import SignalDirection
from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage
from detectors.early_momentum import find_early_momentum_picks
from detectors.scanner import MIN_STOCK_PRICE

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "early_momentum_historical_partial_results.jsonl"
SECTOR_MAPPING_PATH = PROJECT_ROOT / "config" / "sector_mapping.json"

CHECKPOINTS = ["10:00", "11:00", "12:00", "13:00", "14:00"]
EOD_SQUARE_OFF_TIME = "15:15"
TOP_N = 3  # matches EARLY_MOMENTUM_COUNT


def _cached_symbols() -> list[str]:
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _simulate_two_stage(day_df_full: pd.DataFrame, entry_pos: int, direction: SignalDirection,
                         stop: float, target_1: float, target_2: float) -> dict | None:
    if entry_pos >= len(day_df_full):
        return None
    entry_price = float(day_df_full["open"].iloc[entry_pos])
    risk = abs(entry_price - stop)
    if risk <= 0:
        return None
    sign = 1.0 if direction == SignalDirection.LONG else -1.0

    t1_hit = False
    t1_price = None
    t1_pos = None
    stage1_exit_price = None
    stage1_exit_reason = None
    for i in range(entry_pos, len(day_df_full)):
        bar = day_df_full.iloc[i]
        ts = day_df_full.index[i]
        high, low = float(bar["high"]), float(bar["low"])
        if direction == SignalDirection.LONG:
            if low <= stop:
                stage1_exit_price, stage1_exit_reason = stop, "stop_loss"
                break
            if high >= target_1:
                t1_hit, t1_price, t1_pos = True, target_1, i
                break
        else:
            if high >= stop:
                stage1_exit_price, stage1_exit_reason = stop, "stop_loss"
                break
            if low <= target_1:
                t1_hit, t1_price, t1_pos = True, target_1, i
                break
        if ts.strftime("%H:%M") >= EOD_SQUARE_OFF_TIME:
            stage1_exit_price, stage1_exit_reason = float(bar["close"]), "eod_square_off"
            break
    if not t1_hit and stage1_exit_price is None:
        stage1_exit_price, stage1_exit_reason = float(day_df_full["close"].iloc[-1]), "eod_square_off"

    if not t1_hit:
        pnl_points = sign * (stage1_exit_price - entry_price)
        return {
            "entry_price": entry_price, "t1_hit": False, "t2_hit": False,
            "final_exit_reason": stage1_exit_reason,
            "blended_pnl_points": pnl_points, "blended_r_multiple": pnl_points / risk,
        }

    t2_hit = False
    stage2_exit_price = None
    stage2_exit_reason = None
    for i in range(t1_pos + 1, len(day_df_full)):
        bar = day_df_full.iloc[i]
        ts = day_df_full.index[i]
        high, low = float(bar["high"]), float(bar["low"])
        if direction == SignalDirection.LONG:
            if low <= stop:
                stage2_exit_price, stage2_exit_reason = stop, "stop_loss"
                break
            if high >= target_2:
                t2_hit = True
                stage2_exit_price, stage2_exit_reason = target_2, "target_2"
                break
        else:
            if high >= stop:
                stage2_exit_price, stage2_exit_reason = stop, "stop_loss"
                break
            if low <= target_2:
                t2_hit = True
                stage2_exit_price, stage2_exit_reason = target_2, "target_2"
                break
        if ts.strftime("%H:%M") >= EOD_SQUARE_OFF_TIME:
            stage2_exit_price, stage2_exit_reason = float(bar["close"]), "eod_square_off"
            break
    if stage2_exit_price is None:
        stage2_exit_price, stage2_exit_reason = float(day_df_full["close"].iloc[t1_pos]), "eod_square_off"

    pnl_half1 = sign * (t1_price - entry_price)
    pnl_half2 = sign * (stage2_exit_price - entry_price)
    blended_pnl = 0.5 * pnl_half1 + 0.5 * pnl_half2
    return {
        "entry_price": entry_price, "t1_hit": True, "t2_hit": t2_hit,
        "final_exit_reason": f"t1_then_{stage2_exit_reason}",
        "blended_pnl_points": blended_pnl, "blended_r_multiple": blended_pnl / risk,
    }


def _process_day(day, symbol_data: dict[str, pd.DataFrame], sector_mapping: dict[str, str],
                  log: logging.Logger) -> list[dict]:
    already_entered: set[str] = set()
    day_trades: list[dict] = []

    for cp in CHECKPOINTS:
        sample_symbol = next(iter(symbol_data))
        tz = symbol_data[sample_symbol].index.tz
        as_of = pd.Timestamp(f"{day} {cp}:00").tz_localize(tz) if tz is not None else pd.Timestamp(f"{day} {cp}:00")

        try:
            picks = find_early_momentum_picks(symbol_data, sector_mapping, as_of=as_of, top_n=TOP_N)
        except Exception as e:
            log.warning(f"  {day} {cp}: find_early_momentum_picks error -- {e}")
            continue

        for pick in picks:
            if pick.symbol in already_entered:
                continue
            already_entered.add(pick.symbol)

            df = symbol_data[pick.symbol]
            day_df_full = df[df.index.date == day]
            today_df = df.loc[:as_of]
            today_df = today_df[today_df.index.date == day]
            entry_pos = len(today_df)

            outcome = _simulate_two_stage(day_df_full, entry_pos, pick.direction, pick.stop_loss, pick.target_1, pick.target_2)
            if outcome is None:
                continue
            day_trades.append({
                "symbol": pick.symbol, "day": str(day), "checkpoint": cp, "direction": pick.direction.value,
                "signal_move_pct": pick.move_pct_from_open, "signal_volume_zscore": pick.volume_zscore,
                "sector": pick.sector, "sector_aligned": pick.sector_aligned,
                "stop_loss": pick.stop_loss, "target_1": pick.target_1, "target_2": pick.target_2,
                **outcome,
            })

    return day_trades


def _append_day_result(day, trades: list[dict]) -> None:
    PARTIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL_RESULTS_PATH, "a") as f:
        f.write(json.dumps({"day": str(day), "trades": trades}) + "\n")


def _load_completed_days() -> dict[str, list[dict]]:
    if not PARTIAL_RESULTS_PATH.exists():
        return {}
    by_day: dict[str, list[dict]] = {}
    with open(PARTIAL_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_day[record["day"]] = record["trades"]
    return by_day


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()], force=True,
    )
    log = logging.getLogger("early_momentum_historical")
    log.info("=" * 70)
    log.info("EARLY MOMENTUM HISTORICAL SIMULATION -- RUN START (checkpoint/resume enabled, single-threaded)")
    log.info("=" * 70)

    t0 = time.monotonic()
    symbols = _cached_symbols()
    symbol_data = {s: storage.load_combined(s) for s in symbols}
    symbol_data = {
        s: df for s, df in symbol_data.items()
        if not df.empty and float(df["close"].iloc[-1]) >= MIN_STOCK_PRICE
    }
    log.info(f"Loaded {len(symbol_data)} liquid-price symbols via load_combined() in {time.monotonic()-t0:.1f}s")

    sector_mapping = json.loads(SECTOR_MAPPING_PATH.read_text()) if SECTOR_MAPPING_PATH.exists() else {}
    log.info(f"Sector mapping: {len(sector_mapping)} symbols")

    all_days = sorted({d for df in symbol_data.values() for d in set(df.index.date)})
    log.info(f"Distinct trading days across combined archive+rolling cache: {len(all_days)} "
             f"({all_days[0]} to {all_days[-1]})")

    completed = _load_completed_days()
    already_done = [d for d in all_days if str(d) in completed]
    remaining = [d for d in all_days if str(d) not in completed]
    log.info(f"Resume check ({PARTIAL_RESULTS_PATH.name}): {len(already_done)}/{len(all_days)} days already "
              f"complete, skipping. {len(remaining)} remaining.")

    start_time = time.monotonic()
    for i, day in enumerate(remaining):
        day_trades = _process_day(day, symbol_data, sector_mapping, log)
        _append_day_result(day, day_trades)
        elapsed = time.monotonic() - start_time
        done_this_run = i + 1
        eta_min = (elapsed / done_this_run) * (len(remaining) - done_this_run) / 60
        log.info(f"[{len(already_done) + done_this_run}/{len(all_days)}] {day}: {len(day_trades)} entries "
                  f"this day, ETA {eta_min:.1f} min")

    log.info(f"Simulation pass complete in {(time.monotonic()-start_time)/60:.1f} min. Aggregating...")

    all_completed = _load_completed_days()
    all_trades: list[dict] = []
    for day in all_days:
        all_trades.extend(all_completed.get(str(day), []))

    log.info("=" * 70)
    log.info("STEP 1: SIGNAL VOLUME")
    log.info("=" * 70)
    log.info(f"Total signals fired: {len(all_trades)}")
    log.info(f"Distinct trading days: {len(all_days)}")
    log.info(f"Avg signals/day: {len(all_trades) / len(all_days) if all_days else 0:.3f}")

    longs = [t for t in all_trades if t["direction"] == "long"]
    shorts = [t for t in all_trades if t["direction"] == "short"]

    def _report_group(name: str, trades: list[dict]) -> None:
        n = len(trades)
        log.info(f"--- {name} (n={n}) ---")
        if n == 0:
            log.info("  No trades.")
            return
        t1_hits = sum(1 for t in trades if t["t1_hit"])
        t2_hits = sum(1 for t in trades if t["t2_hit"])
        win_rate = t1_hits / n
        t2_hit_rate_of_t1 = (t2_hits / t1_hits) if t1_hits else 0.0

        pseudo_trades = [
            Trade(
                symbol=t["symbol"], direction=SignalDirection(t["direction"]),
                entry_time=pd.Timestamp(f"{t['day']} 10:00"), entry_price=t["entry_price"],
                stop_loss=t["stop_loss"], target=t["target_1"], reasoning=[], confidence=0.5,
                strategy_name="early_momentum", exit_time=pd.Timestamp(f"{t['day']} 15:15"),
                exit_price=t["entry_price"] + t["blended_pnl_points"] * (1 if t["direction"] == "long" else -1),
                exit_reason=t["final_exit_reason"],
            )
            for t in trades
        ]
        report = compute(pseudo_trades)

        log.info(f"  Win rate (target_1 hit rate): {win_rate:.1%}  ({t1_hits}/{n})")
        log.info(f"  Target_2 hit rate (of target_1 hits): {t2_hit_rate_of_t1:.1%}  ({t2_hits}/{t1_hits if t1_hits else 0})")
        log.info(f"  Avg blended return/trade: {sum(t['blended_pnl_points'] for t in trades)/n:+.3f} pts, "
                  f"avg R: {sum(t['blended_r_multiple'] for t in trades)/n:+.3f}")
        pf_str = f"{report.profit_factor:.2f}" if report.profit_factor != float("inf") else "inf"
        log.info(f"  Profit factor: {pf_str}")
        log.info(f"  Sharpe ratio (per-trade, not annualized): {report.sharpe_ratio:.2f}")
        log.info(f"  Max drawdown: {report.max_drawdown_points:.2f} pts")

    log.info("=" * 70)
    log.info("STEP 2-6: OVERALL RESULTS")
    log.info("=" * 70)
    _report_group("ALL TRADES", all_trades)

    log.info("=" * 70)
    log.info("STEP 7: PER-DIRECTION BREAKDOWN")
    log.info("=" * 70)
    _report_group("LONG", longs)
    _report_group("SHORT", shorts)

    log.info("=" * 70)
    log.info("SPOT CHECK: 10 real trades")
    log.info("=" * 70)
    import random
    random.seed(7)
    sample = random.sample(all_trades, min(10, len(all_trades)))
    for t in sample:
        log.info(f"  {t['symbol']:12s} {t['day']} {t['checkpoint']} {t['direction']:5s} "
                  f"entry={t['entry_price']:.2f} stop={t['stop_loss']:.2f} "
                  f"T1={t['target_1']:.2f} T2={t['target_2']:.2f} "
                  f"t1_hit={t['t1_hit']} t2_hit={t['t2_hit']} exit={t['final_exit_reason']} "
                  f"blended_pnl={t['blended_pnl_points']:+.2f} blended_R={t['blended_r_multiple']:+.2f}")

    log.info("RUN COMPLETE")


if __name__ == "__main__":
    main()
