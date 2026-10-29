"""
Historical simulation of detectors/best_picks.py's find_best_picks()
across the full archived history -- built to get honest performance
numbers before treating "Today's Best 2 Picks" (the F8 tab) as reliable.

ARCHITECTURE NOTE, different from examples/run_momentum_entry_historical_
simulation.py: that script parallelizes per SYMBOL, because each symbol's
gate evaluation is fully independent. find_best_picks() is NOT
independent per symbol -- it RANKS every liquid symbol's candidates
together and returns only the global top 2 at each checkpoint. Splitting
that ranking across worker processes would mean either re-loading all
~1,626 symbols' data in every worker (this host has 7.8GB RAM; multiple
duplicate full-universe copies risks the same paging-file failure seen
earlier this session) or a much more complex shared-memory design. Given
a single full-universe find_best_picks() call takes ~10s and there are
only ~225 (day, checkpoint) pairs to evaluate (~45 archived days x 5
checkpoints), a single-threaded sequential pass (~40-50 min total) is the
simpler, memory-safe choice here -- not parallelized.

METHODOLOGY:
For each archived trading day, at 5 checkpoints (10:00, 11:00, 12:00,
13:00, 14:00 IST), call find_best_picks() against the full liquid
universe truncated to that moment -- exactly the real function, not a
reimplementation. Per (symbol, day), only the FIRST checkpoint where that
symbol appears in the top 2 counts as an entry (a symbol staying in the
top 2 across multiple checkpoints the same day is the same real trading
opportunity shown again, not a new one -- find_best_picks() itself has no
_traded_today state, so this script supplies that discipline for
backtesting purposes).

TWO-STAGE PARTIAL-EXIT SIMULATION for each entry: entry at the NEXT 1-min
bar's open after the qualifying checkpoint (no lookahead). Stage 1: walk
forward until stop-loss or target_1 is hit (stop checked first), or EOD
square-off (15:15) if neither. If target_1 is hit first, 50% of the
position exits there ("T1_hit") and stage 2 begins: the remaining 50%
continues from the NEXT bar, walking forward against the SAME original
stop (not moved to breakeven -- not requested) until that stop, target_2,
or EOD closes it. If the stop is hit before target_1, the FULL position
exits there (no partial). Blended trade P&L = 0.5x each stage's exit
price effect when target_1 was reached; 1x the single exit otherwise.

"Win" for the requested win-rate metric means target_1 was reached --
NOT that the blended P&L was positive (a trade can hit target_1, then
give back more than that on the remaining half before target_2/stop/EOD,
landing net-negative despite being a "win" by this definition -- both
numbers are reported honestly, not conflated).

CHECKPOINT/RESUME: appends one line per COMPLETED DAY (all 5 checkpoints'
results for that day) to logs/best_picks_historical_partial_results.jsonl
immediately after that day finishes -- a day already present is skipped
on resume. Progress logs to logs/universe_run.log.

Run: python -m examples.run_best_picks_historical_simulation
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
from detectors.best_picks import find_best_picks
from detectors.scanner import MIN_STOCK_PRICE

LOG_PATH = PROJECT_ROOT / "logs" / "universe_run.log"
PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "best_picks_historical_partial_results.jsonl"

CHECKPOINTS = ["10:00", "11:00", "12:00", "13:00", "14:00"]
EOD_SQUARE_OFF_TIME = "15:15"


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

    # Stage 1: stop vs target_1, or EOD.
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

    # Stage 2: remaining 50% from the bar after target_1 was hit, against the SAME original stop.
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
        # target_1 was hit on the day's last bar -- remaining 50% closes at that same bar's close.
        stage2_exit_price, stage2_exit_reason = float(day_df_full["close"].iloc[t1_pos]), "eod_square_off"

    pnl_half1 = sign * (t1_price - entry_price)
    pnl_half2 = sign * (stage2_exit_price - entry_price)
    blended_pnl = 0.5 * pnl_half1 + 0.5 * pnl_half2
    return {
        "entry_price": entry_price, "t1_hit": True, "t2_hit": t2_hit,
        "final_exit_reason": f"t1_then_{stage2_exit_reason}",
        "blended_pnl_points": blended_pnl, "blended_r_multiple": blended_pnl / risk,
    }


def _process_day(day, symbol_data: dict[str, pd.DataFrame], log: logging.Logger) -> list[dict]:
    already_entered: set[str] = set()
    day_trades: list[dict] = []

    for cp in CHECKPOINTS:
        sample_symbol = next(iter(symbol_data))
        tz = symbol_data[sample_symbol].index.tz
        as_of = pd.Timestamp(f"{day} {cp}:00").tz_localize(tz) if tz is not None else pd.Timestamp(f"{day} {cp}:00")

        try:
            picks = find_best_picks(symbol_data, as_of=as_of, top_n=2)
        except Exception as e:
            log.warning(f"  {day} {cp}: find_best_picks error -- {e}")
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
    log = logging.getLogger("best_picks_historical")
    log.info("=" * 70)
    log.info("BEST PICKS HISTORICAL SIMULATION -- RUN START (checkpoint/resume enabled, single-threaded)")
    log.info("=" * 70)

    t0 = time.monotonic()
    symbols = _cached_symbols()
    symbol_data = {s: storage.load_combined(s) for s in symbols}
    symbol_data = {
        s: df for s, df in symbol_data.items()
        if not df.empty and float(df["close"].iloc[-1]) >= MIN_STOCK_PRICE
    }
    log.info(f"Loaded {len(symbol_data)} liquid-price symbols via load_combined() in {time.monotonic()-t0:.1f}s")

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
        day_trades = _process_day(day, symbol_data, log)
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

        # Trade.pnl_points/r_multiple are @property (computed from exit_price/
        # entry_price/stop_loss), not constructor kwargs -- exit_price below
        # is reverse-engineered so the property recomputes EXACTLY the
        # already-simulated blended_pnl_points (and, since stop_loss is
        # passed as the trade's real original stop, r_multiple too): for
        # LONG, pnl_points = exit_price - entry_price, so exit_price =
        # entry_price + blended_pnl_points reproduces it exactly (mirrored
        # sign for SHORT) -- verified by construction, not just asserted.
        pseudo_trades = [
            Trade(
                symbol=t["symbol"], direction=SignalDirection(t["direction"]),
                entry_time=pd.Timestamp(f"{t['day']} 10:00"), entry_price=t["entry_price"],
                stop_loss=t["stop_loss"], target=t["target_1"], reasoning=[], confidence=0.5,
                strategy_name="best_picks", exit_time=pd.Timestamp(f"{t['day']} 15:15"),
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
