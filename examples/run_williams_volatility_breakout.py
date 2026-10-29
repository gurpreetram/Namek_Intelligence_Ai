"""
Tests Larry Williams' Volatility Breakout: unlike this project's own ORB
(examples/opening_range_breakout.py, which needs TODAY's first-15-min
range to form before it has a trigger), Williams' trigger is known
INSTANTLY at market open, since it's built from YESTERDAY's already-
complete full-day range applied to TODAY's open -- the scan starts at
the very first 1-min bar of the day, not after a formation window.

PRE-REGISTERED BARS (same as every other hypothesis test in this
project): a (k, exit-style) combination is a real edge only if BOTH
hold on its day-level return series: day-level Sharpe (non-annualized)
>= 0.5 AND day-level win rate >= 60%.

DEFINITION (per symbol, per day D):
  yesterday_range = yesterday's high - yesterday's low (day D-1's cached
      1-min data, max high minus min low across the whole session).
  today_open = day D's first cached bar's open.
  buy_trigger  = today_open + k * yesterday_range
  sell_trigger = today_open - k * yesterday_range
Scanned bar-by-bar from the OPEN (not after a formation window -- see
above): the FIRST bar where close > buy_trigger (long) OR
close < sell_trigger (short) wins. Only one direction can ever fire on
a given bar (sell_trigger < today_open < buy_trigger for any k>0), so
there's no tie to break. Three k values tested separately: 0.25, 0.5
(Williams' commonly-cited default), 0.75.

NO VOLUME CONFIRMATION -- deliberately different from this project's own
ORB/crash/compression detectors, which all require an elevated-volume
check. Williams' original method and the user's literal spec don't
include one; adding one here would silently test a different strategy.

LIQUIDITY FILTER APPLIED FROM THE START (yesterday's data, before
anything else is computed for that symbol-day) -- detectors.
liquidity_analysis.analyze_liquidity(), same discipline, same order, as
every prior script in this project.

ENTRY: next 1-min bar's OPEN strictly after the trigger bar (no-
lookahead). Detection runs ONCE per k (3 scans, not 6) -- both exit
styles below are computed from that SAME shared entry per event.

EXIT STYLE A -- EOD square-off at or before 15:15, via detectors.
cross_sectional_ranking.bar_at_or_before() (reused as-is). This is an
explicit ADAPTATION from Williams' original multi-day hold, forced by
this project being intraday-only -- stated plainly, not glossed over.

EXIT STYLE B -- fixed 4x-risk target, risk unit = yesterday_range itself
(NOT re-scaled by k, per the literal instruction): target = entry_price
+/- 4*yesterday_range. Scanned bar-by-bar from entry to the 15:15 bar:
the first bar whose high (long) / low (short) reaches the target exits
AT the target price (matches this project's existing "target hit"
convention); otherwise falls through to the same EOD close as style A.
CAVEAT, stated explicitly and carried into every report of these
results: this exit style has NO protective stop-loss -- the user's
request frames it purely as an early profit-target check, so an adverse
move is not cut short intraday under this variant. This is a real
limitation of the test, not an oversight.

Reuses examples/run_crash_detection.py's day-level Sharpe/win-rate/
verdict/concentration helpers exactly as run_sr_compression_breakout.py
already does, rather than reimplementing them.

Run: python -m examples.run_williams_volatility_breakout
    python -m examples.run_williams_volatility_breakout --limit 300   # fast test
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from config.settings import STORAGE_ROOT
from data import storage
from detectors.cross_sectional_ranking import bar_at_or_before
from detectors.liquidity_analysis import analyze_liquidity
from detectors.scanner import MIN_STOCK_PRICE
from examples.run_crash_detection import (
    COST_PCT_PER_ROUND_TRIP, MIN_SHARPE_BAR, MIN_WINRATE_BAR,
    _day_stats, _report_top_n_concentration, _verdict,
)

K_VALUES = (0.25, 0.5, 0.75)
EOD_SQUARE_OFF_TIME = "15:15"
TARGET_RISK_MULTIPLE = 4.0
FROZEN_RUN_CHECK_BARS = 10
DATA_GAP_EARLY_EXIT_MINUTES = 30
CIRCUIT_BAND_PCTS = (2.0, 5.0, 10.0, 20.0)
CIRCUIT_BAND_TOLERANCE = 0.3


@dataclass
class TriggerEvent:
    symbol: str
    day: date
    k: float
    direction: str  # "long" | "short"
    yesterday_range: float
    today_open: float
    trigger_price: float  # the buy_trigger or sell_trigger that fired
    trigger_time: pd.Timestamp
    avg_traded_value: float
    entry_time: pd.Timestamp
    entry_price: float
    trigger_bar_frozen: bool
    frozen_run_after: int


@dataclass
class ExitResult:
    day: date
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_style: str  # "eod" | "target_4r"
    hit_target: bool


def _frozen(bar: pd.Series) -> bool:
    return float(bar["open"]) == float(bar["high"]) == float(bar["low"]) == float(bar["close"])


def _frozen_run_after(day_df: pd.DataFrame, ts: pd.Timestamp, n_bars: int) -> int:
    after = day_df[day_df.index > ts].iloc[:n_bars]
    run = 0
    for _, bar in after.iterrows():
        if not _frozen(bar):
            break
        run += 1
    return run


def _find_trigger_events(symbol_data: dict[str, pd.DataFrame], k: float, require_liquid: bool = True) -> list[TriggerEvent]:
    events: list[TriggerEvent] = []
    for symbol, df in symbol_data.items():
        days = sorted(set(df.index.date))
        for i in range(1, len(days)):
            day = days[i]
            prev_day_df = df[df.index.date == days[i - 1]]
            if prev_day_df.empty:
                continue
            yesterday_close = float(prev_day_df["close"].iloc[-1])
            if yesterday_close < MIN_STOCK_PRICE:
                continue
            liquidity = analyze_liquidity(prev_day_df)
            if require_liquid and (liquidity is None or not liquidity.is_liquid):
                continue

            yesterday_range = float(prev_day_df["high"].max() - prev_day_df["low"].min())
            if yesterday_range <= 0:
                continue

            day_df = df[df.index.date == day]
            if day_df.empty:
                continue
            today_open = float(day_df["open"].iloc[0])
            buy_trigger = today_open + k * yesterday_range
            sell_trigger = today_open - k * yesterday_range

            long_mask = day_df["close"] > buy_trigger
            short_mask = day_df["close"] < sell_trigger
            long_time = long_mask[long_mask].index[0] if long_mask.any() else None
            short_time = short_mask[short_mask].index[0] if short_mask.any() else None

            if long_time is not None and (short_time is None or long_time <= short_time):
                direction, trigger_time, trigger_price = "long", long_time, buy_trigger
            elif short_time is not None:
                direction, trigger_time, trigger_price = "short", short_time, sell_trigger
            else:
                continue

            entry_candidates = day_df[day_df.index > trigger_time]
            if entry_candidates.empty:
                continue
            entry_bar = entry_candidates.iloc[0]
            trigger_bar = day_df.loc[trigger_time]

            events.append(TriggerEvent(
                symbol=symbol, day=day, k=k, direction=direction, yesterday_range=yesterday_range,
                today_open=today_open, trigger_price=trigger_price, trigger_time=trigger_time,
                avg_traded_value=liquidity.avg_traded_value if liquidity is not None else float("nan"),
                entry_time=entry_bar.name, entry_price=float(entry_bar["open"]),
                trigger_bar_frozen=_frozen(trigger_bar),
                frozen_run_after=_frozen_run_after(day_df, trigger_time, FROZEN_RUN_CHECK_BARS),
            ))
    return events


def _resolve_exit(event: TriggerEvent, df: pd.DataFrame, exit_style: str) -> ExitResult | None:
    day_df = df[df.index.date == event.day]
    eod_ts = pd.Timestamp.combine(event.day, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(day_df.index.tz)
    eod_bar = bar_at_or_before(df, eod_ts)
    if eod_bar is None or eod_bar.name <= event.entry_time:
        return None

    if exit_style == "eod":
        return ExitResult(day=event.day, entry_price=event.entry_price, exit_time=eod_bar.name,
                           exit_price=float(eod_bar["close"]), exit_style="eod", hit_target=False)

    # target_4r: risk unit is yesterday_range itself, not re-scaled by k.
    target = (event.entry_price + TARGET_RISK_MULTIPLE * event.yesterday_range if event.direction == "long"
              else event.entry_price - TARGET_RISK_MULTIPLE * event.yesterday_range)
    path = day_df[(day_df.index > event.entry_time) & (day_df.index <= eod_bar.name)]
    if path.empty:
        return ExitResult(day=event.day, entry_price=event.entry_price, exit_time=eod_bar.name,
                           exit_price=float(eod_bar["close"]), exit_style="target_4r", hit_target=False)

    if event.direction == "long":
        hit_mask = path["high"] >= target
    else:
        hit_mask = path["low"] <= target
    if hit_mask.any():
        hit_time = hit_mask[hit_mask].index[0]
        return ExitResult(day=event.day, entry_price=event.entry_price, exit_time=hit_time,
                           exit_price=target, exit_style="target_4r", hit_target=True)
    return ExitResult(day=event.day, entry_price=event.entry_price, exit_time=eod_bar.name,
                       exit_price=float(eod_bar["close"]), exit_style="target_4r", hit_target=False)


@dataclass
class ReportableEvent:
    """Minimal shape _report_mixed/_report_top_n_concentration need via
    duck typing (.day/.entry_price/.exit_price), carrying the source
    TriggerEvent + ExitResult for the spot-check."""
    day: date
    entry_price: float
    exit_price: float
    trigger: TriggerEvent
    exit: ExitResult


def _data_gap_flag(re: ReportableEvent) -> str | None:
    eod_minutes = 15 * 60 + 15
    exit_minutes = re.exit.exit_time.hour * 60 + re.exit.exit_time.minute
    if not re.exit.hit_target and eod_minutes - exit_minutes >= DATA_GAP_EARLY_EXIT_MINUTES:
        return (f"POSSIBLE DATA GAP: exit found at {re.exit.exit_time.strftime('%H:%M')}, "
                f"{eod_minutes - exit_minutes} min before the 15:15 EOD target -- treat with caution")
    return None


def _circuit_flag(re: ReportableEvent) -> str:
    move_pct = abs(re.trigger.trigger_price - re.trigger.today_open) / re.trigger.today_open * 100.0
    near_band = any(abs(move_pct - band) <= CIRCUIT_BAND_TOLERANCE for band in CIRCUIT_BAND_PCTS)
    if re.trigger.trigger_bar_frozen and re.trigger.frozen_run_after >= 2:
        return f"LIKELY CIRCUIT LOCK (frozen trigger bar + {re.trigger.frozen_run_after} more frozen bars immediately after)"
    if re.trigger.trigger_bar_frozen:
        return "trigger bar frozen (O=H=L=C) -- worth checking, not necessarily a circuit hit"
    if near_band:
        return "trigger distance near a standard circuit band, but trigger bar NOT frozen -- likely genuine"
    return "no circuit-lock indicators"


def _report_mixed(label: str, events: list[ReportableEvent]) -> tuple[float, float, int]:
    """Same day-level Sharpe/win-rate/verdict logic as run_crash_detection.py's
    _report_individual(), but with a PER-EVENT sign instead of one fixed sign
    for the whole list -- needed here because a single (k, exit_style)
    combination mixes long and short events (direction is whichever trigger
    fired first each day), unlike every prior script's tests, which always
    applied one fixed direction to an entire event list."""
    if not events:
        print(f"\n--- {label} ---\nNo qualifying events.")
        return 0.0, 0.0, 0
    pos_returns = []
    by_day: dict[date, list[float]] = {}
    for re in events:
        sign = 1.0 if re.trigger.direction == "long" else -1.0
        ret_pct = sign * (re.exit_price - re.entry_price) / re.entry_price * 100.0
        pos_returns.append(ret_pct)
        by_day.setdefault(re.day, []).append(ret_pct)
    pos_returns = np.array(pos_returns)
    day_returns = np.array([float(np.mean(v)) for v in by_day.values()])
    win_rate, mean, std, sharpe = _day_stats(day_returns)
    pos_win_rate = float((pos_returns > 0).mean())
    print(f"\n--- {label} ---")
    print(f"Events: {len(events)}  |  Days used: {len(by_day)}")
    print(f"Position-level win rate: {pos_win_rate:.1%}  avg return: {pos_returns.mean():+.3f}%  median: {np.median(pos_returns):+.3f}%")
    print(f"Day-level win rate: {win_rate:.1%}  avg daily return: {mean:+.3f}%  std: {std:.3f}%  Sharpe: {sharpe:.2f}")
    print(f"PRE-REGISTERED BAR (Sharpe>=0.5 AND win rate>=60%): {_verdict(sharpe, win_rate)}")
    return sharpe, win_rate, len(by_day)


def _spot_check(events: list[ReportableEvent], n: int, label: str) -> None:
    print(f"\n--- SPOT-CHECK ({label}): {min(n, len(events))} real events ---")
    for re in sorted(events, key=lambda e: abs(e.trigger.trigger_price - e.trigger.today_open), reverse=True)[:n]:
        t = re.trigger
        print(f"  {t.symbol:14s} {t.day}  k={t.k}  {t.direction:5s}  yesterday_range={t.yesterday_range:.2f}  "
              f"open={t.today_open:.2f}  trigger={t.trigger_price:.2f} ({t.trigger_time.strftime('%H:%M')})  "
              f"liquidity=Rs.{t.avg_traded_value:,.0f}/min")
        print(f"    entry={re.entry_price:.2f} ({t.entry_time.strftime('%H:%M')})  "
              f"exit={re.exit_price:.2f} ({re.exit.exit_time.strftime('%H:%M')}, {re.exit.exit_style}"
              f"{', TARGET HIT' if re.exit.hit_target else ''})")
        print(f"    {_circuit_flag(re)}")
        gap_flag = _data_gap_flag(re)
        if gap_flag:
            print(f"    {gap_flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="cap universe size, for a fast test run")
    parser.add_argument("--no-liquidity-filter", action="store_true")
    parser.add_argument("--cost-pct", type=float, default=COST_PCT_PER_ROUND_TRIP)
    args = parser.parse_args()
    require_liquid = not args.no_liquidity_filter

    symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    if args.limit is not None:
        symbols = symbols[:args.limit]

    print(f"Loading {len(symbols)} cached symbols into memory...")
    t0 = time.monotonic()
    symbol_data: dict[str, pd.DataFrame] = {}
    for i, s in enumerate(symbols, 1):
        df = storage.load(s)
        if not df.empty:
            symbol_data[s] = df
        if i % 500 == 0:
            print(f"  loaded {i}/{len(symbols)}...", flush=True)
    print(f"Loaded {len(symbol_data)}/{len(symbols)} symbols with usable data in {time.monotonic() - t0:.1f}s\n")

    print("=" * 70)
    print(f"STEP 1: Williams Volatility Breakout detection, liquidity filter "
          f"{'ON' if require_liquid else 'OFF'} FROM THE START")
    print("=" * 70)

    results: dict[str, tuple[float, float, int]] = {}
    reportable_by_combo: dict[str, list[ReportableEvent]] = {}

    for k in K_VALUES:
        t0 = time.monotonic()
        trigger_events = _find_trigger_events(symbol_data, k, require_liquid=require_liquid)
        all_days = sorted({e.day for e in trigger_events})
        print(f"\nk={k}: {len(trigger_events)} trigger events across {len(all_days)} distinct trading days "
              f"({time.monotonic() - t0:.1f}s)")

        for exit_style, exit_label in (("eod", "EOD square-off"), ("target_4r", "4R target (early exit if hit)")):
            reportable: list[ReportableEvent] = []
            for ev in trigger_events:
                exit_res = _resolve_exit(ev, symbol_data[ev.symbol], exit_style)
                if exit_res is None:
                    continue
                reportable.append(ReportableEvent(day=ev.day, entry_price=ev.entry_price,
                                                   exit_price=exit_res.exit_price, trigger=ev, exit=exit_res))

            if exit_style == "target_4r" and reportable:
                hit_rate = sum(1 for re in reportable if re.exit.hit_target) / len(reportable)
                print(f"\n4R target (yesterday_range x {TARGET_RISK_MULTIPLE:.0f}) hit intraday before EOD: "
                      f"{hit_rate:.1%} of events ({sum(1 for re in reportable if re.exit.hit_target)}/{len(reportable)}) "
                      f"-- the rest fall through to the same EOD close as the plain EOD-square-off style.")

            label = f"k={k}, {exit_label}"
            s, w, d = _report_mixed(label, reportable)
            results[label] = (s, w, d)
            reportable_by_combo[label] = reportable

    print("\n" + "=" * 70)
    print("STEP: concentration test (only combos clearing BOTH pre-registered bars)")
    print("=" * 70)
    winners = [label for label, (s, w, _) in results.items() if s >= MIN_SHARPE_BAR and w >= MIN_WINRATE_BAR]
    if not winners:
        print("\nNo combination cleared both pre-registered bars -- skipping concentration test.")
    else:
        print(f"\nCombination(s) clearing both bars: {', '.join(winners)}")
        for label in winners:
            evs = reportable_by_combo[label]
            labeled = [
                (re.day, abs(re.trigger.trigger_price - re.trigger.today_open),
                 (1.0 if re.trigger.direction == "long" else -1.0) * (re.exit_price - re.entry_price) / re.entry_price * 100.0)
                for re in evs
            ]
            for n in (1, 3, 5):
                _report_top_n_concentration(labeled, n, args.cost_pct, f"{label}, ranked by trigger distance")

    print("\n" + "=" * 70)
    print("STEP: spot-check (BEFORE final verdict)")
    print("=" * 70)
    best_label = max(results, key=lambda l: results[l][0]) if results else None
    if best_label:
        _spot_check(reportable_by_combo[best_label], 10, best_label)
        evs = reportable_by_combo[best_label]
        n_frozen = sum(1 for re in evs if re.trigger.trigger_bar_frozen)
        n_frozen_run = sum(1 for re in evs if re.trigger.frozen_run_after >= 2)
        n_gap = sum(1 for re in evs if _data_gap_flag(re) is not None)
        print(f"\nAcross this variant's {len(evs)} events: {n_frozen} frozen trigger bars, "
              f"{n_frozen_run} with a frozen run after, {n_gap} likely data-gap early exits.")

    print("\n" + "=" * 70)
    print("HONEST VERDICT (pre-registered bars: Sharpe>=0.5 AND win rate>=60%)")
    print("=" * 70)
    print(f"{'Combination':<38s} {'Days':>6s} {'Sharpe':>8s} {'WinRate':>9s}  Verdict")
    for label, (s, w, d) in results.items():
        print(f"{label:<38s} {d:>6d} {s:>8.2f} {w:>8.1%}  {_verdict(s, w)}")


if __name__ == "__main__":
    main()
