"""
Four refinements on examples/run_crash_detection.py's crash-detection
hypothesis (which found NO edge in any of its 4 original tested forms --
see that module's docstring and logs/crash_detection_full.log). Reuses
that module's CrashEvent dataclass, liquidity-upfront + circuit-lock/
data-gap spot-check helpers, and day-level Sharpe/win-rate reporting
as-is -- this module only adds NEW detection variants, not a
reimplementation of the shared plumbing.

PRE-REGISTERED BARS (same as every other hypothesis test in this
project, unchanged from run_crash_detection.py): a (variant, direction)
combination is a real, worth-considering edge only if BOTH hold on its
day-level return series: day-level Sharpe (non-annualized) >= 0.5 AND
day-level win rate >= 60%. Applied identically to every combination
below, set BEFORE running this and not adjusted after seeing results.

REFINEMENT 1 -- confirmation-based entry: trigger window narrowed to
09:15-09:25 (was 09:15-09:30); the first bar there meeting the original
crash criteria (<=-2.5% since open, volume_zscore>=1.0 against a
time-of-day-matched prior-day baseline, exactly run_crash_detection.py's
methodology) is a CANDIDATE, not yet an event. It only becomes an event
if, at the bar at-or-before 09:30, the decline is STILL <=-2.5% (hasn't
bounced back above the threshold) -- filtering for a persistent,
still-ongoing crash rather than a one-bar spike that already reverted.
Entry unchanged: next bar's open after 09:30.

REFINEMENT 2 -- severity tiers: re-buckets run_crash_detection.py's
ORIGINAL 420 events (re-run here via its own _find_crash_events(), not
reimplemented, for exact consistency with the already-reported result)
by |decline_pct| into moderate (2.5-4%), severe (4-6%), extreme (>=6%),
each tested both directions.

REFINEMENT 3 -- melt-up (mirror image): identical methodology to the
original crash detector, just threshold_pct=+2.5 instead of -2.5 (rapid
RISE instead of decline), same 09:15-09:30 window, same time-of-day-
matched volume baseline, same liquidity filter, same fixed 09:30 entry.

REFINEMENT 4 -- crash detection all day (09:15-15:00), not just the
open. Two things HAVE to change vs the original method, not just the
window, to keep this a fair, honest test rather than trivially different:
  (a) return definition: no longer "since today's open" (measuring that
      at 13:00 would just be capturing the whole day's cumulative drift,
      not a "rapid" move) -- switched to a TRAILING 15-minute return
      (close now vs close 15 bars back, 1-min-bar row-shift used as a
      practical proxy for "15 minutes ago" -- an approximation that
      assumes no missed bars, same spirit as this project's other
      research passes accepting a documented simplification rather than
      building new infrastructure for a one-off test).
  (b) volume baseline: the original's "prior-day, time-of-day-matched
      pool" doesn't generalize to "any minute of the day" (there's no
      single matching slot to pool across days for an arbitrary minute).
      Switched to a SAME-DAY rolling baseline -- the preceding 30 bars
      on the SAME day (detectors.breakout_validator.volume_zscore's own
      original convention: current bar vs immediately preceding bars).
      This requires >=30 preceding same-day bars to exist, which
      naturally excludes the first ~30 minutes of trading from this
      variant's scan -- a deliberate, desirable side effect: it keeps
      this variant's population genuinely disjoint from the opening-
      window variants above (no overlap to muddy the "does timing
      matter" comparison), and avoids reintroducing the exact opening-
      volume bias run_crash_detection.py's docstring explains at length
      (opening volume >> immediately-preceding bars' volume on almost
      every stock on almost every day, which would make "elevated
      volume" trivially true near the open under this convention).
  Entry: next bar's open strictly after the trigger bar (no fixed 09:30
  cutoff makes sense when the trigger can land anywhere in the day).
  Events are also split into late-morning (09:45-12:00) and afternoon
  (12:00-15:00) buckets to directly test whether timing-within-day
  matters.

SPOT-CHECK BEFORE VERDICT: whichever combination above (if any) clears
both pre-registered bars gets 5-10 real examples printed -- symbol,
decline/rise %, liquidity, entry/exit prices -- plus the same circuit-
lock (frozen O=H=L=C trigger bar, frozen run after) and data-gap
(early "EOD" exit) checks run_crash_detection.py uses, BEFORE the final
verdict table, not after.

Run: python -m examples.run_crash_detection_refinements
    python -m examples.run_crash_detection_refinements --limit 500   # fast test
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
from detectors.liquidity_analysis import analyze_liquidity
from detectors.cross_sectional_ranking import bar_at_or_before
from detectors.scanner import MIN_STOCK_PRICE
from examples.run_crash_detection import (
    CrashEvent, EOD_SQUARE_OFF_TIME, FROZEN_RUN_CHECK_BARS, MIN_BASELINE_PRIOR_DAYS,
    MIN_SHARPE_BAR, MIN_WINRATE_BAR, _data_gap_flag, _circuit_flag, _find_crash_events,
    _frozen, _frozen_run_after, _load_sector_mapping, _report_individual, _spot_check, _verdict,
)

ENTRY_CUTOFF_TIME = "09:30"
CONFIRM_TRIGGER_WINDOW_END = "09:25"  # refinement 1's narrowed initial-detection window
ALLDAY_WINDOW_START = "09:15"
ALLDAY_WINDOW_END = "15:00"
ALLDAY_RETURN_LOOKBACK_MINUTES = 15
ALLDAY_SAME_DAY_LOOKBACK_BARS = 30


def _scan_events(
    symbol_data: dict[str, pd.DataFrame],
    sector_mapping: dict[str, str],
    require_liquid: bool,
    window_start: str,
    window_end: str,
    threshold_pct: float,
    volume_zscore_min: float,
    baseline_mode: str,
    min_baseline_prior_days: int = MIN_BASELINE_PRIOR_DAYS,
    same_day_lookback_bars: int = ALLDAY_SAME_DAY_LOOKBACK_BARS,
    return_lookback_minutes: int | None = None,
    entry_mode: str = "fixed_cutoff",
    entry_cutoff_time: str = ENTRY_CUTOFF_TIME,
    confirm_window_end: str | None = None,
    confirm_threshold_pct: float | None = None,
) -> list[CrashEvent]:
    """Vectorized (per-day, not per-bar-Python-call) generalization of
    run_crash_detection.py's _find_crash_events() -- see this module's
    docstring for why: the original's per-bar compute_stock_return_today()
    calls are individually cheap but, spread across the whole trading day
    (needed for refinement 4), add up to hours of runtime. Computing the
    same formulas (open-to-now or trailing-N-minute return; volume
    z-score) with plain pandas arithmetic over an already-sliced window
    is mathematically identical, just far faster."""
    events: list[CrashEvent] = []
    for symbol, df in symbol_data.items():
        days = sorted(set(df.index.date))
        pooled_window_volumes: list[float] = []
        prior_days_pooled = 0

        for i, day in enumerate(days):
            day_df = df[df.index.date == day]
            if day_df.empty:
                continue
            day_open = float(day_df["open"].iloc[0])

            eligible_day = i > 0 and day_open > 0
            liquidity = None
            if eligible_day:
                prev_day_df = df[df.index.date == days[i - 1]]
                yesterday_close = float(prev_day_df["close"].iloc[-1]) if not prev_day_df.empty else 0.0
                liquidity = analyze_liquidity(prev_day_df) if not prev_day_df.empty else None
                if yesterday_close < MIN_STOCK_PRICE:
                    eligible_day = False
                elif require_liquid and (liquidity is None or not liquidity.is_liquid):
                    eligible_day = False

            win_start_ts = pd.Timestamp.combine(day, pd.Timestamp(window_start).time()).tz_localize(day_df.index.tz)
            win_end_ts = pd.Timestamp.combine(day, pd.Timestamp(window_end).time()).tz_localize(day_df.index.tz)
            window_df = day_df[(day_df.index >= win_start_ts) & (day_df.index <= win_end_ts)]

            baseline_ready = prior_days_pooled >= min_baseline_prior_days if baseline_mode == "pooled_window" else True

            if eligible_day and baseline_ready and not window_df.empty:
                if return_lookback_minutes is None:
                    pct = (window_df["close"] - day_open) / day_open * 100.0
                else:
                    shifted = day_df["close"].shift(return_lookback_minutes)
                    pct_full = (day_df["close"] - shifted) / shifted * 100.0
                    pct = pct_full.reindex(window_df.index)

                if baseline_mode == "pooled_window":
                    baseline = pd.Series(pooled_window_volumes, dtype=float)
                    bmean, bstd = baseline.mean(), baseline.std()
                    if bstd and bstd > 0 and not pd.isna(bstd):
                        vol_z = (window_df["volume"] - bmean) / bstd
                    else:
                        vol_z = pd.Series(0.0, index=window_df.index)
                else:
                    roll_mean = day_df["volume"].shift(1).rolling(same_day_lookback_bars).mean()
                    roll_std = day_df["volume"].shift(1).rolling(same_day_lookback_bars).std()
                    vz_full = (day_df["volume"] - roll_mean) / roll_std
                    vol_z = vz_full.reindex(window_df.index)

                if threshold_pct < 0:
                    trigger_mask = (pct <= threshold_pct) & (vol_z >= volume_zscore_min)
                else:
                    trigger_mask = (pct >= threshold_pct) & (vol_z >= volume_zscore_min)

                if trigger_mask.fillna(False).any():
                    qualifying_times = trigger_mask[trigger_mask.fillna(False)].index
                    trigger_time = qualifying_times[0]
                    trigger_pct = float(pct.loc[trigger_time])
                    trigger_z = float(vol_z.loc[trigger_time])

                    confirmed = True
                    if confirm_window_end is not None:
                        confirm_ts = pd.Timestamp.combine(day, pd.Timestamp(confirm_window_end).time()).tz_localize(day_df.index.tz)
                        confirm_bars = day_df[day_df.index <= confirm_ts]
                        if confirm_bars.empty:
                            confirmed = False
                        else:
                            confirm_close = float(confirm_bars["close"].iloc[-1])
                            confirm_pct = (confirm_close - day_open) / day_open * 100.0
                            if confirm_threshold_pct is not None:
                                confirmed = (confirm_pct <= confirm_threshold_pct if confirm_threshold_pct < 0
                                             else confirm_pct >= confirm_threshold_pct)

                    if confirmed:
                        if entry_mode == "fixed_cutoff":
                            cutoff_ts = pd.Timestamp.combine(day, pd.Timestamp(entry_cutoff_time).time()).tz_localize(day_df.index.tz)
                            entry_candidates = day_df[day_df.index > cutoff_ts]
                        else:
                            entry_candidates = day_df[day_df.index > trigger_time]

                        if not entry_candidates.empty:
                            entry_bar = entry_candidates.iloc[0]
                            eod_ts = pd.Timestamp.combine(day, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(day_df.index.tz)
                            exit_bar = bar_at_or_before(df, eod_ts)
                            if exit_bar is not None and exit_bar.name > entry_bar.name:
                                trigger_bar = df.loc[trigger_time]
                                events.append(CrashEvent(
                                    symbol=symbol, day=day, sector=sector_mapping.get(symbol),
                                    trigger_time=trigger_time, decline_pct=trigger_pct, volume_zscore=trigger_z,
                                    avg_traded_value=liquidity.avg_traded_value if liquidity is not None else float("nan"),
                                    entry_time=entry_bar.name, entry_price=float(entry_bar["open"]),
                                    exit_time=exit_bar.name, exit_price=float(exit_bar["close"]),
                                    trigger_bar_frozen=_frozen(trigger_bar),
                                    trigger_bar_volume=float(trigger_bar["volume"]),
                                    frozen_run_after=_frozen_run_after(day_df, trigger_time, FROZEN_RUN_CHECK_BARS),
                                ))

            if baseline_mode == "pooled_window" and not window_df.empty:
                pooled_window_volumes.extend(window_df["volume"].tolist())
                prior_days_pooled += 1

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="cap universe size, for a fast test run")
    parser.add_argument("--no-liquidity-filter", action="store_true")
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

    sector_mapping = _load_sector_mapping()
    results: dict[str, tuple[float, float, int]] = {}  # label -> (sharpe, win_rate, n_days)

    # ---- REFINEMENT 1: confirmation-based entry ----
    print("=" * 70)
    print("REFINEMENT 1: confirmation-based entry (09:15-09:25 trigger, must still be <=-2.5% at 09:30)")
    print("=" * 70)
    t0 = time.monotonic()
    confirmed_events = _scan_events(
        symbol_data, sector_mapping, require_liquid,
        window_start="09:15", window_end=CONFIRM_TRIGGER_WINDOW_END,
        threshold_pct=-2.5, volume_zscore_min=1.0, baseline_mode="pooled_window",
        entry_mode="fixed_cutoff", entry_cutoff_time="09:30",
        confirm_window_end="09:30", confirm_threshold_pct=-2.5,
    )
    print(f"Found {len(confirmed_events)} confirmed (still-declining) crash events in {time.monotonic() - t0:.1f}s")
    s, w, d = _report_individual("R1: Confirmed continuation (SHORT)", confirmed_events, sign=-1.0)
    results["R1 confirmed continuation (SHORT)"] = (s, w, d)
    s, w, d = _report_individual("R1: Confirmed reversal (LONG)", confirmed_events, sign=1.0)
    results["R1 confirmed reversal (LONG)"] = (s, w, d)

    # ---- REFINEMENT 2: severity tiers (re-bucket the ORIGINAL 420 events) ----
    print("\n" + "=" * 70)
    print("REFINEMENT 2: severity tiers (re-running the ORIGINAL crash detector for exact consistency)")
    print("=" * 70)
    t0 = time.monotonic()
    original_events = _find_crash_events(symbol_data, sector_mapping, require_liquid=require_liquid)
    print(f"Found {len(original_events)} original crash events (same definition as run_crash_detection.py) "
          f"in {time.monotonic() - t0:.1f}s")
    moderate = [e for e in original_events if 2.5 <= abs(e.decline_pct) < 4.0]
    severe = [e for e in original_events if 4.0 <= abs(e.decline_pct) < 6.0]
    extreme = [e for e in original_events if abs(e.decline_pct) >= 6.0]
    print(f"Moderate (2.5-4%): {len(moderate)}   Severe (4-6%): {len(severe)}   Extreme (>=6%): {len(extreme)}")
    for tier_label, tier_events in (("Moderate (2.5-4%)", moderate), ("Severe (4-6%)", severe), ("Extreme (>=6%)", extreme)):
        s, w, d = _report_individual(f"R2: {tier_label} continuation (SHORT)", tier_events, sign=-1.0)
        results[f"R2 {tier_label} continuation"] = (s, w, d)
        s, w, d = _report_individual(f"R2: {tier_label} reversal (LONG)", tier_events, sign=1.0)
        results[f"R2 {tier_label} reversal"] = (s, w, d)

    # ---- REFINEMENT 3: melt-up (mirror image) ----
    print("\n" + "=" * 70)
    print("REFINEMENT 3: melt-up detection (>=+2.5% in first 15min, elevated volume)")
    print("=" * 70)
    t0 = time.monotonic()
    meltup_events = _scan_events(
        symbol_data, sector_mapping, require_liquid,
        window_start="09:15", window_end="09:30",
        threshold_pct=2.5, volume_zscore_min=1.0, baseline_mode="pooled_window",
        entry_mode="fixed_cutoff", entry_cutoff_time="09:30",
    )
    print(f"Found {len(meltup_events)} melt-up events in {time.monotonic() - t0:.1f}s")
    s, w, d = _report_individual("R3: Melt-up continuation (LONG)", meltup_events, sign=1.0)
    results["R3 melt-up continuation (LONG)"] = (s, w, d)
    s, w, d = _report_individual("R3: Melt-up reversal (SHORT)", meltup_events, sign=-1.0)
    results["R3 melt-up reversal (SHORT)"] = (s, w, d)

    # ---- REFINEMENT 4: crash detection all day ----
    print("\n" + "=" * 70)
    print("REFINEMENT 4: crash detection throughout the day (09:15-15:00, trailing-15min return, "
          "same-day rolling volume baseline)")
    print("=" * 70)
    t0 = time.monotonic()
    allday_events = _scan_events(
        symbol_data, sector_mapping, require_liquid,
        window_start=ALLDAY_WINDOW_START, window_end=ALLDAY_WINDOW_END,
        threshold_pct=-2.5, volume_zscore_min=1.0, baseline_mode="same_day_rolling",
        same_day_lookback_bars=ALLDAY_SAME_DAY_LOOKBACK_BARS,
        return_lookback_minutes=ALLDAY_RETURN_LOOKBACK_MINUTES,
        entry_mode="next_bar_after_trigger",
    )
    print(f"Found {len(allday_events)} all-day crash events in {time.monotonic() - t0:.1f}s "
          f"(first ~30min of trading structurally excluded -- see module docstring)")
    s, w, d = _report_individual("R4: All-day continuation (SHORT)", allday_events, sign=-1.0)
    results["R4 all-day continuation (SHORT)"] = (s, w, d)
    s, w, d = _report_individual("R4: All-day reversal (LONG)", allday_events, sign=1.0)
    results["R4 all-day reversal (LONG)"] = (s, w, d)

    morning = [e for e in allday_events if e.trigger_time.hour < 12]
    afternoon = [e for e in allday_events if e.trigger_time.hour >= 12]
    print(f"\nTiming split: late-morning (09:45-12:00): {len(morning)} events   afternoon (12:00-15:00): {len(afternoon)} events")
    for bucket_label, bucket_events in (("late-morning", morning), ("afternoon", afternoon)):
        s, w, d = _report_individual(f"R4: {bucket_label} continuation (SHORT)", bucket_events, sign=-1.0)
        results[f"R4 {bucket_label} continuation"] = (s, w, d)
        s, w, d = _report_individual(f"R4: {bucket_label} reversal (LONG)", bucket_events, sign=1.0)
        results[f"R4 {bucket_label} reversal"] = (s, w, d)

    # ---- Spot-check BEFORE verdict ----
    print("\n" + "=" * 70)
    print("SPOT-CHECK (before final verdict)")
    print("=" * 70)
    winners = {label: v for label, v in results.items() if v[0] >= MIN_SHARPE_BAR and v[1] >= MIN_WINRATE_BAR}
    variant_events = {
        "R1": confirmed_events, "R2 Moderate (2.5-4%)": moderate, "R2 Severe (4-6%)": severe,
        "R2 Extreme (>=6%)": extreme, "R3": meltup_events, "R4 late-morning": morning, "R4 afternoon": afternoon,
        "R4 all-day": allday_events,
    }
    if winners:
        best_label = max(winners, key=lambda l: winners[l][0])
        print(f"\nMost promising combination clearing both bars: {best_label} (Sharpe={winners[best_label][0]:.2f}, "
              f"win rate={winners[best_label][1]:.1%})")
        spot_events = None
        for k, evs in variant_events.items():
            if best_label.startswith(k) or k in best_label:
                spot_events = evs
                break
        if spot_events:
            _spot_check(spot_events, min(10, len(spot_events)), best_label)
            n_frozen = sum(1 for e in spot_events if e.trigger_bar_frozen)
            n_frozen_run = sum(1 for e in spot_events if e.frozen_run_after >= 2)
            n_gap = sum(1 for e in spot_events if _data_gap_flag(e) is not None)
            print(f"\nAcross this variant's {len(spot_events)} events: {n_frozen} frozen trigger bars, "
                  f"{n_frozen_run} with a frozen run after, {n_gap} likely data-gap early exits.")
    else:
        print("\nNo combination cleared both pre-registered bars across any refinement -- "
              "showing a light sanity sample from Refinement 1 anyway for transparency.")
        if confirmed_events:
            _spot_check(confirmed_events, min(8, len(confirmed_events)), "R1 confirmed events (no bar cleared)")
            n_frozen = sum(1 for e in confirmed_events if e.trigger_bar_frozen)
            n_gap = sum(1 for e in confirmed_events if _data_gap_flag(e) is not None)
            print(f"\nAcross R1's {len(confirmed_events)} events: {n_frozen} frozen trigger bars, "
                  f"{n_gap} likely data-gap early exits.")

    # ---- Honest verdict ----
    print("\n" + "=" * 70)
    print("HONEST VERDICT ACROSS ALL FOUR REFINEMENTS")
    print("=" * 70)
    print(f"{'Combination':<42s} {'Days':>6s} {'Sharpe':>8s} {'WinRate':>9s}  Verdict")
    for label, (sharpe, win, ndays) in results.items():
        print(f"{label:<42s} {ndays:>6d} {sharpe:>8.2f} {win:>8.1%}  {_verdict(sharpe, win)}")

    print(f"\nEvent counts used: R1 confirmed={len(confirmed_events)}, "
          f"R2 original(pre-bucket)={len(original_events)} (moderate={len(moderate)}, severe={len(severe)}, extreme={len(extreme)}), "
          f"R3 melt-up={len(meltup_events)}, R4 all-day={len(allday_events)} (late-morning={len(morning)}, afternoon={len(afternoon)})")


if __name__ == "__main__":
    main()
