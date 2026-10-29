"""
Tests the "Support/Resistance Compression Breakout" hypothesis: a stock
that repeatedly touches a level without breaking it, with each touch
overshooting LESS than the last (a coiling/compression signature), then
finally closes beyond that level on elevated volume -- does trading WITH
that breakout (long a resistance break, short a support breakdown) have
a real edge?

PRE-REGISTERED BARS (same as every other hypothesis test in this
project): a (touch-tier, direction) combination is a real edge only if
BOTH hold on its day-level return series: day-level Sharpe (non-
annualized) >= 0.5 AND day-level win rate >= 60%. Set BEFORE running
this, applied identically to all 4 combinations (3-touch/resistance,
3-touch/support, 5-touch/resistance, 5-touch/support).

LEVEL DETECTION (STEP 1) -- reuses detectors/support_resistance.py's
find_levels()/average_true_range() AS-IS, not reimplemented. Levels are
built from 15-MINUTE bars (data/resample.py), not 1-minute -- running
find_levels() on 1m data produces noisy, low-significance levels per
that module's own docstring. For each cached symbol and each candidate
day D (needs >= LOOKBACK_DAYS prior cached trading days), find_levels()
runs on ONLY the LOOKBACK_DAYS trading days strictly BEFORE D (no
lookahead: a level must already be established before D's session
opens), with cluster_tolerance_atr_mult=0.3 matching the requested
0.3xATR touch tolerance.

find_levels() clusters swing highs AND swing lows together into one
touch-counted pool -- it does not itself say whether a level is acting
as resistance or support, or whether any touch already closed beyond it.
Both are added here, on top of the reused clustering, not inside it:
  - SIDE CLASSIFICATION + "never closed beyond": for a candidate level,
    every one of its clustered touches' 15m CLOSE must sit on the SAME
    side (all <= level.price -> resistance candidate; all >= level.price
    -> support candidate). A level with even one touch that already
    closed through it before the eventual breakout isn't "touched
    without breaking" -- literally excluded, not down-weighted.
  - COMPRESSION ("narrowing over time"): overshoot per touch = the
    touch's own swing extreme (15m high for a resistance candidate, 15m
    low for a support candidate) minus level.price, in absolute value.
    Touches are already time-ordered by find_levels(); split into first-
    half / second-half chronologically and require
    mean(|overshoot| second half) < mean(|overshoot| first half) --
    later touches coiling tighter than earlier ones.
LOOKBACK_DAYS=7: the midpoint of the requested 5-10 trading day range,
not separately tested against 5 or 10 -- only touch-count (3 vs 5) is
the requested two-way comparison here.

BREAKOUT TRIGGER: scans day D's 1-MINUTE bars (fine-grained, for a real
entry price/time) for the FIRST bar whose CLOSE crosses the level in the
classified direction, with volume_zscore() (detectors/breakout_validator.py,
reused as-is) >= 1.0. Baseline is a SAME-DAY rolling window (preceding 30
1-min bars, current day only) -- the SAME choice and the SAME reasoning
as examples/run_crash_detection_refinements.py's Refinement 4 (a breakout
can happen at any time of day here, not just the open, so a prior-day
time-of-day-matched pool doesn't apply; a same-day rolling baseline
avoids the well-documented opening-volume bias, at the cost of
structurally excluding the first ~30 minutes of trading from ever
qualifying as a breakout bar -- same deliberate, documented trade-off).

ENTRY: next 1-min bar's OPEN strictly after the breakout bar (no-
lookahead, matching backtest/engine.py's and every other script's
convention). EXIT: EOD square-off at or before 15:15, via
detectors/cross_sectional_ranking.py's bar_at_or_before() (reused as-is).

LIQUIDITY FILTER APPLIED FROM THE START: detectors/liquidity_analysis.py's
analyze_liquidity() on day D's own most recent PRIOR cached day, gating
whether day D is even considered AT ALL -- before any level/compression
work happens for that day, per the lesson learned from the gap-fill
research pass (examples/run_gap_fill_vs_continuation.py's docstring).

TWO DIRECTIONS (STEP 2), going WITH the breakout only (this hypothesis,
unlike the crash-detection tests, doesn't also test fading the move):
  - resistance breakout -> LONG
  - support breakdown -> SHORT

Reuses run_crash_detection.py's _day_stats/_verdict/_report_individual
(day-level Sharpe/win-rate reporting) and
run_crash_detection_refinements.py's _report_top_n_concentration
(the concentration test), rather than reimplementing either -- both are
already generic over any event type exposing .day/.entry_price/.exit_price.

Run: python -m examples.run_sr_compression_breakout
    python -m examples.run_sr_compression_breakout --limit 300   # fast test
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
from data.resample import resample
from detectors.breakout_validator import volume_zscore
from detectors.cross_sectional_ranking import bar_at_or_before
from detectors.liquidity_analysis import analyze_liquidity
from detectors.scanner import MIN_STOCK_PRICE
from detectors.support_resistance import Level, average_true_range, find_levels
from examples.run_crash_detection import MIN_SHARPE_BAR, MIN_WINRATE_BAR, COST_PCT_PER_ROUND_TRIP, \
    _day_stats, _verdict, _report_individual, _report_top_n_concentration

LOOKBACK_DAYS = 7  # midpoint of the requested 5-10 trading day range
RESAMPLE_MINUTES = 15
CLUSTER_TOLERANCE_ATR_MULT = 0.3  # the requested 0.3xATR touch tolerance
BASE_MIN_TOUCHES = 3  # superset passed to find_levels(); >=5 tier is a subset of this, not a separate scan
HIGH_MIN_TOUCHES = 5
BREAKOUT_VOLUME_ZSCORE_MIN = 1.0
SAME_DAY_LOOKBACK_BARS = 30
EOD_SQUARE_OFF_TIME = "15:15"
DATA_GAP_EARLY_EXIT_MINUTES = 30
FROZEN_RUN_CHECK_BARS = 10
CIRCUIT_BAND_PCTS = (2.0, 5.0, 10.0, 20.0)
CIRCUIT_BAND_TOLERANCE = 0.3
# Once a level has genuinely broken out, it isn't a valid "compressed, not-yet-broken"
# level anymore -- but find_levels() re-derives levels fresh from a ROLLING lookback
# window every day, so the SAME level can keep getting rediscovered (and re-"breaking
# out") on every subsequent day for as long as price simply remains beyond it. Caught
# via this script's own spot-check (AAVAS's 1494.10 level firing on both 2026-06-30
# AND 2026-07-01 with near-identical touch/overshoot stats) -- not a hypothetical
# concern. A level within this tolerance of one already credited with a breakout for
# this symbol is skipped on every later day, so each real compression episode is
# counted once, not once per day it happens to still be above/below its old level.
TRIGGERED_LEVEL_DEDUPE_TOLERANCE_PCT = 1.0


@dataclass
class CompressionBreakoutEvent:
    symbol: str
    day: date
    direction: str  # "resistance_breakout" (LONG) | "support_breakdown" (SHORT)
    level_price: float
    touch_count: int
    first_touch: pd.Timestamp
    last_touch: pd.Timestamp
    overshoot_first_half_avg: float
    overshoot_second_half_avg: float
    breakout_time: pd.Timestamp
    breakout_volume_zscore: float
    avg_traded_value: float
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    breakout_bar_frozen: bool
    frozen_run_after: int


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


def _classify_and_validate_level(level: Level, lookback_15m: pd.DataFrame) -> dict | None:
    """
    Reuses the already-clustered Level as-is (touch_count, touch_timestamps
    from find_levels()) and adds the two NEW checks this hypothesis needs
    on top: side classification + "never closed beyond" + narrowing. See
    module docstring for the exact definitions. Returns None if the level
    doesn't qualify as a clean, one-sided, compressing candidate.
    """
    closes = [float(lookback_15m.loc[ts, "close"]) for ts in level.touch_timestamps]
    if all(c <= level.price for c in closes):
        side = "resistance"
        extremes = [float(lookback_15m.loc[ts, "high"]) for ts in level.touch_timestamps]
    elif all(c >= level.price for c in closes):
        side = "support"
        extremes = [float(lookback_15m.loc[ts, "low"]) for ts in level.touch_timestamps]
    else:
        return None  # mixed -- at least one touch already closed through it, not a clean compression

    overshoots = [abs(e - level.price) for e in extremes]
    n = len(overshoots)
    half = max(1, n // 2)
    first_half_avg = float(np.mean(overshoots[:half]))
    second_half_avg = float(np.mean(overshoots[half:])) if n > half else first_half_avg
    if not (second_half_avg < first_half_avg):
        return None  # no narrowing -- touches aren't coiling tighter over time

    return {
        "side": side, "n": n, "first_half_avg": first_half_avg, "second_half_avg": second_half_avg,
        "first_touch": level.touch_timestamps[0], "last_touch": level.touch_timestamps[-1],
    }


def _find_events(symbol_data: dict[str, pd.DataFrame], require_liquid: bool = True) -> list[CompressionBreakoutEvent]:
    events: list[CompressionBreakoutEvent] = []
    for symbol, df in symbol_data.items():
        days = sorted(set(df.index.date))
        if len(days) <= LOOKBACK_DAYS:
            continue
        df_15m = resample(df, RESAMPLE_MINUTES)
        if df_15m.empty:
            continue

        triggered_level_prices: list[float] = []

        for i in range(LOOKBACK_DAYS, len(days)):
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

            lookback_days_set = set(days[i - LOOKBACK_DAYS:i])
            lookback_15m = df_15m[pd.Series(df_15m.index.date, index=df_15m.index).isin(lookback_days_set)]
            if lookback_15m.empty:
                continue

            levels = find_levels(lookback_15m, min_touches=BASE_MIN_TOUCHES,
                                  cluster_tolerance_atr_mult=CLUSTER_TOLERANCE_ATR_MULT)
            if not levels:
                continue

            day_df = df[df.index.date == day]
            if day_df.empty:
                continue

            for level in levels:
                if any(abs(level.price - t) / t * 100.0 <= TRIGGERED_LEVEL_DEDUPE_TOLERANCE_PCT
                       for t in triggered_level_prices):
                    continue  # already credited with a breakout for this symbol -- see module docstring

                info = _classify_and_validate_level(level, lookback_15m)
                if info is None:
                    continue

                roll_mean = day_df["volume"].shift(1).rolling(SAME_DAY_LOOKBACK_BARS).mean()
                roll_std = day_df["volume"].shift(1).rolling(SAME_DAY_LOOKBACK_BARS).std()
                vol_z = (day_df["volume"] - roll_mean) / roll_std

                if info["side"] == "resistance":
                    trigger_mask = (day_df["close"] > level.price) & (vol_z >= BREAKOUT_VOLUME_ZSCORE_MIN)
                    direction = "resistance_breakout"
                else:
                    trigger_mask = (day_df["close"] < level.price) & (vol_z >= BREAKOUT_VOLUME_ZSCORE_MIN)
                    direction = "support_breakdown"
                trigger_mask = trigger_mask.fillna(False)
                if not trigger_mask.any():
                    continue

                breakout_time = trigger_mask[trigger_mask].index[0]
                entry_candidates = day_df[day_df.index > breakout_time]
                if entry_candidates.empty:
                    continue
                entry_bar = entry_candidates.iloc[0]

                eod_ts = pd.Timestamp.combine(day, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(day_df.index.tz)
                exit_bar = bar_at_or_before(df, eod_ts)
                if exit_bar is None or exit_bar.name <= entry_bar.name:
                    continue

                breakout_bar = day_df.loc[breakout_time]
                triggered_level_prices.append(level.price)
                events.append(CompressionBreakoutEvent(
                    symbol=symbol, day=day, direction=direction, level_price=level.price,
                    touch_count=info["n"], first_touch=info["first_touch"], last_touch=info["last_touch"],
                    overshoot_first_half_avg=info["first_half_avg"], overshoot_second_half_avg=info["second_half_avg"],
                    breakout_time=breakout_time, breakout_volume_zscore=float(vol_z.loc[breakout_time]),
                    avg_traded_value=liquidity.avg_traded_value if liquidity is not None else float("nan"),
                    entry_time=entry_bar.name, entry_price=float(entry_bar["open"]),
                    exit_time=exit_bar.name, exit_price=float(exit_bar["close"]),
                    breakout_bar_frozen=_frozen(breakout_bar),
                    frozen_run_after=_frozen_run_after(day_df, breakout_time, FROZEN_RUN_CHECK_BARS),
                ))

    return events


def _data_gap_flag(e: CompressionBreakoutEvent) -> str | None:
    eod_minutes = 15 * 60 + 15
    exit_minutes = e.exit_time.hour * 60 + e.exit_time.minute
    if eod_minutes - exit_minutes >= DATA_GAP_EARLY_EXIT_MINUTES:
        return (f"POSSIBLE DATA GAP: exit found at {e.exit_time.strftime('%H:%M')}, "
                f"{eod_minutes - exit_minutes} min before the 15:15 EOD target -- treat with caution")
    return None


def _circuit_flag(e: CompressionBreakoutEvent) -> str:
    break_pct = abs(e.exit_price - e.level_price) / e.level_price * 100.0
    near_band = any(abs(break_pct - band) <= CIRCUIT_BAND_TOLERANCE for band in CIRCUIT_BAND_PCTS)
    if e.breakout_bar_frozen and e.frozen_run_after >= 2:
        return f"LIKELY CIRCUIT LOCK (frozen breakout bar + {e.frozen_run_after} more frozen bars immediately after)"
    if e.breakout_bar_frozen:
        return "breakout bar frozen (O=H=L=C) -- worth checking, not necessarily a circuit hit"
    if near_band:
        return "move size near a standard circuit band, but breakout bar NOT frozen -- likely genuine"
    return "no circuit-lock indicators"


def _spot_check(events: list[CompressionBreakoutEvent], n: int, label: str) -> None:
    print(f"\n--- SPOT-CHECK ({label}): {min(n, len(events))} real events ---")
    for e in sorted(events, key=lambda e: e.touch_count, reverse=True)[:n]:
        print(f"  {e.symbol:14s} {e.day}  {e.direction:18s} level={e.level_price:.2f}  touches={e.touch_count}  "
              f"overshoot {e.overshoot_first_half_avg:.3f}->{e.overshoot_second_half_avg:.3f}  "
              f"liquidity=Rs.{e.avg_traded_value:,.0f}/min")
        print(f"    breakout={e.breakout_time.strftime('%H:%M')} (vol_z={e.breakout_volume_zscore:+.2f})  "
              f"entry={e.entry_price:.2f} ({e.entry_time.strftime('%H:%M')})  "
              f"exit={e.exit_price:.2f} ({e.exit_time.strftime('%H:%M')})")
        print(f"    {_circuit_flag(e)}")
        gap_flag = _data_gap_flag(e)
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
    print(f"STEP 1: compression + breakout detection (>=3 touches, 0.3xATR tolerance, "
          f"{LOOKBACK_DAYS}-day lookback), liquidity filter {'ON' if require_liquid else 'OFF'} FROM THE START")
    print("=" * 70)
    t0 = time.monotonic()
    all_events = _find_events(symbol_data, require_liquid=require_liquid)
    print(f"Found {len(all_events)} qualifying compression-breakout events in {time.monotonic() - t0:.1f}s")
    all_days = sorted({e.day for e in all_events})
    print(f"Distinct trading days with >=1 event: {len(all_days)}")
    print("=" * 70)
    print("SAMPLE-SIZE CAVEAT: results below come from Upstox's ~29-33 day rolling "
          "1-minute cache. This is a SMALL sample -- treat all findings as preliminary.")
    print("=" * 70)

    if not all_events:
        print("\nNo compression-breakout events found in cached history -- nothing further to test.")
        return

    tier3 = all_events
    tier5 = [e for e in all_events if e.touch_count >= HIGH_MIN_TOUCHES]
    resistance3 = [e for e in tier3 if e.direction == "resistance_breakout"]
    support3 = [e for e in tier3 if e.direction == "support_breakdown"]
    resistance5 = [e for e in tier5 if e.direction == "resistance_breakout"]
    support5 = [e for e in tier5 if e.direction == "support_breakdown"]
    print(f"\n3-touch tier: {len(tier3)} events ({len(resistance3)} resistance, {len(support3)} support)")
    print(f"5-touch tier: {len(tier5)} events ({len(resistance5)} resistance, {len(support5)} support)")

    print("\n" + "=" * 70)
    print("STEP 3: backtest all 4 combinations")
    print("=" * 70)
    results: dict[str, tuple[float, float, int]] = {}
    combos = [
        ("3-touch resistance breakout (LONG)", resistance3, 1.0),
        ("3-touch support breakdown (SHORT)", support3, -1.0),
        ("5-touch resistance breakout (LONG)", resistance5, 1.0),
        ("5-touch support breakdown (SHORT)", support5, -1.0),
    ]
    for label, evs, sign in combos:
        s, w, d = _report_individual(label, evs, sign=sign)
        results[label] = (s, w, d)

    print("\n" + "=" * 70)
    print("STEP 4: concentration test (only combos clearing BOTH pre-registered bars)")
    print("=" * 70)
    winners = [label for label, (s, w, _) in results.items() if s >= MIN_SHARPE_BAR and w >= MIN_WINRATE_BAR]
    combo_by_label = {label: (evs, sign) for label, evs, sign in combos}
    if not winners:
        print("\nNo combination cleared both pre-registered bars -- skipping concentration test.")
    else:
        print(f"\nCombination(s) clearing both bars: {', '.join(winners)}")
        for label in winners:
            evs, sign = combo_by_label[label]
            labeled = [(e.day, e.touch_count, sign * (e.exit_price - e.entry_price) / e.entry_price * 100.0) for e in evs]
            for n in (1, 3, 5):
                _report_top_n_concentration(labeled, n, args.cost_pct, f"{label}, ranked by touch count")

    print("\n" + "=" * 70)
    print("STEP 5: spot-check (BEFORE final verdict)")
    print("=" * 70)
    if winners:
        best_label = max(winners, key=lambda l: results[l][0])
        spot_events = combo_by_label[best_label][0]
    else:
        best_label = max(results, key=lambda l: results[l][0])
        spot_events = combo_by_label[best_label][0]
    _spot_check(spot_events, min(10, len(spot_events)), best_label)
    n_frozen = sum(1 for e in spot_events if e.breakout_bar_frozen)
    n_frozen_run = sum(1 for e in spot_events if e.frozen_run_after >= 2)
    n_gap = sum(1 for e in spot_events if _data_gap_flag(e) is not None)
    print(f"\nAcross this variant's {len(spot_events)} events: {n_frozen} frozen breakout bars, "
          f"{n_frozen_run} with a frozen run after, {n_gap} likely data-gap early exits.")

    print("\n" + "=" * 70)
    print("STEP 6: HONEST VERDICT (pre-registered bars: Sharpe>=0.5 AND win rate>=60%)")
    print("=" * 70)
    print(f"{'Combination':<38s} {'Days':>6s} {'Sharpe':>8s} {'WinRate':>9s}  Verdict")
    for label, (s, w, d) in results.items():
        print(f"{label:<38s} {d:>6d} {s:>8.2f} {w:>8.1%}  {_verdict(s, w)}")

    print(f"\nEvent counts used: 3-touch total={len(tier3)} (resistance={len(resistance3)}, support={len(support3)}), "
          f"5-touch total={len(tier5)} (resistance={len(resistance5)}, support={len(support5)}), "
          f"across {len(all_days)} distinct trading days.")


if __name__ == "__main__":
    main()
