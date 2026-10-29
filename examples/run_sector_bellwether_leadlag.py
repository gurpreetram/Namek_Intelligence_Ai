"""
Tests whether a sector's most liquid ("bellwether") stock leads its
sector peers intraday: does a significant early move in the bellwether
predict the SAME directional move in other sector constituents over the
following window, at a rate meaningfully above what an unrelated random
group of stocks shows in the same window?

PRE-REGISTERED BARS, set BEFORE running this and not adjusted after
seeing results:
  - LEAD-LAG OCCURRENCE is real, not sampling noise, only if the
    sector-follower same-direction agreement rate exceeds the random-
    stock-group baseline rate by >= 15 percentage points.
  - TRADING EDGE (following the bellwether vs random entry timing in the
    SAME stocks/days) is real only if BOTH hold on the day-level return
    series: day-level Sharpe (non-annualized) >= 0.5 AND day-level win
    rate >= 60% -- same bar examples/run_gap_fill_vs_continuation.py uses.

BELLWETHER SELECTION: per sector (config/sector_mapping.json), the
constituent with the highest average traded value (price x volume) across
its FULL cached history -- reuses detectors.liquidity_analysis's traded-
value convention, computed ONCE across the whole window (not per-day), so
the bellwether identity is stable and not re-chosen based on which stock
happens to be busiest on any one given day.

TRIGGER: scans the bellwether's own bars from market open to 10:30 AM;
at each bar, computes its trailing TRIGGER_WINDOW_MINUTES return via
detectors.sector_momentum.compute_stock_return_today(df, as_of,
lookback_minutes=TRIGGER_WINDOW_MINUTES) -- reused as-is, not
reimplemented. The FIRST bar where |that return| >= TRIGGER_MIN_MOVE_PCT
is that day's trigger (time, direction, magnitude). A sector with no such
bar that day has no trigger that day and contributes nothing to that
day's stats. No lookahead: by the time a bar prints, its own trailing
window is already fully known.

FOLLOWER CHECK: for every OTHER sector constituent, computes its own
FORWARD return over [trigger_time, trigger_time + FOLLOW_WINDOW_MINUTES]
(_forward_return_pct() below -- a new, narrowly-scoped helper; no
existing function in this project computes a forward-looking window,
by design, since everything else here is intentionally backward-looking
to avoid lookahead -- this one deliberately looks forward because it's
being computed AFTER the fact, once the whole trigger+follow window's
data already exists in the cache, for a historical/analytical test, not
a live decision). "Followed" = same sign as the trigger AND
|forward return| >= FOLLOW_MIN_MOVE_PCT (a small noise floor -- without
it, tiny moves would count as "agreement" by pure chance about half the
time). RANDOM BASELINE: for the exact same trigger day/time/direction,
an equally-sized RANDOM sample of stocks from OUTSIDE the triggering
sector is checked against the identical criteria -- this controls for
"the whole market moved that morning" confounds, since the random group
faces the exact same market conditions, just no sector relationship to
the bellwether.

TRADING TEST: enters each follower at the NEXT bar's open after the
follow window ends, direction = trigger direction, exits at EOD square-
off (15:15) -- compared against a RANDOM-ENTRY-TIME control: the SAME
follower stocks on the SAME days, same directional bet, but entered at a
RANDOM bar time drawn from the 09:15-10:30 window instead of specifically
timed to the bellwether's trigger (averaged over N_RANDOM_DRAWS draws per
event to reduce the control's own sampling noise, fixed seed for
reproducibility). This isolates whether the LEAD-LAG TIMING itself adds
value, holding the stocks/days/direction fixed, rather than "this sector
just trended that day regardless of when you got in."

Run: python -m examples.run_sector_bellwether_leadlag
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage
from detectors.liquidity_analysis import analyze_liquidity
from detectors.sector_momentum import compute_stock_return_today

TRIGGER_WINDOW_MINUTES = 20  # midpoint of the requested 15-30 min range
FOLLOW_WINDOW_MINUTES = 20
TRIGGER_MIN_MOVE_PCT = 1.0
FOLLOW_MIN_MOVE_PCT = 0.3  # noise floor -- see module docstring
TRIGGER_CUTOFF_TIME = "10:30"
EOD_SQUARE_OFF_TIME = "15:15"
MARKET_OPEN_TIME = "09:15"
MIN_SHARPE_BAR = 0.5
MIN_WINRATE_BAR = 0.60
MIN_LEADLAG_GAP_PP = 15.0  # percentage points, see module docstring
N_RANDOM_DRAWS = 10
RANDOM_SEED = 42


@dataclass
class TriggerEvent:
    sector: str
    bellwether: str
    day: date
    trigger_time: pd.Timestamp
    direction: float  # +1.0 or -1.0
    magnitude: float


def _load_sector_mapping() -> dict[str, str]:
    path = PROJECT_ROOT / "config" / "sector_mapping.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _pick_bellwethers(sector_mapping: dict[str, str], symbol_data: dict[str, pd.DataFrame]) -> dict[str, str]:
    """{sector: bellwether_symbol} -- highest avg traded value (price x
    volume) across the FULL cached history of each constituent."""
    by_sector: dict[str, list[str]] = {}
    for symbol, sector in sector_mapping.items():
        by_sector.setdefault(sector, []).append(symbol)

    bellwethers = {}
    for sector, members in by_sector.items():
        best_symbol, best_value = None, -1.0
        for symbol in members:
            df = symbol_data.get(symbol)
            if df is None or df.empty:
                continue
            traded_value = float((df["close"] * df["volume"]).mean())
            if traded_value > best_value:
                best_symbol, best_value = symbol, traded_value
        if best_symbol is not None:
            bellwethers[sector] = best_symbol
    return bellwethers


def _forward_return_pct(df: pd.DataFrame, start_ts: pd.Timestamp, forward_minutes: int) -> float | None:
    """% return from the bar AT-OR-AFTER start_ts to the bar closest to
    (but not after) start_ts + forward_minutes, same day only. Forward-
    looking by design -- see module docstring on why that's fine here
    (a historical/analytical pass over already-complete cached days, not
    a live decision)."""
    day = start_ts.date()
    day_df = df[df.index.date == day]
    from_bars = day_df[day_df.index >= start_ts]
    if from_bars.empty:
        return None
    start_price = float(from_bars["close"].iloc[0])

    window_end = start_ts + pd.Timedelta(minutes=forward_minutes)
    window_bars = day_df[(day_df.index >= start_ts) & (day_df.index <= window_end)]
    if window_bars.empty:
        return None
    end_price = float(window_bars["close"].iloc[-1])
    if start_price <= 0:
        return None
    return (end_price - start_price) / start_price * 100.0


def _find_trigger_events(sector: str, bellwether: str, df: pd.DataFrame) -> list[TriggerEvent]:
    events = []
    days = sorted(set(df.index.date))
    for day in days:
        day_df = df[df.index.date == day]
        cutoff = pd.Timestamp.combine(day, pd.Timestamp(TRIGGER_CUTOFF_TIME).time()).tz_localize(day_df.index.tz)
        candidates = day_df[day_df.index <= cutoff]
        if candidates.empty:
            continue
        for as_of in candidates.index:
            ret = compute_stock_return_today(df, as_of, lookback_minutes=TRIGGER_WINDOW_MINUTES)
            if ret is not None and abs(ret) >= TRIGGER_MIN_MOVE_PCT:
                events.append(TriggerEvent(
                    sector=sector, bellwether=bellwether, day=day, trigger_time=as_of,
                    direction=1.0 if ret > 0 else -1.0, magnitude=abs(ret),
                ))
                break  # first qualifying bar only, per day
    return events


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    sector_mapping = _load_sector_mapping()
    sectors = sorted(set(sector_mapping.values()))
    print(f"Sectors mapped: {len(sectors)}  |  Symbols mapped: {len(sector_mapping)}")

    mapped_symbols = sorted(set(sector_mapping.keys()))
    print(f"Loading {len(mapped_symbols)} sector-mapped symbols into memory...")
    t0 = time.monotonic()
    symbol_data: dict[str, pd.DataFrame] = {}
    for s in mapped_symbols:
        df = storage.load(s)
        if not df.empty:
            symbol_data[s] = df
    print(f"Loaded {len(symbol_data)}/{len(mapped_symbols)} with usable data in {time.monotonic() - t0:.1f}s\n")

    bellwethers = _pick_bellwethers(sector_mapping, symbol_data)
    print("Bellwethers selected (highest avg traded value per sector):")
    for sector, bw in sorted(bellwethers.items()):
        print(f"  {sector:22s} -> {bw}")
    print()

    print("Scanning for bellwether trigger events (>=1% move in a 20-min window, before 10:30)...")
    t0 = time.monotonic()
    all_events: list[TriggerEvent] = []
    for sector, bw in bellwethers.items():
        all_events.extend(_find_trigger_events(sector, bw, symbol_data[bw]))
    print(f"Found {len(all_events)} trigger events across {len(bellwethers)} sectors in {time.monotonic() - t0:.1f}s\n")

    by_sector: dict[str, list[str]] = {}
    for symbol, sector in sector_mapping.items():
        by_sector.setdefault(sector, []).append(symbol)
    all_mapped_symbols = list(symbol_data.keys())

    # ---- OCCURRENCE TEST: sector-follower vs random-group agreement rate ----
    print("=" * 70)
    print("LEAD-LAG OCCURRENCE TEST")
    print("=" * 70)
    sector_agree, sector_total = 0, 0
    random_agree, random_total = 0, 0
    lags_and_magnitudes: list[tuple[float, float]] = []

    trading_records: list[dict] = []  # for the trading test below

    for ev in all_events:
        followers = [s for s in by_sector.get(ev.sector, []) if s != ev.bellwether and s in symbol_data]
        if not followers:
            continue
        outside_pool = [s for s in all_mapped_symbols if sector_mapping.get(s) != ev.sector]
        if len(outside_pool) < len(followers):
            continue
        random_group = list(rng.choice(outside_pool, size=len(followers), replace=False))

        for follower in followers:
            fret = _forward_return_pct(symbol_data[follower], ev.trigger_time, FOLLOW_WINDOW_MINUTES)
            if fret is None:
                continue
            sector_total += 1
            followed = (np.sign(fret) == ev.direction) and (abs(fret) >= FOLLOW_MIN_MOVE_PCT)
            if followed:
                sector_agree += 1
                lags_and_magnitudes.append((FOLLOW_WINDOW_MINUTES, fret))
            trading_records.append({
                "follower": follower, "day": ev.day, "trigger_time": ev.trigger_time,
                "direction": ev.direction, "sector": ev.sector,
            })

        for rsym in random_group:
            rret = _forward_return_pct(symbol_data[rsym], ev.trigger_time, FOLLOW_WINDOW_MINUTES)
            if rret is None:
                continue
            random_total += 1
            if (np.sign(rret) == ev.direction) and (abs(rret) >= FOLLOW_MIN_MOVE_PCT):
                random_agree += 1

    sector_rate = sector_agree / sector_total if sector_total else 0.0
    random_rate = random_agree / random_total if random_total else 0.0
    gap_pp = (sector_rate - random_rate) * 100.0
    print(f"Trigger events: {len(all_events)}  |  Sector-follower observations: {sector_total}  |  Random-control observations: {random_total}")
    print(f"Sector-follower same-direction agreement rate: {sector_rate:.1%}")
    print(f"Random-group same-direction agreement rate:    {random_rate:.1%}")
    print(f"Gap: {gap_pp:+.1f} percentage points")
    occurrence_verdict = "REAL (gap >= 15pp)" if gap_pp >= MIN_LEADLAG_GAP_PP else "NOT SUPPORTED (gap < 15pp pre-registered bar)"
    print(f"PRE-REGISTERED BAR (gap >= 15pp): {occurrence_verdict}")
    if lags_and_magnitudes:
        mags = [m for _, m in lags_and_magnitudes]
        print(f"Among agreeing followers: avg forward-window move magnitude {np.mean(np.abs(mags)):+.3f}%")

    # ---- TRADING TEST: post-trigger entry vs random-entry-time control ----
    print()
    print("=" * 70)
    print("TRADING TEST: following the bellwether vs random entry timing")
    print("=" * 70)
    if not trading_records:
        print("No trading records -- skipping.")
        return

    real_day_returns: dict[date, list[float]] = {}
    random_day_returns: dict[date, list[float]] = {}

    for rec in trading_records:
        df = symbol_data[rec["follower"]]
        day = rec["day"]
        direction = rec["direction"]
        day_df = df[df.index.date == day]
        eod_ts = pd.Timestamp.combine(day, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(day_df.index.tz)
        exit_bars = day_df[day_df.index <= eod_ts]
        if exit_bars.empty:
            continue
        exit_price = float(exit_bars["close"].iloc[-1])

        # Real: enter next bar after the follow window ends.
        follow_end = rec["trigger_time"] + pd.Timedelta(minutes=FOLLOW_WINDOW_MINUTES)
        entry_candidates = day_df[day_df.index > follow_end]
        if entry_candidates.empty:
            continue
        entry_bar = entry_candidates.iloc[0]
        entry_price = float(entry_bar["open"])
        if entry_bar.name >= exit_bars.index[-1]:
            continue
        real_ret = direction * (exit_price - entry_price) / entry_price * 100.0
        real_day_returns.setdefault(day, []).append(real_ret)

        # Random-entry-time control: N_RANDOM_DRAWS random bars in the
        # 09:15-10:30 window, same direction, same exit -- averaged to
        # reduce the control's own sampling noise.
        open_ts = pd.Timestamp.combine(day, pd.Timestamp(MARKET_OPEN_TIME).time()).tz_localize(day_df.index.tz)
        cutoff_ts = pd.Timestamp.combine(day, pd.Timestamp(TRIGGER_CUTOFF_TIME).time()).tz_localize(day_df.index.tz)
        window_bars = day_df[(day_df.index >= open_ts) & (day_df.index <= cutoff_ts) & (day_df.index < exit_bars.index[-1])]
        if window_bars.empty:
            continue
        draws = rng.choice(len(window_bars), size=min(N_RANDOM_DRAWS, len(window_bars)), replace=False)
        draw_rets = []
        for idx in draws:
            rand_entry_price = float(window_bars.iloc[idx]["open"])
            draw_rets.append(direction * (exit_price - rand_entry_price) / rand_entry_price * 100.0)
        random_day_returns.setdefault(day, []).append(float(np.mean(draw_rets)))

    def _day_stats(by_day: dict[date, list[float]]) -> tuple[int, float, float, float, float]:
        arr = np.array([float(np.mean(v)) for v in by_day.values()])
        if len(arr) == 0:
            return 0, 0.0, 0.0, 0.0, 0.0
        win_rate = float((arr > 0).mean())
        mean, std = float(arr.mean()), float(arr.std())
        sharpe = mean / std if std > 0 else 0.0
        return len(arr), win_rate, mean, std, sharpe

    n_days, win_rate, mean, std, sharpe = _day_stats(real_day_returns)
    rn_days, rwin_rate, rmean, rstd, rsharpe = _day_stats(random_day_returns)

    print(f"REAL (post-trigger entry): days={n_days}  win_rate={win_rate:.1%}  avg_daily_return={mean:+.3f}%  std={std:.3f}%  Sharpe={sharpe:.2f}")
    print(f"RANDOM-TIME control:       days={rn_days}  win_rate={rwin_rate:.1%}  avg_daily_return={rmean:+.3f}%  std={rstd:.3f}%  Sharpe={rsharpe:.2f}")
    trading_verdict = (
        "REAL EDGE (meets both pre-registered bars)"
        if sharpe >= MIN_SHARPE_BAR and win_rate >= MIN_WINRATE_BAR
        else "suggestive only (meets one bar)"
        if sharpe >= MIN_SHARPE_BAR or win_rate >= MIN_WINRATE_BAR
        else "NOISE (meets neither pre-registered bar)"
    )
    print(f"PRE-REGISTERED BAR (Sharpe>=0.5 AND win rate>=60%, REAL variant): {trading_verdict}")
    print(f"REAL vs RANDOM-TIME edge (does the specific lead-lag timing add anything over the same directional bet at a random time): "
          f"Sharpe {sharpe:.2f} vs {rsharpe:.2f}, avg return {mean:+.3f}% vs {rmean:+.3f}%")

    print()
    print("=" * 70)
    print(f"SAMPLE-SIZE CAVEAT: results above come from Upstox's ~29-33 day rolling "
          f"1-minute cache and {len(all_events)} bellwether trigger events across {len(bellwethers)} "
          f"sectors. This is a SMALL sample -- treat all findings as preliminary.")
    print("=" * 70)


if __name__ == "__main__":
    main()
