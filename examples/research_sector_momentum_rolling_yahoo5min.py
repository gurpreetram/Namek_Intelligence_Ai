"""
RESEARCH SCRIPT -- NOT a live feature. Re-run of
examples/research_sector_momentum_rolling.py's exact methodology, using
data/storage_yahoo_5min.py (a free, unauthenticated, RESEARCH-ONLY
supplementary source giving ~60 days of 5-minute NSE data -- see that
module's docstring for the verification behind it) instead of storage.py's
~21-day 1-minute Upstox cache. This is the ONLY change from the original
script -- same checkpoint grid, same (lookback_min, forward_min)
combinations, same top-2-up/bottom-2-down sector selection, same random
baseline, same pre-registered bar (edge >= 0.15pp, day-win-rate >= 55%),
kept byte-identical rather than re-tuned so this result is directly
comparable to the original 21-day one, not a moved goalpost.

WHY NO METHODOLOGY CHANGE WAS NEEDED (checked directly, not assumed):
_window_return() below computes each window via
day_df.between_time(start, end_inclusive) -- filtered by ACTUAL CLOCK
TIME, not by counting N bars. Every checkpoint/lookback/forward boundary
in COMBINATIONS is already defined in real minutes (15, 20, 30), all of
which are multiples of 5, so every window boundary lands exactly on a
real 5-minute bar in this dataset -- no rounding, no interpolation, no
change in what any window "means" in wall-clock terms. This is
DIFFERENT from detectors/sector_trend.py's compute_sector_trend(), which
uses a BAR-COUNT lookback_bars parameter (20 bars = 100 real minutes on
5-min data vs 20 minutes on 1-min data) -- that module is NOT used here,
so its bar-count issue doesn't apply to this script.

HONEST CAVEAT that IS real despite the above: a "15-minute lookback" on
this dataset is computed from only 3 underlying 5-minute bars (vs 15
bars on the original 1-minute data) -- same real-time window, but a
coarser, lower-resolution price series feeding it. That's a genuine
difference in what's being measured, even though the window boundaries
themselves are unchanged -- see the final report for how this is
weighed against the sample-size gain.

Run: python -m examples.research_sector_momentum_rolling_yahoo5min
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

from data import storage_yahoo_5min

MARKET_OPEN_TIME = time(9, 15)
EOD_SQUARE_OFF_TIME = time(15, 15)
CHECKPOINT_INTERVAL_MIN = 15
CHECKPOINT_START = time(9, 30)
CHECKPOINT_END = time(15, 0)

# Identical to the original script -- see module docstring for why this
# must not be re-tuned for a like-for-like comparison.
COMBINATIONS = [(15, 15), (15, 30), (20, 20), (30, 15)]

MIN_SECTOR_MEMBERS = 3
MIN_VALID_SECTORS = 4
RANDOM_SEED = 42

MIN_EDGE_PP = 0.15
MIN_WIN_RATE = 0.55


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _minutes_to_time(m: int) -> time:
    return time(m // 60, m % 60)


def _generate_checkpoints() -> list[time]:
    checkpoints = []
    m = _time_to_minutes(CHECKPOINT_START)
    end_m = _time_to_minutes(CHECKPOINT_END)
    while m <= end_m:
        checkpoints.append(_minutes_to_time(m))
        m += CHECKPOINT_INTERVAL_MIN
    return checkpoints


def _window_return(day_df: pd.DataFrame, start: time, end_inclusive: time) -> float | None:
    window = day_df.between_time(start, end_inclusive, inclusive="both")
    if window.empty:
        return None
    open_price = float(window["open"].iloc[0])
    close_price = float(window["close"].iloc[-1])
    if open_price <= 0:
        return None
    return (close_price - open_price) / open_price * 100.0


@dataclass
class CheckpointResult:
    sector_directional_returns: list[float]
    random_directional_returns: list[float]


def main() -> None:
    mapping = json.load(open("config/sector_mapping.json"))
    print(f"Loading {len(mapping)} sector-mapped symbols from storage/yahoo_5min/...")
    symbol_data = {s: storage_yahoo_5min.load(s) for s in mapping}
    symbol_data = {s: df for s, df in symbol_data.items() if not df.empty}
    print(f"  {len(symbol_data)} have cached 5-min data\n")

    by_sector: dict[str, list[str]] = {}
    for symbol, sector in mapping.items():
        if symbol in symbol_data:
            by_sector.setdefault(sector, []).append(symbol)

    date_counts: dict[date, int] = {}
    for df in symbol_data.values():
        for d in set(df.index.date):
            date_counts[d] = date_counts.get(d, 0) + 1
    full_universe = len(symbol_data)
    trading_days = sorted(d for d, c in date_counts.items() if c >= 0.9 * full_universe)
    excluded_days = sorted(d for d, c in date_counts.items() if c < 0.9 * full_universe)
    print(f"Trading days with full coverage: {len(trading_days)}  "
          f"(original 1-min Upstox re-test had 21)")
    if excluded_days:
        print(f"Excluded as partial/incomplete: {[(str(d), date_counts[d]) for d in excluded_days]}\n")

    all_checkpoints = _generate_checkpoints()
    print(f"Checkpoint grid: {len(all_checkpoints)} checkpoints/day, every {CHECKPOINT_INTERVAL_MIN} min, "
          f"{CHECKPOINT_START}-{CHECKPOINT_END}\n")

    rng = np.random.default_rng(RANDOM_SEED)

    day_bars_cache: dict[date, dict[str, pd.DataFrame]] = {}
    for day in trading_days:
        day_bars: dict[str, pd.DataFrame] = {}
        for symbol, df in symbol_data.items():
            day_df = df[df.index.date == day]
            if not day_df.empty:
                day_bars[symbol] = day_df
        day_bars_cache[day] = day_bars

    print("=" * 78)
    print("SECTOR-MOMENTUM ROLLING PREDICTIVE POWER -- YAHOO 5-MIN RE-TEST (not a live feature)")
    print("=" * 78)

    verdicts = []
    for lookback_min, forward_min in COMBINATIONS:
        valid_checkpoints = [
            cp for cp in all_checkpoints
            if _time_to_minutes(cp) - lookback_min >= _time_to_minutes(MARKET_OPEN_TIME)
            and _time_to_minutes(cp) + forward_min <= _time_to_minutes(EOD_SQUARE_OFF_TIME)
        ]

        day_sector_avgs: list[float] = []
        day_random_avgs: list[float] = []
        all_pooled_sector: list[float] = []
        all_pooled_random: list[float] = []
        total_checkpoints_used = 0

        for day in trading_days:
            day_bars = day_bars_cache[day]
            day_sector_returns: list[float] = []
            day_random_returns: list[float] = []

            for cp in valid_checkpoints:
                lookback_start = _minutes_to_time(_time_to_minutes(cp) - lookback_min)
                lookback_end = _minutes_to_time(_time_to_minutes(cp) - 1)
                forward_start = cp
                forward_end = _minutes_to_time(_time_to_minutes(cp) + forward_min - 1)

                sector_trailing_return: dict[str, float] = {}
                for sector, members in by_sector.items():
                    member_returns = []
                    for symbol in members:
                        df = day_bars.get(symbol)
                        if df is None:
                            continue
                        r = _window_return(df, lookback_start, lookback_end)
                        if r is not None:
                            member_returns.append(r)
                    if len(member_returns) >= MIN_SECTOR_MEMBERS:
                        sector_trailing_return[sector] = sum(member_returns) / len(member_returns)

                if len(sector_trailing_return) < MIN_VALID_SECTORS:
                    continue

                ranked = sorted(sector_trailing_return.items(), key=lambda kv: kv[1], reverse=True)
                selected_sectors = {s: r for s, r in ranked[:2] + ranked[-2:]}

                cp_sector_returns: list[float] = []
                for sector, trail_ret in selected_sectors.items():
                    sign = 1.0 if trail_ret >= 0 else -1.0
                    for symbol in by_sector[sector]:
                        df = day_bars.get(symbol)
                        if df is None:
                            continue
                        fwd_ret = _window_return(df, forward_start, forward_end)
                        if fwd_ret is not None:
                            cp_sector_returns.append(sign * fwd_ret)

                if not cp_sector_returns:
                    continue

                candidate_pool = [s for s in day_bars if s in mapping]
                n = len(cp_sector_returns)
                if len(candidate_pool) < n:
                    continue
                random_symbols = rng.choice(candidate_pool, size=n, replace=False)
                random_directions = rng.choice([1.0, -1.0], size=n)
                cp_random_returns: list[float] = []
                for symbol, rand_sign in zip(random_symbols, random_directions):
                    df = day_bars[symbol]
                    fwd_ret = _window_return(df, forward_start, forward_end)
                    if fwd_ret is not None:
                        cp_random_returns.append(rand_sign * fwd_ret)

                if not cp_random_returns:
                    continue

                total_checkpoints_used += 1
                day_sector_returns.extend(cp_sector_returns)
                day_random_returns.extend(cp_random_returns)
                all_pooled_sector.extend(cp_sector_returns)
                all_pooled_random.extend(cp_random_returns)

            if day_sector_returns and day_random_returns:
                day_sector_avgs.append(float(np.mean(day_sector_returns)))
                day_random_avgs.append(float(np.mean(day_random_returns)))

        n_days = len(day_sector_avgs)
        print(f"\n{'-' * 78}\nLookback {lookback_min}min / Forward {forward_min}min\n{'-' * 78}")
        print(f"Valid checkpoints/day (schedule): {len(valid_checkpoints)}  "
              f"Total checkpoint-day observations used: {total_checkpoints_used}")
        print(f"Days with at least one valid checkpoint: {n_days}/{len(trading_days)}")

        if n_days < 5:
            print("TOO FEW DAYS to trust this combination's result -- skipping stats.")
            continue

        pooled_sector_avg = float(np.mean(all_pooled_sector))
        pooled_random_avg = float(np.mean(all_pooled_random))
        pooled_std = float(np.std(all_pooled_sector))
        print(f"Pooled (NOT independent -- see original script's docstring) avg directional return -- "
              f"sector-selected: {pooled_sector_avg:+.3f}%  random: {pooled_random_avg:+.3f}%  "
              f"(pooled std {pooled_std:.3f}, N={len(all_pooled_sector)})")

        day_wins = sum(1 for s, r in zip(day_sector_avgs, day_random_avgs) if s > r)
        win_rate = day_wins / n_days
        edge_pp = float(np.mean(day_sector_avgs)) - float(np.mean(day_random_avgs))
        print(f"DAY-LEVEL (independence-respecting) edge: {edge_pp:+.3f}pp")
        print(f"DAY-LEVEL win rate: sector-selected beat random on {day_wins}/{n_days} days ({win_rate:.1%})")
        print(f"Per-day sector avg range: [{min(day_sector_avgs):+.3f}%, {max(day_sector_avgs):+.3f}%]  "
              f"(std {np.std(day_sector_avgs):.3f})")

        clears_bar = edge_pp >= MIN_EDGE_PP and win_rate >= MIN_WIN_RATE
        verdicts.append((lookback_min, forward_min, n_days, edge_pp, win_rate, clears_bar))
        print(f"Clears pre-committed bar (edge >= {MIN_EDGE_PP}pp AND win rate >= {MIN_WIN_RATE:.0%})? "
              f"{'YES' if clears_bar else 'no'}")

    print(f"\n{'=' * 78}\nFINAL VERDICT (Yahoo 5-min, extended history)\n{'=' * 78}")
    if not verdicts:
        print("No combination had enough days to evaluate. No conclusion possible with this cache.")
        return

    winners = [v for v in verdicts if v[5]]
    if not winners:
        print("NO lookback/forward combination showed a real, consistent edge over random "
              f"selection at the pre-committed bar (edge >= {MIN_EDGE_PP}pp AND "
              f"day-win-rate >= {MIN_WIN_RATE:.0%}).")
        print("Do not build continuous/rolling sector-momentum detection as a live feature "
              "based on this evidence.")
    else:
        best = max(winners, key=lambda v: v[3])
        print(f"Lookback {best[0]}min / Forward {best[1]}min showed the strongest edge: "
              f"{best[3]:+.3f}pp, {best[4]:.1%} day win rate, over {best[2]} days.")
        print("Still treat this as preliminary -- see the sample-size and non-independence "
              "caveats in the original script before building a live feature.")

    print(f"\nSample size: {len(trading_days)} independent TRADING DAYS (vs 21 in the original "
          f"1-min Upstox re-test) -- roughly {len(trading_days)/21:.1f}x more days, at 5-minute "
          f"instead of 1-minute underlying resolution. See this script's module docstring for the "
          f"honest trade-off between more days and coarser per-window resolution.")


if __name__ == "__main__":
    main()
