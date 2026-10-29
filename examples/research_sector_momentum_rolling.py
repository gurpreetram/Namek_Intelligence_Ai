"""
RESEARCH SCRIPT -- NOT a live feature. Corrected re-test of the sector-
momentum hypothesis from examples/research_sector_momentum_predictive.py,
which tested a single fixed early-window (15/30/45/60 min) predicting the
REST of the day and found no edge. This version tests a different, more
plausible mechanism: CONTINUOUS rolling detection -- at many checkpoints
through the day, is a sector moving RIGHT NOW (short trailing lookback),
and if so, do its stocks keep moving in the NEAR-TERM future (a short
forward window immediately after), not the rest of the whole day.

Methodology, per cached trading day, per (lookback_min, forward_min)
combination in COMBINATIONS below:

  1. Checkpoints every CHECKPOINT_INTERVAL_MIN (15) minutes from 09:30 to
     15:00. A checkpoint is only valid for a given combo if its lookback
     window doesn't reach before market open (09:15) and its forward
     window doesn't reach past EOD square-off (15:15).
  2. At each valid checkpoint, compute each sector's TRAILING return over
     the preceding `lookback_min` minutes (equal-weighted average of
     member stocks with data, min MIN_SECTOR_MEMBERS to count).
  3. Take the top-2 up / bottom-2 down sectors by that trailing return.
  4. For every member stock of those 4 sectors, compute a DIRECTIONAL
     FORWARD return over the next `forward_min` minutes: sign(sector's
     trailing return) * stock's own forward return. Same convention as
     the prior script -- positive means the stock continued in the
     direction its sector was just moving.
  5. Matched random baseline, same checkpoint/day: random stocks from the
     same 180-symbol sector-mapped pool, same sample size, random
     direction instead of the sector-implied one.

AUTOCORRELATION / INDEPENDENCE, addressed directly (this is the reason a
separate, more careful script was needed, not just "run the old one more
often"): checkpoints 15 minutes apart on the same day share overlapping
lookback/forward windows and are NOT independent observations. Treating
every checkpoint as one independent data point (21 days x ~20 checkpoints
= ~400 "observations") would be pseudo-replication -- it would make noisy
within-day correlation look like broad, independent confirmation. This
script reports TWO different things and does not conflate them:
  - POOLED average across every checkpoint-stock observation: a big-N
    point estimate of the average directional return, useful for seeing
    the sign and rough size of any effect, but NOT a valid basis for a
    confidence/significance claim (the N overstates independent evidence).
  - DAY-LEVEL win rate: each of the 21 days is collapsed to ONE number
    (that day's average across all its checkpoints, separately for
    sector-selected and random) before comparing sector vs random. This
    is the statistically honest unit of independence here, and it is
    what "day-by-day win rate" means in this script (matching the
    earlier script's per-day comparison, just each "day" now aggregates
    many checkpoints instead of representing a single one).

Verdict bar: same as the earlier script (edge >= 0.15pp, day-win-rate >=
55%), kept identical rather than re-tuned, specifically so this result is
directly comparable to the earlier one rather than moving the goalposts.
Actual per-checkpoint return magnitudes are reported so the bar's
stringency at this shorter horizon can be judged in context.

Run: python -m examples.research_sector_momentum_rolling
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

from data import storage

MARKET_OPEN_TIME = time(9, 15)
EOD_SQUARE_OFF_TIME = time(15, 15)
CHECKPOINT_INTERVAL_MIN = 15
CHECKPOINT_START = time(9, 30)
CHECKPOINT_END = time(15, 0)

# (lookback_minutes, forward_minutes) combinations requested.
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
    print(f"Loading {len(mapping)} sector-mapped symbols...")
    symbol_data = {s: storage.load(s) for s in mapping}
    symbol_data = {s: df for s, df in symbol_data.items() if not df.empty}
    print(f"  {len(symbol_data)} have cached data\n")

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
    print(f"Trading days with full coverage: {len(trading_days)}")
    if excluded_days:
        print(f"Excluded as partial/incomplete: {[(str(d), date_counts[d]) for d in excluded_days]}\n")

    all_checkpoints = _generate_checkpoints()
    print(f"Checkpoint grid: {len(all_checkpoints)} checkpoints/day, every {CHECKPOINT_INTERVAL_MIN} min, "
          f"{CHECKPOINT_START}-{CHECKPOINT_END}\n")

    rng = np.random.default_rng(RANDOM_SEED)

    # Pre-load each day's per-symbol day-frame once, reused across all 4 combos.
    day_bars_cache: dict[date, dict[str, pd.DataFrame]] = {}
    for day in trading_days:
        day_bars: dict[str, pd.DataFrame] = {}
        for symbol, df in symbol_data.items():
            day_df = df[df.index.date == day]
            if not day_df.empty:
                day_bars[symbol] = day_df
        day_bars_cache[day] = day_bars

    print("=" * 78)
    print("SECTOR-MOMENTUM ROLLING PREDICTIVE POWER -- RESEARCH RESULT (not a live feature)")
    print("=" * 78)

    verdicts = []
    for lookback_min, forward_min in COMBINATIONS:
        valid_checkpoints = [
            cp for cp in all_checkpoints
            if _time_to_minutes(cp) - lookback_min >= _time_to_minutes(MARKET_OPEN_TIME)
            and _time_to_minutes(cp) + forward_min <= _time_to_minutes(EOD_SQUARE_OFF_TIME)
        ]

        # Per-day aggregates (the independence unit -- see module docstring).
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
        print(f"Pooled (NOT independent -- see docstring) avg directional return -- "
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

    print(f"\n{'=' * 78}\nFINAL VERDICT\n{'=' * 78}")
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
              "caveats above before building a live feature.")

    print(f"\nSample size caveat: only {len(trading_days)} independent TRADING DAYS are cached "
          f"(~1 calendar month). The rolling design generates many more checkpoint observations "
          f"per day than the earlier single-checkpoint test, but those checkpoints are NOT "
          f"independent of each other (heavy overlap within a day) -- the day-level numbers above "
          f"are the ones that actually respect that, and 21 days is still a small sample for any "
          f"of them.")


if __name__ == "__main__":
    main()
