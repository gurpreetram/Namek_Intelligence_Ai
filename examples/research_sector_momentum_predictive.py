"""
RESEARCH SCRIPT -- NOT a live feature. Tests one hypothesis against
already-cached historical data only: does identifying the top-2 trending-up
and top-2 trending-down NSE sectors EARLY in the session predict which of
their member stocks show favorable momentum for the REST of the day,
better than picking stocks at random? Per this project's "no feature
addiction" discipline, this only gets built into something live if this
script shows a real, consistent edge -- not on the strength of the idea
alone.

Methodology, per cached trading day, per early-window length W in
{15, 30, 45, 60} minutes:

  1. For each of the 15 sectors in config/sector_mapping.json, compute its
     equal-weighted average % return from market open (09:15) to
     09:15+W, using only its cached, sector-mapped member stocks that
     have data that day. A sector needs at least MIN_SECTOR_MEMBERS
     stocks with valid data to be ranked that day -- a "sector return"
     from 1 stock isn't really a sector signal.
  2. Rank all valid sectors by that early return. Take the top 2 (most
     positive) and bottom 2 (most negative) -- 4 sectors total. A day is
     skipped for a given W if fewer than MIN_VALID_SECTORS sectors have
     a valid early return (can't meaningfully pick "top 2 / bottom 2"
     from a near-empty field).
  3. For every member stock of those 4 sectors, compute a DIRECTIONAL
     rest-of-day return: sign(sector's early return) * stock's own
     return from window-end to EOD square-off (15:15). Positive means
     the stock continued in the direction its sector suggested; negative
     means it reversed. This is deliberately a DIRECTIONAL metric, not
     magnitude-of-movement -- "good momentum" for a down-trending
     sector's stocks means continuing DOWN, and a metric that only
     measured |return| couldn't distinguish "kept falling" from
     "reversed and rallied just as hard", which is exactly the
     distinction a real trading signal would need to get right.
  4. Baseline (matched, same day, same metric): draw a random sample of
     stocks -- SAME size as the sector-selected group, drawn from the
     SAME 180-symbol sector-mapped pool (not the full 2373-symbol
     universe) -- deliberately, to isolate the effect of the sector-
     trend SIGNAL itself rather than confounding it with "sector-mapped
     stocks tend to be large/liquid caps that behave differently from
     the broader universe." Each random stock gets a RANDOM predicted
     direction (+1/-1, independent fair coin flips) and the SAME
     directional-return formula is applied. This baseline has ~0
     expected value by construction if sector selection carries no real
     predictive signal -- any measured edge for the sector-selected
     group over this baseline is the actual thing being tested here.

Honesty checks built in, not added after the fact:
  - Reports days-tested count per window (sample size matters, and it
    legitimately differs per window since a day can have enough data for
    a 15-min window's ranking but not survive to 60 minutes, etc.)
  - Reports day-by-day win rate (sector-selected day-average beats
    random-baseline day-average), not just a pooled average, since a
    pooled average can be dominated by a few extreme days.
  - Does NOT round a marginal or inconsistent result up to "this works" --
    the verdict section applies an explicit bar (see MIN_EDGE_PP and
    MIN_WIN_RATE below) and states plainly when no window clears it.

Run: python -m examples.research_sector_momentum_predictive
"""
from __future__ import annotations

import json
from datetime import date, time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import storage

MARKET_OPEN_TIME = time(9, 15)
EOD_SQUARE_OFF_TIME = time(15, 15)
WINDOW_LENGTHS_MIN = [15, 30, 45, 60]
MIN_SECTOR_MEMBERS = 3   # a sector's early return needs at least this many stocks with data
MIN_VALID_SECTORS = 4    # need at least this many ranked sectors to pick top-2/bottom-2 meaningfully
RANDOM_SEED = 42

# Verdict thresholds -- deliberately explicit and pre-committed here rather
# than eyeballed after seeing the numbers. An "edge" must be BOTH a real
# average-return gap AND consistent across most days, not just a big
# number pulled up by a couple of outlier days.
MIN_EDGE_PP = 0.15       # sector-avg minus random-avg, in percentage points
MIN_WIN_RATE = 0.55      # fraction of days sector-selected beat random


def _minutes_after_open(minutes: int) -> time:
    total = MARKET_OPEN_TIME.hour * 60 + MARKET_OPEN_TIME.minute + minutes
    return time(total // 60, total % 60)


def _window_return(day_df: pd.DataFrame, start: time, end_inclusive: time) -> float | None:
    """% return from `start` bar's open to the last bar at/before
    `end_inclusive`'s close. None if no bars in range."""
    window = day_df.between_time(start, end_inclusive, inclusive="both")
    if window.empty:
        return None
    open_price = float(window["open"].iloc[0])
    close_price = float(window["close"].iloc[-1])
    if open_price <= 0:
        return None
    return (close_price - open_price) / open_price * 100.0


@dataclass
class DayWindowResult:
    day: date
    window_min: int
    up_sectors: list[str]
    down_sectors: list[str]
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

    # All dates present, EXCLUDING any date where fewer than 90% of the
    # mapped universe has data -- confirmed one such partial day exists
    # (2026-07-08, only 5/180 symbols, an incomplete/mid-fetch snapshot),
    # which would corrupt sector rankings computed from near-nothing.
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

    rng = np.random.default_rng(RANDOM_SEED)
    results: dict[int, list[DayWindowResult]] = {w: [] for w in WINDOW_LENGTHS_MIN}

    for day in trading_days:
        day_bars: dict[str, pd.DataFrame] = {}
        for symbol, df in symbol_data.items():
            day_df = df[df.index.date == day]
            if not day_df.empty:
                day_bars[symbol] = day_df

        for window_min in WINDOW_LENGTHS_MIN:
            window_end = _minutes_after_open(window_min)

            # Step 1: each sector's early-window return.
            sector_early_return: dict[str, float] = {}
            for sector, members in by_sector.items():
                member_returns = []
                for symbol in members:
                    df = day_bars.get(symbol)
                    if df is None:
                        continue
                    r = _window_return(df, MARKET_OPEN_TIME, window_end)
                    if r is not None:
                        member_returns.append(r)
                if len(member_returns) >= MIN_SECTOR_MEMBERS:
                    sector_early_return[sector] = sum(member_returns) / len(member_returns)

            if len(sector_early_return) < MIN_VALID_SECTORS:
                continue  # not enough sectors ranked today for this window -- skip

            ranked = sorted(sector_early_return.items(), key=lambda kv: kv[1], reverse=True)
            up_sectors = [s for s, _ in ranked[:2]]
            down_sectors = [s for s, _ in ranked[-2:]]
            selected_sectors = {s: sector_early_return[s] for s in up_sectors + down_sectors}

            # Step 3: directional rest-of-day return for each selected
            # sector's members.
            sector_directional_returns: list[float] = []
            selected_symbols: set[str] = set()
            for sector, early_ret in selected_sectors.items():
                sign = 1.0 if early_ret >= 0 else -1.0
                for symbol in by_sector[sector]:
                    df = day_bars.get(symbol)
                    if df is None:
                        continue
                    rest_ret = _window_return(df, window_end, EOD_SQUARE_OFF_TIME)
                    if rest_ret is None:
                        continue
                    sector_directional_returns.append(sign * rest_ret)
                    selected_symbols.add(symbol)

            if not sector_directional_returns:
                continue

            # Step 4: matched random baseline, same day, same pool, same
            # sample size, random direction instead of sector-implied one.
            candidate_pool = [s for s in day_bars if s in mapping]
            n = len(sector_directional_returns)
            if len(candidate_pool) < n:
                continue
            random_symbols = rng.choice(candidate_pool, size=n, replace=False)
            random_directions = rng.choice([1.0, -1.0], size=n)
            random_directional_returns: list[float] = []
            for symbol, rand_sign in zip(random_symbols, random_directions):
                df = day_bars[symbol]
                rest_ret = _window_return(df, window_end, EOD_SQUARE_OFF_TIME)
                if rest_ret is not None:
                    random_directional_returns.append(rand_sign * rest_ret)

            if not random_directional_returns:
                continue

            results[window_min].append(DayWindowResult(
                day=day, window_min=window_min, up_sectors=up_sectors, down_sectors=down_sectors,
                sector_directional_returns=sector_directional_returns,
                random_directional_returns=random_directional_returns,
            ))

    # --- Report ---
    print("=" * 78)
    print("SECTOR-MOMENTUM PREDICTIVE POWER -- RESEARCH RESULT (not a live feature)")
    print("=" * 78)

    verdicts = []
    for window_min in WINDOW_LENGTHS_MIN:
        day_results = results[window_min]
        n_days = len(day_results)
        print(f"\n{'-' * 78}\nWindow: {window_min} minutes from open\n{'-' * 78}")
        print(f"Days tested: {n_days} / {len(trading_days)}")
        if n_days < 5:
            print("TOO FEW DAYS to trust this window's result -- skipping stats.")
            continue

        all_sector_returns = [r for dr in day_results for r in dr.sector_directional_returns]
        all_random_returns = [r for dr in day_results for r in dr.random_directional_returns]
        pooled_sector_avg = float(np.mean(all_sector_returns))
        pooled_random_avg = float(np.mean(all_random_returns))

        day_sector_avgs = [float(np.mean(dr.sector_directional_returns)) for dr in day_results]
        day_random_avgs = [float(np.mean(dr.random_directional_returns)) for dr in day_results]
        day_wins = sum(1 for s, r in zip(day_sector_avgs, day_random_avgs) if s > r)
        win_rate = day_wins / n_days

        edge_pp = pooled_sector_avg - pooled_random_avg
        print(f"Pooled avg directional return -- sector-selected: {pooled_sector_avg:+.3f}%  "
              f"random baseline: {pooled_random_avg:+.3f}%")
        print(f"Edge (sector - random): {edge_pp:+.3f} percentage points")
        print(f"Day-by-day: sector-selected beat random baseline on {day_wins}/{n_days} days ({win_rate:.1%})")
        print(f"Per-day sector avg range: [{min(day_sector_avgs):+.2f}%, {max(day_sector_avgs):+.2f}%]  "
              f"(std {np.std(day_sector_avgs):.2f})")

        clears_bar = edge_pp >= MIN_EDGE_PP and win_rate >= MIN_WIN_RATE
        verdicts.append((window_min, n_days, edge_pp, win_rate, clears_bar))
        print(f"Clears pre-committed bar (edge >= {MIN_EDGE_PP}pp AND win rate >= {MIN_WIN_RATE:.0%})? "
              f"{'YES' if clears_bar else 'no'}")

    print(f"\n{'=' * 78}\nFINAL VERDICT\n{'=' * 78}")
    if not verdicts:
        print("No window had enough days to evaluate. No conclusion possible with this cache.")
        return

    winners = [v for v in verdicts if v[4]]
    if not winners:
        print("NO time-window showed a real, consistent edge over random selection at the "
              f"pre-committed bar (edge >= {MIN_EDGE_PP}pp AND day-win-rate >= {MIN_WIN_RATE:.0%}).")
        print("Do not build this as a live feature based on this evidence.")
    else:
        best = max(winners, key=lambda v: v[2])
        print(f"Window {best[0]} min showed the strongest edge: {best[2]:+.3f}pp, "
              f"{best[3]:.1%} day win rate, over {best[1]} days.")
        print("Still treat this as preliminary -- see the sample-size caveat above before building a live feature.")

    print(f"\nSample size caveat: only {len(trading_days)} trading days are cached "
          f"(~1 calendar month). This is a SMALL sample for any statistical claim -- "
          f"treat every number above as a first look, not a proven edge, regardless of "
          f"which way it points.")


if __name__ == "__main__":
    main()
