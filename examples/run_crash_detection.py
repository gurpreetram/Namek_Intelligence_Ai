"""
Tests a "crash detection" hypothesis: does a RAPID, LARGE decline in a
stock's first 15 minutes of trading (09:15-09:30) -- a genuine "something
is wrong" move, not ordinary opening volatility -- predict anything
tradeable? Tests four angles on the SAME underlying event: does the
stock itself keep falling (continuation/SHORT) or bounce back
(reversal/LONG), and does the crash spread to other stocks in the same
sector (occurrence + a sector-mates trading test)?

PRE-REGISTERED BARS, set BEFORE running this and not adjusted after
seeing results -- same bars used by every other hypothesis test in this
project (examples/run_gap_fill_vs_continuation.py,
examples/run_sector_bellwether_leadlag.py):
  - INDIVIDUAL-STOCK TRADING EDGE (continuation or reversal) is real only
    if BOTH hold on the day-level return series: day-level Sharpe
    (non-annualized) >= 0.5 AND day-level win rate >= 60%.
  - SECTOR-IMPACT OCCURRENCE is real, not sampling noise, only if the
    sector-follower correlated-decline rate exceeds the random-stock-group
    baseline rate by >= 15 percentage points (same bar as the bellwether
    script's lead-lag occurrence test).
  - SECTOR-IMPACT TRADING EDGE (shorting sector-mates after a crash
    trigger) uses the same Sharpe/win-rate bar as the individual tests,
    and its significance is CONDITIONED on the occurrence bar above being
    met -- an edge computed on top of a non-real occurrence pattern isn't
    a real edge, it's overfitting to the same noise twice.
Meeting only one applicable bar is suggestive, not confirmed. Meeting
neither is noise.

CRASH DEFINITION (STEP 1): for each LIQUID symbol (detectors.
liquidity_analysis.analyze_liquidity() applied to YESTERDAY's data,
BEFORE any event is even considered -- liquidity filter applied FROM THE
START, not bolted on after finding events, per the lesson learned from
the gap-fill research pass where an unfiltered first look produced a
false-positive "edge" made of illiquid/circuit-locked names) on each
cached day: scanning bar-by-bar from 09:15 to 09:30 using detectors.
sector_momentum.compute_stock_return_today() (reused as-is -- no
reimplementation of the open-to-now return calc), the FIRST bar where
both hold is that day's crash trigger:
  - decline_pct <= -2.5% (today's open to that bar's close) -- a much
    larger, sharper threshold than this project's normal ORB opening-
    range logic, specifically meant to catch outsized "something is
    wrong" moves, not ordinary volatility.
  - volume_zscore(that bar's volume, baseline) >= 1.0, reusing detectors.
    breakout_validator.volume_zscore() as-is (not reimplemented).
    BASELINE CHOICE: deliberately NOT breakout_validator's own convention
    of "whatever N bars immediately precede this bar" -- for a 09:1x bar,
    the immediately-preceding bars are literally YESTERDAY's last bars
    (end-of-day volume), and opening volume is almost always higher than
    closing volume for almost every stock on almost every day. Using that
    convention here would make the "elevated volume" filter nearly
    meaningless (it would trivially pass most mornings regardless of
    anything unusual happening). Instead, the baseline is a TIME-OF-DAY-
    MATCHED pool: every 09:15-09:30 bar's volume from every PRIOR cached
    day for that SAME symbol (no lookahead -- only days strictly before
    the one being evaluated are ever in the pool). This asks the honest
    question -- "is THIS stock's open unusually busy for THIS stock's OWN
    typical open" -- not "busier than last night's close," which isn't
    the same claim. Requires >= MIN_BASELINE_PRIOR_DAYS prior days pooled
    before a symbol-day is even eligible for detection (an early cached
    day with only 1-2 prior days has too thin a baseline to trust a
    z-score from).

ENTRY (STEP 2): NEXT bar's open strictly AFTER 09:30 -- the "confirmation
after 9:30" approach, and the same "next bar after cutoff" no-lookahead
convention examples/run_sector_bellwether_leadlag.py's trading test uses
(the decision is made once the 09:15-09:30 window is over; the bar that
was still forming AT 09:30 cannot be entered at a price decided using its
own future close). EXIT: EOD square-off at or before 15:15, via
detectors.cross_sectional_ranking.bar_at_or_before() (reused as-is), same
convention as every other script in this project.

TWO OPPOSING BETS on the SAME crash events (STEP 2):
  - continuation: SHORT, betting the decline continues through EOD.
  - reversal: LONG, betting on a bounce back toward the pre-crash level.

SECTOR IMPACT (STEP 3): for crash events in a sector-mapped stock
(config/sector_mapping.json), checks whether OTHER stocks in the SAME
sector show a correlated DECLINE (>= a 0.3% noise floor, same floor the
bellwether script uses) in the FOLLOWING 20 minutes (midpoint of the
requested 15-30 min range, same reasoning/value the bellwether script
uses for its own trigger/follow windows) -- against a RANDOM baseline
group of equal size drawn from OUTSIDE the triggering sector, on the
SAME day/trigger-time, to control for "the whole market was falling that
morning" confounds. If that occurrence gap clears the pre-registered bar,
tests actually shorting the sector-mates (entered next-bar-open after the
follow window, exited EOD) against a RANDOM-ENTRY-TIME control (same
stocks/days, entered at a random 09:15-09:30-window bar instead) --
isolates whether the crash-triggered TIMING itself adds value, holding
the stocks/days/direction fixed.

CONCENTRATION (STEP 4): whichever combination(s) above clear BOTH
pre-registered bars get tested at N=1/3/5 trades/day (ranked by the
triggering event's |decline_pct|, deepest crash first -- for the
sector-impact case, by the trigger's own |decline_pct|, since multiple
sector-mate trades can share one trigger), gross and net of the same
0.075% round-trip cost assumption run_gap_fill_vs_continuation.py and
run_cross_sectional_backtest.py use.

SPOT-CHECK BEFORE VERDICT (STEP 5): whichever variant looks most
promising after steps 2-4 gets 8-10 real examples printed -- symbol,
decline%, liquidity (avg traded value), entry/exit prices -- and an
explicit circuit-lock check (frozen O=H=L=C trigger bar, a run of frozen
bars immediately after it, and proximity to a standard NSE circuit band)
BEFORE the final verdict is printed, not after -- same discipline that
caught the gap-fill script's original false positive.

Run: python -m examples.run_crash_detection
    python -m examples.run_crash_detection --limit 500   # fast test
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage
from detectors.breakout_validator import volume_zscore
from detectors.cross_sectional_ranking import bar_at_or_before
from detectors.liquidity_analysis import analyze_liquidity
from detectors.scanner import MIN_STOCK_PRICE
from detectors.sector_momentum import compute_stock_return_today

CRASH_WINDOW_START = "09:15"
CRASH_WINDOW_END = "09:30"
CRASH_DECLINE_THRESHOLD_PCT = -2.5
CRASH_VOLUME_ZSCORE_MIN = 1.0
MIN_BASELINE_PRIOR_DAYS = 5
ENTRY_CUTOFF_TIME = "09:30"
EOD_SQUARE_OFF_TIME = "15:15"
MIN_SHARPE_BAR = 0.5
MIN_WINRATE_BAR = 0.60
MIN_LEADLAG_GAP_PP = 15.0
SECTOR_FOLLOW_WINDOW_MINUTES = 20
SECTOR_FOLLOW_MIN_MOVE_PCT = 0.3
N_RANDOM_DRAWS = 10
RANDOM_SEED = 42
# Same assumption/reasoning as run_gap_fill_vs_continuation.py's
# COST_PCT_PER_ROUND_TRIP.
COST_PCT_PER_ROUND_TRIP = 0.075
CIRCUIT_BAND_PCTS = (2.0, 5.0, 10.0, 20.0)
CIRCUIT_BAND_TOLERANCE = 0.3
FROZEN_RUN_CHECK_BARS = 10


@dataclass
class CrashEvent:
    symbol: str
    day: date
    sector: str | None
    trigger_time: pd.Timestamp
    decline_pct: float
    volume_zscore: float
    avg_traded_value: float
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    trigger_bar_frozen: bool
    trigger_bar_volume: float
    frozen_run_after: int  # how many of the FROZEN_RUN_CHECK_BARS bars after the trigger are also O=H=L=C


def _load_sector_mapping() -> dict[str, str]:
    path = PROJECT_ROOT / "config" / "sector_mapping.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _frozen(bar: pd.Series) -> bool:
    return float(bar["open"]) == float(bar["high"]) == float(bar["low"]) == float(bar["close"])


def _frozen_run_after(day_df: pd.DataFrame, trigger_time: pd.Timestamp, n_bars: int) -> int:
    after = day_df[day_df.index > trigger_time].iloc[:n_bars]
    run = 0
    for _, bar in after.iterrows():
        if not _frozen(bar):
            break
        run += 1
    return run


def _find_crash_events(
    symbol_data: dict[str, pd.DataFrame], sector_mapping: dict[str, str], require_liquid: bool = True,
) -> list[CrashEvent]:
    events: list[CrashEvent] = []
    for symbol, df in symbol_data.items():
        days = sorted(set(df.index.date))
        pooled_window_volumes: list[float] = []
        prior_days_pooled = 0

        for i, day in enumerate(days):
            day_df = df[df.index.date == day]
            win_start = pd.Timestamp.combine(day, pd.Timestamp(CRASH_WINDOW_START).time()).tz_localize(day_df.index.tz)
            win_end = pd.Timestamp.combine(day, pd.Timestamp(CRASH_WINDOW_END).time()).tz_localize(day_df.index.tz)
            window_df = day_df[(day_df.index >= win_start) & (day_df.index <= win_end)]

            eligible = i > 0 and not window_df.empty and prior_days_pooled >= MIN_BASELINE_PRIOR_DAYS
            if eligible:
                prev_day_df = df[df.index.date == days[i - 1]]
                yesterday_close = float(prev_day_df["close"].iloc[-1]) if not prev_day_df.empty else 0.0
                liquidity = analyze_liquidity(prev_day_df) if not prev_day_df.empty else None
                if yesterday_close < MIN_STOCK_PRICE:
                    eligible = False
                elif require_liquid and (liquidity is None or not liquidity.is_liquid):
                    eligible = False

            if eligible:
                baseline = pd.Series(pooled_window_volumes)
                for bar_time in window_df.index:
                    ret = compute_stock_return_today(df, bar_time, lookback_minutes=None)
                    if ret is None or ret > CRASH_DECLINE_THRESHOLD_PCT:
                        continue
                    bar_volume = float(df.loc[bar_time, "volume"])
                    z = volume_zscore(bar_volume, baseline)
                    if z < CRASH_VOLUME_ZSCORE_MIN:
                        continue

                    cutoff_ts = pd.Timestamp.combine(day, pd.Timestamp(ENTRY_CUTOFF_TIME).time()).tz_localize(day_df.index.tz)
                    entry_candidates = day_df[day_df.index > cutoff_ts]
                    if entry_candidates.empty:
                        break
                    entry_bar = entry_candidates.iloc[0]

                    eod_ts = pd.Timestamp.combine(day, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(day_df.index.tz)
                    exit_bar = bar_at_or_before(df, eod_ts)
                    if exit_bar is None or exit_bar.name <= entry_bar.name:
                        break

                    trigger_bar = df.loc[bar_time]
                    events.append(CrashEvent(
                        symbol=symbol, day=day, sector=sector_mapping.get(symbol),
                        trigger_time=bar_time, decline_pct=ret, volume_zscore=z,
                        avg_traded_value=liquidity.avg_traded_value if liquidity is not None else float("nan"),
                        entry_time=entry_bar.name, entry_price=float(entry_bar["open"]),
                        exit_time=exit_bar.name, exit_price=float(exit_bar["close"]),
                        trigger_bar_frozen=_frozen(trigger_bar), trigger_bar_volume=bar_volume,
                        frozen_run_after=_frozen_run_after(day_df, bar_time, FROZEN_RUN_CHECK_BARS),
                    ))
                    break  # first qualifying bar only, per symbol-day

            if not window_df.empty:
                pooled_window_volumes.extend(window_df["volume"].tolist())
                prior_days_pooled += 1

    return events


def _day_stats(daily_returns: np.ndarray) -> tuple[float, float, float, float]:
    if len(daily_returns) == 0:
        return 0.0, 0.0, 0.0, 0.0
    win_rate = float((daily_returns > 0).mean())
    mean = float(daily_returns.mean())
    std = float(daily_returns.std())
    sharpe = mean / std if std > 0 else 0.0
    return win_rate, mean, std, sharpe


def _verdict(sharpe: float, win_rate: float) -> str:
    if sharpe >= MIN_SHARPE_BAR and win_rate >= MIN_WINRATE_BAR:
        return "REAL EDGE (meets both pre-registered bars)"
    if sharpe >= MIN_SHARPE_BAR or win_rate >= MIN_WINRATE_BAR:
        return "suggestive only (meets one bar)"
    return "NOISE (meets neither pre-registered bar)"


def _report_individual(label: str, events: list[CrashEvent], sign: float) -> tuple[float, float, int]:
    """sign: -1.0 for continuation (SHORT), +1.0 for reversal (LONG).
    Returns (sharpe, win_rate, n_days) for the STEP 6 summary table."""
    if not events:
        print(f"\n--- {label} ---\nNo qualifying events.")
        return 0.0, 0.0, 0
    pos_returns = []
    by_day: dict[date, list[float]] = {}
    for e in events:
        ret_pct = sign * (e.exit_price - e.entry_price) / e.entry_price * 100.0
        pos_returns.append(ret_pct)
        by_day.setdefault(e.day, []).append(ret_pct)
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


def _forward_return_pct(df: pd.DataFrame, start_ts: pd.Timestamp, forward_minutes: int) -> float | None:
    """Same helper as run_sector_bellwether_leadlag.py's -- forward-looking
    by design (a historical/analytical pass over already-complete cached
    days, not a live decision). Not shared via import since neither script
    exposes it as a public detectors/ function (both are one-off research
    passes); duplicated deliberately rather than introducing a shared
    module for a single 20-line helper used twice."""
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


DATA_GAP_EARLY_EXIT_MINUTES = 30  # exit_bar found meaningfully before 15:15 => that symbol-day's cache
# likely doesn't cover the full afternoon (a fetch/cache gap, not a real EOD square-off) -- bar_at_or_before()
# silently returns the LAST bar it has, which for a truncated day is well before 15:15. Same underlying
# behavior every other script using bar_at_or_before() inherits; flagged here explicitly because an
# artificially early "EOD" exit changes that specific event's return in an unpredictable direction and
# should be eyeballed before trusting it, same spirit as the circuit-lock check.


def _data_gap_flag(e: CrashEvent) -> str | None:
    eod_minutes = 15 * 60 + 15
    exit_minutes = e.exit_time.hour * 60 + e.exit_time.minute
    if eod_minutes - exit_minutes >= DATA_GAP_EARLY_EXIT_MINUTES:
        return f"POSSIBLE DATA GAP: exit found at {e.exit_time.strftime('%H:%M')}, {eod_minutes - exit_minutes} min before the 15:15 EOD target -- this symbol-day's cache likely doesn't reach EOD; treat this event's return with caution"
    return None


def _circuit_flag(e: CrashEvent) -> str:
    near_band = any(abs(abs(e.decline_pct) - band) <= CIRCUIT_BAND_TOLERANCE for band in CIRCUIT_BAND_PCTS)
    if e.trigger_bar_frozen and e.frozen_run_after >= 2:
        return f"LIKELY CIRCUIT LOCK (frozen trigger bar + {e.frozen_run_after} more frozen bars immediately after)"
    if e.trigger_bar_frozen and near_band:
        return "LIKELY CIRCUIT LOCK (frozen trigger bar + decline near a standard circuit band)"
    if e.trigger_bar_frozen:
        return "trigger bar frozen (O=H=L=C) -- worth checking, not necessarily a circuit hit"
    if near_band:
        return "decline near a standard circuit band, but trigger bar NOT frozen -- likely genuine"
    return "no circuit-lock indicators"


def _spot_check(events: list[CrashEvent], n: int, label: str) -> None:
    print(f"\n--- SPOT-CHECK ({label}): {min(n, len(events))} real events, deepest declines first ---")
    for e in sorted(events, key=lambda e: e.decline_pct)[:n]:
        print(f"  {e.symbol:14s} {e.day}  decline={e.decline_pct:+.2f}%  vol_z={e.volume_zscore:+.2f}  "
              f"avg_traded_value=Rs.{e.avg_traded_value:,.0f}/min  "
              f"trigger={e.trigger_time.strftime('%H:%M')}  "
              f"entry={e.entry_price:.2f} ({e.entry_time.strftime('%H:%M')})  "
              f"exit={e.exit_price:.2f} ({e.exit_time.strftime('%H:%M')})")
        print(f"    {_circuit_flag(e)}")
        gap_flag = _data_gap_flag(e)
        if gap_flag:
            print(f"    {gap_flag}")


def _report_top_n_concentration(
    labeled: list[tuple[date, float, float]], top_n: int, cost_pct: float, label: str,
) -> None:
    """labeled: (day, |rank_key|, gross_return_pct) tuples -- gross_return_pct
    already carries the correct direction sign. Groups by day, keeps the
    top_n by rank_key each day, reports gross/net day-level stats. Same
    rigor as run_gap_fill_vs_continuation.py's --top-n concentration test."""
    by_day: dict[date, list[tuple[float, float]]] = {}
    for day, rank_key, gross in labeled:
        by_day.setdefault(day, []).append((rank_key, gross))

    day_gross: dict[date, list[float]] = {}
    day_net: dict[date, list[float]] = {}
    pos_gross_all: list[float] = []
    pos_net_all: list[float] = []
    for day, items in by_day.items():
        top = sorted(items, key=lambda t: t[0], reverse=True)[:top_n]
        for _, gross in top:
            net = gross - cost_pct
            day_gross.setdefault(day, []).append(gross)
            day_net.setdefault(day, []).append(net)
            pos_gross_all.append(gross)
            pos_net_all.append(net)

    gross_arr = np.array([float(np.mean(v)) for v in day_gross.values()])
    net_arr = np.array([float(np.mean(v)) for v in day_net.values()])
    g_win, g_mean, g_std, g_sharpe = _day_stats(gross_arr)
    n_win, n_mean, n_std, n_sharpe = _day_stats(net_arr)
    pos_gross_all = np.array(pos_gross_all)
    pos_net_all = np.array(pos_net_all)
    avg_per_day = pos_gross_all.size / len(by_day) if by_day else 0.0

    print(f"\n--- {label}: top-{top_n}, avg {avg_per_day:.1f} positions/day, cost={cost_pct:.3f}%/position ---")
    print(f"Days used: {len(by_day)}  |  Total positions: {pos_gross_all.size}")
    print(f"Position-level -- GROSS win rate: {(pos_gross_all > 0).mean():.1%}  avg: {pos_gross_all.mean():+.3f}%   "
          f"NET win rate: {(pos_net_all > 0).mean():.1%}  avg: {pos_net_all.mean():+.3f}%")
    print(f"Day-level GROSS -- win rate: {g_win:.1%}  avg daily return: {g_mean:+.3f}%  std: {g_std:.3f}%  Sharpe: {g_sharpe:.2f}   [{_verdict(g_sharpe, g_win)}]")
    print(f"Day-level NET   -- win rate: {n_win:.1%}  avg daily return: {n_mean:+.3f}%  std: {n_std:.3f}%  Sharpe: {n_sharpe:.2f}   [{_verdict(n_sharpe, n_win)}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="cap universe size, for a fast test run")
    parser.add_argument("--no-liquidity-filter", action="store_true",
                         help="Reproduce an UNFILTERED comparison run -- does NOT apply analyze_liquidity() "
                              "before accepting a crash event. Default is filtered (liquid symbols only), "
                              "applied FROM THE START per the gap-fill lesson.")
    parser.add_argument("--cost-pct", type=float, default=COST_PCT_PER_ROUND_TRIP)
    args = parser.parse_args()
    rng = np.random.default_rng(RANDOM_SEED)

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
    require_liquid = not args.no_liquidity_filter

    print("=" * 70)
    print("STEP 1: crash event detection (>=-2.5% in first 15min, volume_zscore>=1.0), "
          f"liquidity filter {'ON' if require_liquid else 'OFF'} FROM THE START")
    print("=" * 70)
    t0 = time.monotonic()
    events = _find_crash_events(symbol_data, sector_mapping, require_liquid=require_liquid)
    print(f"Found {len(events)} qualifying crash events in {time.monotonic() - t0:.1f}s")
    all_days = sorted({e.day for e in events})
    sector_events = [e for e in events if e.sector is not None]
    print(f"Distinct trading days with >=1 crash event: {len(all_days)}")
    print(f"Of which sector-mapped (eligible for STEP 3): {len(sector_events)} events")
    print("=" * 70)
    print("SAMPLE-SIZE CAVEAT: results below come from Upstox's ~29-33 day rolling "
          "1-minute cache. This is a SMALL sample -- treat all findings as preliminary.")
    print("=" * 70)

    if not events:
        print("\nNo crash events found in cached history -- nothing further to test.")
        return

    print("\n" + "=" * 70)
    print("STEP 2: individual-stock continuation (SHORT) vs reversal (LONG)")
    print("=" * 70)
    short_sharpe, short_win, short_days = _report_individual("Continuation (SHORT the crash)", events, sign=-1.0)
    long_sharpe, long_win, long_days = _report_individual("Reversal (LONG, bet on a bounce)", events, sign=1.0)

    print("\n" + "=" * 70)
    print("STEP 3: sector-impact occurrence + trading test")
    print("=" * 70)
    by_sector: dict[str, list[str]] = {}
    for symbol, sector in sector_mapping.items():
        by_sector.setdefault(sector, []).append(symbol)
    all_mapped_symbols = [s for s in sector_mapping if s in symbol_data]

    sector_agree, sector_total = 0, 0
    random_agree, random_total = 0, 0
    trading_records: list[dict] = []

    for ev in sector_events:
        followers = [s for s in by_sector.get(ev.sector, []) if s != ev.symbol and s in symbol_data]
        if not followers:
            continue
        outside_pool = [s for s in all_mapped_symbols if sector_mapping.get(s) != ev.sector]
        if len(outside_pool) < len(followers):
            continue
        random_group = list(rng.choice(outside_pool, size=len(followers), replace=False))

        for follower in followers:
            fret = _forward_return_pct(symbol_data[follower], ev.trigger_time, SECTOR_FOLLOW_WINDOW_MINUTES)
            if fret is None:
                continue
            sector_total += 1
            followed = fret <= -SECTOR_FOLLOW_MIN_MOVE_PCT
            if followed:
                sector_agree += 1
            trading_records.append({"follower": follower, "day": ev.day, "trigger_time": ev.trigger_time,
                                     "decline_pct": ev.decline_pct})

        for rsym in random_group:
            rret = _forward_return_pct(symbol_data[rsym], ev.trigger_time, SECTOR_FOLLOW_WINDOW_MINUTES)
            if rret is None:
                continue
            random_total += 1
            if rret <= -SECTOR_FOLLOW_MIN_MOVE_PCT:
                random_agree += 1

    sector_rate = sector_agree / sector_total if sector_total else 0.0
    random_rate = random_agree / random_total if random_total else 0.0
    gap_pp = (sector_rate - random_rate) * 100.0
    print(f"Sector-mapped trigger events: {len(sector_events)}  |  Sector-follower observations: {sector_total}  |  "
          f"Random-control observations: {random_total}")
    print(f"Sector-follower correlated-decline rate: {sector_rate:.1%}")
    print(f"Random-group correlated-decline rate:    {random_rate:.1%}")
    print(f"Gap: {gap_pp:+.1f} percentage points")
    occurrence_real = gap_pp >= MIN_LEADLAG_GAP_PP
    print(f"PRE-REGISTERED BAR (gap >= 15pp): {'REAL (gap >= 15pp)' if occurrence_real else 'NOT SUPPORTED (gap < 15pp pre-registered bar)'}")

    sector_sharpe, sector_win, sector_days = 0.0, 0.0, 0
    if not trading_records:
        print("\nNo sector trading records -- skipping trading test.")
    else:
        real_day_returns: dict[date, list[float]] = {}
        random_day_returns: dict[date, list[float]] = {}
        sector_labeled: list[tuple[date, float, float]] = []

        for rec in trading_records:
            df = symbol_data[rec["follower"]]
            day = rec["day"]
            day_df = df[df.index.date == day]
            eod_ts = pd.Timestamp.combine(day, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(day_df.index.tz)
            exit_bars = day_df[day_df.index <= eod_ts]
            if exit_bars.empty:
                continue
            exit_price = float(exit_bars["close"].iloc[-1])

            follow_end = rec["trigger_time"] + pd.Timedelta(minutes=SECTOR_FOLLOW_WINDOW_MINUTES)
            entry_candidates = day_df[day_df.index > follow_end]
            if entry_candidates.empty:
                continue
            entry_bar = entry_candidates.iloc[0]
            entry_price = float(entry_bar["open"])
            if entry_bar.name >= exit_bars.index[-1]:
                continue
            real_ret = -1.0 * (exit_price - entry_price) / entry_price * 100.0  # SHORT the sector-mate
            real_day_returns.setdefault(day, []).append(real_ret)
            sector_labeled.append((day, abs(rec["decline_pct"]), real_ret))

            win_start = pd.Timestamp.combine(day, pd.Timestamp(CRASH_WINDOW_START).time()).tz_localize(day_df.index.tz)
            win_end = pd.Timestamp.combine(day, pd.Timestamp(CRASH_WINDOW_END).time()).tz_localize(day_df.index.tz)
            window_bars = day_df[(day_df.index >= win_start) & (day_df.index <= win_end) & (day_df.index < exit_bars.index[-1])]
            if window_bars.empty:
                continue
            draws = rng.choice(len(window_bars), size=min(N_RANDOM_DRAWS, len(window_bars)), replace=False)
            draw_rets = []
            for idx in draws:
                rand_entry_price = float(window_bars.iloc[idx]["open"])
                draw_rets.append(-1.0 * (exit_price - rand_entry_price) / rand_entry_price * 100.0)
            random_day_returns.setdefault(day, []).append(float(np.mean(draw_rets)))

        real_arr = np.array([float(np.mean(v)) for v in real_day_returns.values()])
        random_arr = np.array([float(np.mean(v)) for v in random_day_returns.values()])
        sector_win, sector_mean, sector_std, sector_sharpe = _day_stats(real_arr)
        r_win, r_mean, r_std, r_sharpe = _day_stats(random_arr)
        sector_days = len(real_arr)

        print(f"\nTRADING TEST: SHORT sector-mates after crash trigger, vs random-entry-time control")
        print(f"REAL (post-trigger entry): days={len(real_arr)}  win_rate={sector_win:.1%}  "
              f"avg_daily_return={sector_mean:+.3f}%  std={sector_std:.3f}%  Sharpe={sector_sharpe:.2f}")
        print(f"RANDOM-TIME control:       days={len(random_arr)}  win_rate={r_win:.1%}  "
              f"avg_daily_return={r_mean:+.3f}%  std={r_std:.3f}%  Sharpe={r_sharpe:.2f}")
        sector_verdict = _verdict(sector_sharpe, sector_win) if occurrence_real else \
            "NOT EVALUATED (occurrence bar not met -- an edge on top of a non-real occurrence pattern isn't a real edge)"
        print(f"PRE-REGISTERED BAR (Sharpe>=0.5 AND win rate>=60%, CONDITIONED on occurrence being real): {sector_verdict}")

    # ---- STEP 4: concentration test on whichever combo(s) clear both bars ----
    print("\n" + "=" * 70)
    print("STEP 4: concentration test (only combos clearing BOTH pre-registered bars)")
    print("=" * 70)
    winners = []
    if short_sharpe >= MIN_SHARPE_BAR and short_win >= MIN_WINRATE_BAR:
        winners.append("continuation")
    if long_sharpe >= MIN_SHARPE_BAR and long_win >= MIN_WINRATE_BAR:
        winners.append("reversal")
    if occurrence_real and sector_sharpe >= MIN_SHARPE_BAR and sector_win >= MIN_WINRATE_BAR:
        winners.append("sector-impact")

    if not winners:
        print("\nNo combination cleared both pre-registered bars -- skipping concentration test "
              "(nothing to concentrate; a bigger bet on noise is still noise).")
    else:
        print(f"\nCombination(s) clearing both bars: {', '.join(winners)}")
        if "continuation" in winners:
            labeled = [(e.day, abs(e.decline_pct), -1.0 * (e.exit_price - e.entry_price) / e.entry_price * 100.0) for e in events]
            for n in (1, 3, 5):
                _report_top_n_concentration(labeled, n, args.cost_pct, "Continuation (SHORT), deepest crash first")
        if "reversal" in winners:
            labeled = [(e.day, abs(e.decline_pct), 1.0 * (e.exit_price - e.entry_price) / e.entry_price * 100.0) for e in events]
            for n in (1, 3, 5):
                _report_top_n_concentration(labeled, n, args.cost_pct, "Reversal (LONG), deepest crash first")
        if "sector-impact" in winners:
            for n in (1, 3, 5):
                _report_top_n_concentration(sector_labeled, n, args.cost_pct, "Sector-impact SHORT, ranked by trigger's |decline%|")

    # ---- STEP 5: spot-check BEFORE verdict ----
    print("\n" + "=" * 70)
    print("STEP 5: spot-check (BEFORE final verdict)")
    print("=" * 70)
    spot_label = winners[0] if winners else "continuation"  # most-promising by construction, else default view
    if spot_label in ("continuation", "reversal") or not winners:
        _spot_check(events, 10, f"individual crash events ({spot_label})")
    else:
        _spot_check(events, 10, "individual crash events (no combo cleared both bars -- showing raw events anyway)")
    n_frozen = sum(1 for e in events if e.trigger_bar_frozen)
    n_frozen_run = sum(1 for e in events if e.frozen_run_after >= 2)
    n_data_gap = sum(1 for e in events if _data_gap_flag(e) is not None)
    print(f"\nAcross ALL {len(events)} events: {n_frozen} have a frozen (O=H=L=C) trigger bar, "
          f"{n_frozen_run} show >=2 more frozen bars immediately after (the stronger circuit-lock tell), "
          f"{n_data_gap} have a likely data-gap early exit (>= {DATA_GAP_EARLY_EXIT_MINUTES}min before 15:15).")

    # ---- STEP 6: honest verdict ----
    print("\n" + "=" * 70)
    print("STEP 6: HONEST VERDICT (pre-registered bars: Sharpe>=0.5 AND win rate>=60%, or gap>=15pp for occurrence)")
    print("=" * 70)
    print(f"{'Combination':<28s} {'Days':>6s} {'Sharpe':>8s} {'WinRate':>9s}  Verdict")
    print(f"{'Individual SHORT (cont.)':<28s} {short_days:>6d} {short_sharpe:>8.2f} {short_win:>8.1%}  {_verdict(short_sharpe, short_win)}")
    print(f"{'Individual LONG (reversal)':<28s} {long_days:>6d} {long_sharpe:>8.2f} {long_win:>8.1%}  {_verdict(long_sharpe, long_win)}")
    occ_verdict = "REAL (gap >= 15pp)" if occurrence_real else "NOT SUPPORTED (gap < 15pp)"
    print(f"{'Sector-impact OCCURRENCE':<28s} {'n/a':>6s} {'n/a':>8s} {gap_pp:>7.1f}pp  {occ_verdict}")
    if trading_records:
        sect_v = _verdict(sector_sharpe, sector_win) if occurrence_real else "NOT EVALUATED (occurrence not real)"
        print(f"{'Sector-impact TRADING':<28s} {sector_days:>6d} {sector_sharpe:>8.2f} {sector_win:>8.1%}  {sect_v}")
    else:
        print(f"{'Sector-impact TRADING':<28s} {'n/a':>6s} {'n/a':>8s} {'n/a':>9s}  NO DATA")

    print(f"\nEvents used: {len(events)} individual crash events across {len(all_days)} distinct trading days "
          f"({len(sector_events)} sector-mapped, contributing {sector_total} sector-follower observations "
          f"and {random_total} random-control observations to STEP 3).")


if __name__ == "__main__":
    main()
