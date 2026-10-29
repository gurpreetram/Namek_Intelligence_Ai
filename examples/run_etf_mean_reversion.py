"""
Tests cross-sectional mean-reversion's underlying idea at the INDEX/ETF
level instead of individual stocks: examples/run_cross_sectional_backtest.py
found the mean-reversion edge (--reverse: short morning winners/long
morning losers) lived in DIVERSIFICATION -- a 20-stock basket showed a
real Sharpe, but collapsed once concentrated to --top-n 1 (see
logs/cross_sectional_top1.log). Since a broad-market or sector ETF is
ALREADY a diversified basket by construction, this asks whether trading
ONE such instrument, based on ITS OWN morning move, can capture a
similar effect in a single trade/day -- a genuinely different mechanism
from "pick 1-2 individual stocks," not a rerun of the same test.

PRE-REGISTERED BARS (same as every other hypothesis test in this
project): a (ETF-or-combined, threshold-type, direction) combination is
a real edge only if BOTH hold on its day-level return series: day-level
Sharpe (non-annualized) >= 0.5 AND day-level win rate >= 60%.

STEP 1 -- ETF universe: candidates are every cached symbol whose ticker
ends in "BEES" or "ETF" (the two unambiguous Indian ETF-naming
conventions -- tickers like NIFTY1/BANKNIFTY1/ENIFTY/QNIFTY look like
index-futures/synthetic feeds, not real deliverable ETF units, and are
deliberately excluded rather than guessed about). Each candidate is
liquidity-checked via detectors.liquidity_analysis.analyze_liquidity()
on its most recent cached day -- reused, not assumed just because "it's
an ETF." From the genuinely liquid set, this script further restricts
to BROAD-MARKET or SECTOR EQUITY index trackers (matching the user's
literal NIFTYBEES/BANKBEES/JUNIORBEES framing) -- gold/silver/gilt/
liquid-fund ETFs are excluded even if liquid, since a commodity or debt
instrument's morning move is a different hypothesis (commodity price
action, not equity cross-sectional reversion), not what was asked for.
This exclusion list is reported explicitly, not silently applied.

STEP 2 -- signal, same 09:15-10:00 formation window as
detectors.cross_sectional_ranking.rank_symbols_for_day()'s default
(reusing detectors.sector_momentum.compute_stock_return_today() exactly
as that module does, not reimplemented), TWO threshold styles tested
separately:
  FIXED: |morning move| >= 1.0% (either direction).
  ROLLING: a z-score against the ETF's OWN trailing
    ROLLING_LOOKBACK_DAYS morning moves (mean/std of the prior N
    same-window moves for this SAME ETF, prior days only -- no
    lookahead), flagged if |z| >= ROLLING_ZSCORE_THRESHOLD. This is a
    z-score, not a literal percentile-rank -- the user's brief said
    "rolling-percentile...based on recent volatility" as an example
    framing; a z-score against a short trailing window is this
    project's existing standard way of expressing "more than usual"
    (detectors.breakout_validator.volume_zscore is the same idea applied
    elsewhere) and is more robust on a ~20-30 day cache than trying to
    rank against a percentile computed from too few points. Needs
    ROLLING_LOOKBACK_DAYS prior days before a symbol-day is even
    eligible, which further shrinks an already-small sample -- flagged
    explicitly in the final report, not glossed over.
Both REVERSAL (fade the morning move) and CONTINUATION (follow it) are
tested on every qualifying day, for completeness.

LIQUIDITY FILTER APPLIED FROM THE START: analyze_liquidity() on the
PRIOR day's data gates every symbol-day before a signal is even
computed -- same discipline, same order, as every prior script in this
project, even though ETFs are expected to pass it almost every time.

ENTRY: next 1-min bar's open strictly after the 10:00 cutoff (reuses
detectors.cross_sectional_ranking.next_bar_after() as-is). EXIT: EOD
square-off at or before 15:15 (reuses bar_at_or_before() as-is).

STEP 4 -- costs: this is already inherently one trade/day/ETF (no
concentration test needed, per the task) -- gross vs net of the standard
0.075% round-trip cost (COST_PCT_PER_ROUND_TRIP, same assumption/
reasoning as every other script) is reported directly.

Run: python -m examples.run_etf_mean_reversion
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from config.settings import STORAGE_ROOT
from data import storage
from detectors.cross_sectional_ranking import bar_at_or_before, next_bar_after
from detectors.liquidity_analysis import analyze_liquidity
from detectors.sector_momentum import compute_stock_return_today

FORMATION_CUTOFF = "10:00"
EOD_SQUARE_OFF_TIME = "15:15"
FIXED_THRESHOLD_PCT = 1.0
ROLLING_LOOKBACK_DAYS = 10
ROLLING_ZSCORE_THRESHOLD = 1.0
MIN_SHARPE_BAR = 0.5
MIN_WINRATE_BAR = 0.60
COST_PCT_PER_ROUND_TRIP = 0.075
FROZEN_RUN_CHECK_BARS = 10
DATA_GAP_EARLY_EXIT_MINUTES = 30

# Broad-market/sector EQUITY index trackers only -- excludes gold/silver/
# gilt/liquid-fund ETFs even if they pass the liquidity filter (see
# module docstring). Populated from the liquidity screen at runtime;
# this is the exclusion RULE (by underlying asset class, via name
# pattern), not a hardcoded inclusion list, so any liquid equity-index
# ETF in the cache is picked up automatically.
_NON_EQUITY_PATTERNS = ("GOLD", "SILVER", "GILT", "LIQUID", "GSEC")


@dataclass
class ETFEvent:
    symbol: str
    day: date
    morning_move_pct: float
    threshold_type: str  # "fixed" | "rolling"
    zscore: float | None
    avg_traded_value: float
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    trigger_bar_frozen: bool
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


def _data_gap_flag(e: ETFEvent) -> str | None:
    eod_minutes = 15 * 60 + 15
    exit_minutes = e.exit_time.hour * 60 + e.exit_time.minute
    if eod_minutes - exit_minutes >= DATA_GAP_EARLY_EXIT_MINUTES:
        return (f"POSSIBLE DATA GAP: exit found at {e.exit_time.strftime('%H:%M')}, "
                f"{eod_minutes - exit_minutes} min before the 15:15 EOD target -- treat with caution")
    return None


def _find_etf_candidates() -> list[str]:
    return sorted(
        p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet")
        if p.stem.replace("_1min", "").upper().endswith(("BEES", "ETF"))
    )


def _screen_liquidity(candidates: list[str]) -> dict[str, dict]:
    """{symbol: {"df": df, "is_liquid": bool, "avg_traded_value": float}}
    -- liquidity judged off the symbol's own MOST RECENT cached day, a
    universe-screening check (Step 1), separate from the per-day PRIOR-
    day liquidity gate applied during the actual backtest (Steps 2-3)."""
    screened: dict[str, dict] = {}
    for symbol in candidates:
        df = storage.load(symbol)
        if df.empty:
            continue
        last_day = sorted(set(df.index.date))[-1]
        last_day_df = df[df.index.date == last_day]
        reading = analyze_liquidity(last_day_df)
        screened[symbol] = {
            "df": df,
            "is_liquid": reading is not None and reading.is_liquid,
            "avg_traded_value": reading.avg_traded_value if reading is not None else float("nan"),
        }
    return screened


def _find_events(symbol: str, df: pd.DataFrame) -> list[ETFEvent]:
    events: list[ETFEvent] = []
    days = sorted(set(df.index.date))
    trailing_moves: list[float] = []

    for i in range(1, len(days)):
        day = days[i]
        prev_day_df = df[df.index.date == days[i - 1]]
        if prev_day_df.empty:
            continue
        liquidity = analyze_liquidity(prev_day_df)
        day_df = df[df.index.date == day]
        if day_df.empty:
            continue
        cutoff_ts = pd.Timestamp.combine(day, pd.Timestamp(FORMATION_CUTOFF).time()).tz_localize(day_df.index.tz)
        move = compute_stock_return_today(df, cutoff_ts)

        eligible_day = liquidity is not None and liquidity.is_liquid and move is not None
        if eligible_day:
            triggered_types: list[tuple[str, float | None]] = []
            if abs(move) >= FIXED_THRESHOLD_PCT:
                triggered_types.append(("fixed", None))
            if len(trailing_moves) >= ROLLING_LOOKBACK_DAYS:
                window = np.array(trailing_moves[-ROLLING_LOOKBACK_DAYS:])
                std = window.std()
                if std > 0:
                    z = (move - window.mean()) / std
                    if abs(z) >= ROLLING_ZSCORE_THRESHOLD:
                        triggered_types.append(("rolling", float(z)))

            if triggered_types:
                entry_bar = next_bar_after(df, cutoff_ts)
                eod_ts = pd.Timestamp.combine(day, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(day_df.index.tz)
                exit_bar = bar_at_or_before(df, eod_ts)
                if entry_bar is not None and exit_bar is not None and entry_bar.name < exit_bar.name:
                    trigger_bar = day_df.loc[day_df.index <= cutoff_ts].iloc[-1]
                    for threshold_type, z in triggered_types:
                        events.append(ETFEvent(
                            symbol=symbol, day=day, morning_move_pct=move, threshold_type=threshold_type,
                            zscore=z, avg_traded_value=liquidity.avg_traded_value,
                            entry_time=entry_bar.name, entry_price=float(entry_bar["open"]),
                            exit_time=exit_bar.name, exit_price=float(exit_bar["close"]),
                            trigger_bar_frozen=_frozen(trigger_bar),
                            frozen_run_after=_frozen_run_after(day_df, trigger_bar.name, FROZEN_RUN_CHECK_BARS),
                        ))

        if move is not None:
            trailing_moves.append(move)

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


def _report(label: str, events: list[ETFEvent], direction: str, cost_pct: float) -> tuple[float, float, int]:
    """direction: 'reversal' (fade the morning move) or 'continuation' (follow it)."""
    if not events:
        print(f"\n--- {label} ---\nNo qualifying events.")
        return 0.0, 0.0, 0
    gross_returns, net_returns = [], []
    by_day_gross: dict[date, list[float]] = {}
    by_day_net: dict[date, list[float]] = {}
    for e in events:
        move_sign = 1.0 if e.morning_move_pct > 0 else -1.0
        # reversal: opposite of the move's own direction; continuation: same direction.
        position_sign = -move_sign if direction == "reversal" else move_sign
        gross = position_sign * (e.exit_price - e.entry_price) / e.entry_price * 100.0
        net = gross - cost_pct
        gross_returns.append(gross)
        net_returns.append(net)
        by_day_gross.setdefault(e.day, []).append(gross)
        by_day_net.setdefault(e.day, []).append(net)

    gross_returns, net_returns = np.array(gross_returns), np.array(net_returns)
    day_gross = np.array([float(np.mean(v)) for v in by_day_gross.values()])
    day_net = np.array([float(np.mean(v)) for v in by_day_net.values()])
    g_win, g_mean, g_std, g_sharpe = _day_stats(day_gross)
    n_win, n_mean, n_std, n_sharpe = _day_stats(day_net)

    print(f"\n--- {label} ---")
    print(f"Events: {len(events)}  |  Days used: {len(by_day_gross)}")
    print(f"Position-level -- GROSS win rate: {(gross_returns > 0).mean():.1%}  avg: {gross_returns.mean():+.3f}%   "
          f"NET win rate: {(net_returns > 0).mean():.1%}  avg: {net_returns.mean():+.3f}%")
    print(f"Day-level GROSS -- win rate: {g_win:.1%}  avg daily return: {g_mean:+.3f}%  std: {g_std:.3f}%  Sharpe: {g_sharpe:.2f}   [{_verdict(g_sharpe, g_win)}]")
    print(f"Day-level NET   -- win rate: {n_win:.1%}  avg daily return: {n_mean:+.3f}%  std: {n_std:.3f}%  Sharpe: {n_sharpe:.2f}   [{_verdict(n_sharpe, n_win)}]")
    return g_sharpe, g_win, len(by_day_gross)


def _spot_check(events: list[ETFEvent], n: int, label: str) -> None:
    print(f"\n--- SPOT-CHECK ({label}): {min(n, len(events))} real events, largest |morning move| first ---")
    for e in sorted(events, key=lambda e: abs(e.morning_move_pct), reverse=True)[:n]:
        print(f"  {e.symbol:12s} {e.day}  morning_move={e.morning_move_pct:+.2f}%  "
              f"({e.threshold_type}{f', z={e.zscore:+.2f}' if e.zscore is not None else ''})  "
              f"liquidity=Rs.{e.avg_traded_value:,.0f}/min")
        print(f"    entry={e.entry_price:.2f} ({e.entry_time.strftime('%H:%M')})  "
              f"exit={e.exit_price:.2f} ({e.exit_time.strftime('%H:%M')})")
        if e.trigger_bar_frozen:
            print(f"    WARNING: formation-cutoff bar frozen (O=H=L=C) -- worth checking")
        elif e.frozen_run_after >= 2:
            print(f"    WARNING: {e.frozen_run_after} frozen bars immediately after formation cutoff")
        else:
            print(f"    no circuit-lock indicators")
        gap_flag = _data_gap_flag(e)
        if gap_flag:
            print(f"    {gap_flag}")


def main() -> None:
    print("=" * 70)
    print("STEP 1: ETF universe -- candidates, liquidity screen, equity-index filter")
    print("=" * 70)
    candidates = _find_etf_candidates()
    print(f"Candidate tickers (name ends in BEES/ETF): {len(candidates)}")

    t0 = time.monotonic()
    screened = _screen_liquidity(candidates)
    print(f"Screened {len(screened)} candidates with usable cached data in {time.monotonic() - t0:.1f}s\n")

    liquid = {s: v for s, v in screened.items() if v["is_liquid"]}
    print(f"Liquid ETFs (analyze_liquidity() on most recent cached day): {len(liquid)}")
    for s, v in sorted(liquid.items(), key=lambda kv: -kv[1]["avg_traded_value"]):
        print(f"  {s:14s} avg_traded_value=Rs.{v['avg_traded_value']:,.0f}/min")

    non_equity = {s: v for s, v in liquid.items() if any(p in s.upper() for p in _NON_EQUITY_PATTERNS)}
    equity_liquid = {s: v for s, v in liquid.items() if s not in non_equity}
    print(f"\nExcluded as non-equity-index trackers (gold/silver/gilt/liquid-fund, even though liquid): "
          f"{sorted(non_equity.keys())}")
    print(f"\nFINAL TEST UNIVERSE (liquid, broad-market/sector EQUITY index ETFs): {sorted(equity_liquid.keys())}")

    if not equity_liquid:
        print("\nNo liquid equity-index ETF found -- nothing further to test.")
        return

    print("\n" + "=" * 70)
    print(f"STEP 2/3: signal detection + backtest (formation cutoff {FORMATION_CUTOFF}, EOD {EOD_SQUARE_OFF_TIME})")
    print("=" * 70)
    all_events: dict[str, list[ETFEvent]] = {}
    for symbol, v in equity_liquid.items():
        events = _find_events(symbol, v["df"])
        all_events[symbol] = events
        n_fixed = sum(1 for e in events if e.threshold_type == "fixed")
        n_rolling = sum(1 for e in events if e.threshold_type == "rolling")
        print(f"  {symbol:14s} {len(events)} events (fixed={n_fixed}, rolling={n_rolling})")

    results: dict[str, tuple[float, float, int]] = {}
    combo_events: dict[str, list[ETFEvent]] = {}

    for threshold_type in ("fixed", "rolling"):
        for direction in ("reversal", "continuation"):
            for symbol, events in all_events.items():
                subset = [e for e in events if e.threshold_type == threshold_type]
                label = f"{symbol} | {threshold_type} | {direction}"
                s, w, d = _report(label, subset, direction, COST_PCT_PER_ROUND_TRIP)
                results[label] = (s, w, d)
                combo_events[label] = subset

            combined = [e for events in all_events.values() for e in events if e.threshold_type == threshold_type]
            label = f"COMBINED (all ETFs) | {threshold_type} | {direction}"
            s, w, d = _report(label, combined, direction, COST_PCT_PER_ROUND_TRIP)
            results[label] = (s, w, d)
            combo_events[label] = combined

    print("\n" + "=" * 70)
    print("STEP 5: spot-check (BEFORE final verdict)")
    print("=" * 70)
    # Spot-check EVERY combo clearing both bars, not just the single
    # highest-Sharpe one -- the highest Sharpe here (NIFTYBEES fixed
    # continuation, n=3) is exactly the kind of number a "pick the best"
    # selection would misleadingly highlight as most-promising when it's
    # actually the least credible (smallest sample). Sample size (days
    # used) is printed alongside each spot-check for this reason.
    winners = [label for label, (s, w, _) in results.items() if s >= MIN_SHARPE_BAR and w >= MIN_WINRATE_BAR]
    if not winners:
        print("\nNo combination cleared both pre-registered bars -- nothing to spot-check as a 'winner'; "
              "showing the overall combined fixed-threshold set for transparency instead.")
        fallback = combo_events.get("COMBINED (all ETFs) | fixed | reversal", [])
        if fallback:
            _spot_check(fallback, min(8, len(fallback)), "COMBINED (all ETFs) | fixed | reversal")
    else:
        for label in sorted(winners, key=lambda l: -results[l][2]):  # largest sample first
            days_used = results[label][2]
            print(f"\n[{label}] -- {days_used} days used (sample-size credibility noted before trusting this)")
            _spot_check(combo_events[label], min(10, len(combo_events[label])), label)

    print("\n" + "=" * 70)
    print("STEP 6: HONEST VERDICT (pre-registered bars: Sharpe>=0.5 AND win rate>=60%, GROSS)")
    print("=" * 70)
    print(f"{'Combination':<52s} {'Days':>6s} {'Sharpe':>8s} {'WinRate':>9s}  Verdict")
    for label, (s, w, d) in results.items():
        print(f"{label:<52s} {d:>6d} {s:>8.2f} {w:>8.1%}  {_verdict(s, w)}")

    n_total_days = len({e.day for events in all_events.values() for e in events})
    print(f"\nTest universe: {sorted(equity_liquid.keys())} ({len(equity_liquid)} liquid equity-index ETFs)")
    print(f"Rolling z-score variant needs >= {ROLLING_LOOKBACK_DAYS} prior trading days per ETF before it's "
          f"even eligible -- shrinks an already-small ~29-33 day cache further.")
    print("SAMPLE-SIZE CAVEAT: results above come from Upstox's ~29-33 day rolling 1-minute cache, "
          "and only a handful of ETFs. This is a SMALL sample -- treat all findings as preliminary.")


if __name__ == "__main__":
    main()
