"""
Tests two opposing hypotheses about overnight gaps: does a stock's
overnight gap (today's open vs yesterday's close) tend to FILL (reverse
back toward yesterday's close) or CONTINUE (extend further in the gap's
direction) by end of day?

PRE-REGISTERED BAR, set BEFORE running this and not adjusted after seeing
results: a (strategy, gap-size) combination counts as a real, worth-
considering edge only if BOTH hold on its day-level return series:
  - day-level Sharpe (non-annualized) >= 0.5 in the hypothesized direction
  - day-level win rate >= 60%
Meeting only one of these is suggestive, not confirmed. Meeting neither is
noise. Applied separately to each of the 4 combinations tested (fill x
moderate, fill x large, continuation x moderate, continuation x large).

GAP DEFINITION: (today's first cached bar's open - yesterday's last
cached bar's close) / yesterday's close * 100, using each symbol's OWN
previous cached trading day (not a fixed calendar date) -- handles any
gaps in a given symbol's cache the same way every other day-grouping in
this project does (e.g. detectors/cross_sectional_ranking.py). Only
symbols priced >= detectors.scanner.MIN_STOCK_PRICE (judged off
yesterday's close) qualify -- same reasoning as every other scan in this
project for excluding penny-stock noise. Moderate gap: 1% <= |gap| < 2%.
Large gap: |gap| >= 2%. Segmented separately since the fill/continuation
tendency may plausibly differ by size.

ENTRY: at today's very first cached bar's OWN open. This is deliberately
different from detectors/cross_sectional_ranking.py's "next bar after
cutoff" convention -- that convention exists there because the ranking
signal depends on a formation WINDOW extending past the entry decision
point (no lookahead requires waiting for the next bar). The gap signal
here is fully known at the instant the first bar prints (yesterday's
close is already history; today's open IS this bar's own open) -- so
entering at this bar's own open uses no information not yet available.
EXIT: EOD square-off at or before 15:15, same convention as every other
strategy in this project (detectors/cross_sectional_ranking.py's
bar_at_or_before(), reused as-is).

Two opposing bets tested on the SAME gapped population:
  - gap-fill: bets the gap reverses -- SHORT a gap-up, LONG a gap-down
  - gap-continuation: bets the gap extends -- LONG a gap-up, SHORT a gap-down

CONCENTRATION TEST (--top-n): same rigor applied to the earlier mean-
reversion cross-sectional finding (examples/run_cross_sectional_backtest.py)
-- does gap-fill survive being concentrated down to a practical few-
trades-a-day style, trading only the TOP-N largest |gap%| events each day
(gap-fill direction only, mixing moderate/large rather than the fixed
magnitude buckets above), instead of every qualifying gap? Reports BOTH
gross and net of an estimated round-trip transaction cost (--cost-pct,
default COST_PCT_PER_ROUND_TRIP -- same 0.075% assumption and reasoning
as run_cross_sectional_backtest.py's, for direct comparability).

Run: python -m examples.run_gap_fill_vs_continuation
    python -m examples.run_gap_fill_vs_continuation --limit 500   # fast test
    python -m examples.run_gap_fill_vs_continuation --top-n 5     # concentration test, gap-fill only
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

EOD_SQUARE_OFF_TIME = "15:15"
MODERATE_GAP_MIN = 1.0
LARGE_GAP_MIN = 2.0
MIN_SHARPE_BAR = 0.5
MIN_WINRATE_BAR = 0.60
# Same assumption/reasoning as examples/run_cross_sectional_backtest.py's
# COST_PCT_PER_ROUND_TRIP -- brokerage + STT + exchange/GST + slippage
# combined, as a % of notional, applied once per position regardless of
# direction. 0.075% is the midpoint of the commonly-cited 0.05-0.1% range
# for Indian retail intraday equity.
COST_PCT_PER_ROUND_TRIP = 0.075


@dataclass
class GapEvent:
    symbol: str
    day: date
    gap_pct: float
    entry_price: float
    entry_time: pd.Timestamp
    exit_price: float
    exit_time: pd.Timestamp
    avg_traded_value: float  # yesterday's, from analyze_liquidity() -- see _find_gap_events()
    entry_bar_frozen: bool  # entry bar's open==high==low==close -- possible circuit-lock tell
    entry_bar_volume: float

    @property
    def magnitude_bucket(self) -> str:
        return "large" if abs(self.gap_pct) >= LARGE_GAP_MIN else "moderate"


def _find_gap_events(symbol_data: dict[str, pd.DataFrame], require_liquid: bool = True) -> list[GapEvent]:
    """
    require_liquid=True (default): applies detectors.liquidity_analysis's
    analyze_liquidity() to YESTERDAY's data (the last 30 bars before
    today's gap) before accepting an event -- the same no-lookahead
    principle as everywhere else (only information already known before
    the gap is used to judge tradeability). A symbol with insufficient
    history to judge (None) is EXCLUDED, not assumed liquid -- deliberately
    more conservative than backtest/engine.py's live-strategy gate (which
    treats None as "proceed," since there a missing reading is a data-
    availability gap, not evidence of risk); here, for a retrospective
    research pass with no time pressure to decide, excluding an unverifiable
    symbol costs nothing and avoids exactly the kind of thin-stock false
    positive that produced the real BANG (+180R) and AFFLE (+46R) cases
    documented in backtest/engine.py's own module docstring. Set False only
    to reproduce the ORIGINAL (unfiltered) result for direct before/after
    comparison.
    """
    events: list[GapEvent] = []
    for symbol, df in symbol_data.items():
        days = sorted(set(df.index.date))
        for i in range(1, len(days)):
            prev_day, today = days[i - 1], days[i]
            prev_day_df = df[df.index.date == prev_day]
            today_df = df[df.index.date == today]
            if prev_day_df.empty or today_df.empty:
                continue

            yesterday_close = float(prev_day_df["close"].iloc[-1])
            if yesterday_close < MIN_STOCK_PRICE:
                continue

            if require_liquid:
                liquidity = analyze_liquidity(prev_day_df)
                if liquidity is None or not liquidity.is_liquid:
                    continue
                avg_traded_value = liquidity.avg_traded_value
            else:
                probe = analyze_liquidity(prev_day_df)
                avg_traded_value = probe.avg_traded_value if probe is not None else float("nan")

            entry_bar = today_df.iloc[0]
            today_open = float(entry_bar["open"])
            gap_pct = (today_open - yesterday_close) / yesterday_close * 100.0
            if abs(gap_pct) < MODERATE_GAP_MIN:
                continue

            eod_ts = pd.Timestamp.combine(today, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(entry_bar.name.tz)
            exit_bar = bar_at_or_before(df, eod_ts)
            if exit_bar is None or exit_bar.name <= entry_bar.name:
                continue

            entry_frozen = (
                float(entry_bar["open"]) == float(entry_bar["high"]) == float(entry_bar["low"]) == float(entry_bar["close"])
            )

            events.append(GapEvent(
                symbol=symbol, day=today, gap_pct=gap_pct,
                entry_price=today_open, entry_time=entry_bar.name,
                exit_price=float(exit_bar["close"]), exit_time=exit_bar.name,
                avg_traded_value=avg_traded_value,
                entry_bar_frozen=entry_frozen, entry_bar_volume=float(entry_bar["volume"]),
            ))
    return events


def _day_stats(daily_returns: np.ndarray) -> tuple[float, float, float, float]:
    if len(daily_returns) == 0:
        return 0.0, 0.0, 0.0, 0.0
    win_rate = float((daily_returns > 0).mean())
    mean = float(daily_returns.mean())
    std = float(daily_returns.std())
    sharpe = mean / std if std > 0 else 0.0
    return win_rate, mean, std, sharpe


# NSE price-band circuit percentages commonly applied to individual
# equities (2%, 5%, 10%, 20% -- the exact band is per-symbol and not
# published in this project's cached OHLCV data, so this is a heuristic
# proxy, not a definitive circuit-hit determination). A gap landing within
# this tolerance of one of these AND showing a frozen entry bar is flagged
# as a possible circuit-limit artifact worth manually excluding, not
# silently trusted as a real, tradeable gap-and-fill move.
CIRCUIT_BAND_PCTS = (2.0, 5.0, 10.0, 20.0)
CIRCUIT_BAND_TOLERANCE = 0.3


def _circuit_flag(event: "GapEvent") -> str:
    near_band = any(abs(abs(event.gap_pct) - band) <= CIRCUIT_BAND_TOLERANCE for band in CIRCUIT_BAND_PCTS)
    if event.entry_bar_frozen and near_band:
        return "LIKELY CIRCUIT LOCK (frozen entry bar + gap near a standard circuit band)"
    if event.entry_bar_frozen:
        return "entry bar frozen (O=H=L=C) -- worth checking, not necessarily a circuit hit"
    if near_band:
        return "gap near a standard circuit band, but entry bar NOT frozen -- likely genuine"
    return "no circuit-lock indicators"


def _spot_check_top_n(all_events: list[GapEvent], top_n: int, n_days: int, direction: str = "fill") -> None:
    """Prints full real detail (symbol, gap%, liquidity, entry/exit
    prices, the direction-specific position return, circuit-lock
    heuristic) for the top-N largest |gap%| events on the first n_days
    distinct trading days found -- so real examples can be manually
    eyeballed, not just aggregate statistics trusted at face value."""
    by_day: dict[date, list[GapEvent]] = {}
    for e in all_events:
        by_day.setdefault(e.day, []).append(e)

    days_to_show = sorted(by_day.keys())[:n_days]
    print(f"\n--- SPOT-CHECK ({direction}): top-{top_n} pick(s) on {len(days_to_show)} real days ---")
    for day in days_to_show:
        top = sorted(by_day[day], key=lambda e: abs(e.gap_pct), reverse=True)[:top_n]
        print(f"\n{day}:")
        for e in top:
            sign = _direction_sign(e, direction)
            pos_ret = sign * (e.exit_price - e.entry_price) / e.entry_price * 100.0
            print(f"  {e.symbol:14s} gap={e.gap_pct:+.2f}%  avg_traded_value=Rs.{e.avg_traded_value:,.0f}/min  "
                  f"entry={e.entry_price:.2f} ({e.entry_time.strftime('%H:%M')})  "
                  f"exit={e.exit_price:.2f} ({e.exit_time.strftime('%H:%M')})  "
                  f"entry_bar_volume={e.entry_bar_volume:,.0f}  {direction}_return={pos_ret:+.3f}%")
            print(f"    {_circuit_flag(e)}")


def _report_combination(label: str, events: list[GapEvent], sign_fn) -> None:
    """sign_fn(event) -> +1.0 (LONG) or -1.0 (SHORT), the direction this
    strategy takes on this event given its gap direction."""
    if not events:
        print(f"\n--- {label} ---\nNo qualifying events.")
        return

    pos_returns = []
    by_day: dict[date, list[float]] = {}
    for e in events:
        sign = sign_fn(e)
        ret_pct = sign * (e.exit_price - e.entry_price) / e.entry_price * 100.0
        pos_returns.append(ret_pct)
        by_day.setdefault(e.day, []).append(ret_pct)

    pos_returns = np.array(pos_returns)
    day_returns = np.array([float(np.mean(v)) for v in by_day.values()])
    win_rate, mean, std, sharpe = _day_stats(day_returns)
    pos_win_rate = float((pos_returns > 0).mean())

    verdict = (
        "REAL EDGE (meets both pre-registered bars)"
        if abs(sharpe) >= MIN_SHARPE_BAR and win_rate >= MIN_WINRATE_BAR and sharpe > 0
        else "suggestive only (meets one bar)"
        if (abs(sharpe) >= MIN_SHARPE_BAR and sharpe > 0) or win_rate >= MIN_WINRATE_BAR
        else "NOISE (meets neither pre-registered bar)"
    )

    print(f"\n--- {label} ---")
    print(f"Events: {len(events)}  |  Days used: {len(by_day)}")
    print(f"Position-level win rate: {pos_win_rate:.1%}  avg return: {pos_returns.mean():+.3f}%  median: {np.median(pos_returns):+.3f}%")
    print(f"Day-level win rate: {win_rate:.1%}  avg daily return: {mean:+.3f}%  std: {std:.3f}%  Sharpe: {sharpe:.2f}")
    print(f"PRE-REGISTERED BAR (Sharpe>=0.5 AND win rate>=60%): {verdict}")


def _direction_sign(e: "GapEvent", direction: str) -> float:
    if direction == "fill":
        return -1.0 if e.gap_pct > 0 else 1.0  # SHORT a gap-up, LONG a gap-down
    return 1.0 if e.gap_pct > 0 else -1.0  # continuation: LONG a gap-up, SHORT a gap-down


def _report_top_n_concentration(all_events: list[GapEvent], top_n: int, cost_pct: float, direction: str = "fill") -> None:
    """CONCENTRATION TEST -- restricted to the TOP-N largest |gap%| events
    each day (mixing moderate/large rather than the fixed magnitude
    buckets), reported both gross and net of an estimated round-trip
    cost. direction: "fill" (SHORT a gap-up, LONG a gap-down) or
    "continuation" (the opposite). Same rigor as
    run_cross_sectional_backtest.py's --top-n test applied to the
    mean-reversion cross-sectional finding."""
    by_day: dict[date, list[GapEvent]] = {}
    for e in all_events:
        by_day.setdefault(e.day, []).append(e)

    day_gross: dict[date, list[float]] = {}
    day_net: dict[date, list[float]] = {}
    pos_gross_all: list[float] = []
    pos_net_all: list[float] = []

    for day, day_events in by_day.items():
        top = sorted(day_events, key=lambda e: abs(e.gap_pct), reverse=True)[:top_n]
        for e in top:
            sign = _direction_sign(e, direction)
            gross = sign * (e.exit_price - e.entry_price) / e.entry_price * 100.0
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

    avg_positions_per_day = pos_gross_all.size / len(by_day) if by_day else 0.0

    def _verdict(win_rate: float, sharpe: float) -> str:
        if sharpe >= MIN_SHARPE_BAR and win_rate >= MIN_WINRATE_BAR:
            return "REAL EDGE (meets both pre-registered bars)"
        if sharpe >= MIN_SHARPE_BAR or win_rate >= MIN_WINRATE_BAR:
            return "suggestive only (meets one bar)"
        return "NOISE (meets neither pre-registered bar)"

    print(f"\n--- Top-{top_n} gap-{direction}, avg {avg_positions_per_day:.1f} positions/day, cost={cost_pct:.3f}%/position ---")
    print(f"Days used: {len(by_day)}  |  Total positions: {pos_gross_all.size}")
    print(f"Position-level -- GROSS win rate: {(pos_gross_all > 0).mean():.1%}  avg: {pos_gross_all.mean():+.3f}%   "
          f"NET win rate: {(pos_net_all > 0).mean():.1%}  avg: {pos_net_all.mean():+.3f}%")
    print(f"Day-level GROSS -- win rate: {g_win:.1%}  avg daily return: {g_mean:+.3f}%  std: {g_std:.3f}%  Sharpe: {g_sharpe:.2f}   [{_verdict(g_win, g_sharpe)}]")
    print(f"Day-level NET   -- win rate: {n_win:.1%}  avg daily return: {n_mean:+.3f}%  std: {n_std:.3f}%  Sharpe: {n_sharpe:.2f}   [{_verdict(n_win, n_sharpe)}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="cap universe size, for a fast test run")
    parser.add_argument("--top-n", type=int, nargs="+", default=None,
                         help="CONCENTRATION TEST: restrict to the top-N largest |gap%%| events per day, "
                              "gap-fill direction only, reported gross and net of --cost-pct. Accepts multiple "
                              "values (e.g. --top-n 1 3 5 10), computing gap events ONCE and reporting each N "
                              "-- re-scanning the full universe per N would be pure waste, same events either "
                              "way. Omit for the default full moderate/large-bucket gap-fill vs "
                              "gap-continuation report.")
    parser.add_argument("--cost-pct", type=float, default=COST_PCT_PER_ROUND_TRIP,
                         help=f"Estimated round-trip transaction cost as %% of notional (default "
                              f"{COST_PCT_PER_ROUND_TRIP}%%) -- only used by --top-n.")
    parser.add_argument("--no-liquidity-filter", action="store_true",
                         help="Reproduce the ORIGINAL (unfiltered) behavior for direct before/after "
                              "comparison -- does NOT apply analyze_liquidity() to yesterday's data before "
                              "accepting a gap event. Default is filtered (liquid symbols only).")
    parser.add_argument("--spot-check", type=int, default=None, metavar="N_DAYS",
                         help="Print full real detail (symbol, gap%%, avg traded value, entry/exit prices, "
                              "circuit-lock heuristic) for the top-N pick(s) on the first N_DAYS distinct "
                              "days found -- requires --top-n with exactly one value.")
    parser.add_argument("--direction", choices=["fill", "continuation"], default="fill",
                         help="Only used by --top-n: 'fill' (default, SHORT a gap-up/LONG a gap-down) or "
                              "'continuation' (LONG a gap-up/SHORT a gap-down).")
    args = parser.parse_args()

    symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    if args.limit is not None:
        symbols = symbols[:args.limit]

    print(f"Loading {len(symbols)} cached symbols into memory...")
    start = time.monotonic()
    symbol_data: dict[str, pd.DataFrame] = {}
    for i, s in enumerate(symbols, 1):
        df = storage.load(s)
        if not df.empty:
            symbol_data[s] = df
        if i % 500 == 0:
            print(f"  loaded {i}/{len(symbols)}...", flush=True)
    print(f"Loaded {len(symbol_data)}/{len(symbols)} symbols with usable data in {time.monotonic() - start:.1f}s\n")

    require_liquid = not args.no_liquidity_filter
    print(f"Scanning for overnight gap events (>=1% in either direction), "
          f"liquidity filter {'ON' if require_liquid else 'OFF'}...")
    t0 = time.monotonic()
    events = _find_gap_events(symbol_data, require_liquid=require_liquid)
    print(f"Found {len(events)} qualifying gap events in {time.monotonic() - t0:.1f}s\n")

    moderate = [e for e in events if e.magnitude_bucket == "moderate"]
    large = [e for e in events if e.magnitude_bucket == "large"]
    all_days = sorted({e.day for e in events})
    print(f"Moderate gaps (1-2%): {len(moderate)} events   Large gaps (>=2%): {len(large)} events")
    print(f"Distinct trading days with at least one qualifying gap: {len(all_days)}")
    print("=" * 70)
    print("SAMPLE-SIZE CAVEAT: results below come from Upstox's ~29-33 day rolling "
          "1-minute cache. This is a SMALL sample -- treat all findings as preliminary.")
    print("=" * 70)

    if args.top_n is not None:
        for n in args.top_n:
            print("\n" + "=" * 70)
            print(f"CONCENTRATION TEST: gap-{args.direction} only, top-{n} largest |gap%| per day")
            print("=" * 70)
            _report_top_n_concentration(events, n, args.cost_pct, direction=args.direction)
        if args.spot_check is not None:
            if len(args.top_n) != 1:
                print("\n--spot-check requires exactly one --top-n value -- skipping.")
            else:
                _spot_check_top_n(events, args.top_n[0], args.spot_check, direction=args.direction)
        # Falls through (no early return) to also print the moderate/large
        # bucket fill-vs-continuation comparison below -- one scan serves
        # both STEP 2/4 (concentration + spot-check) and STEP 3 (bucket
        # comparison), avoiding redundant full-universe rescans.

    def fill_sign(e: GapEvent) -> float:
        return -1.0 if e.gap_pct > 0 else 1.0  # SHORT a gap-up, LONG a gap-down

    def continuation_sign(e: GapEvent) -> float:
        return 1.0 if e.gap_pct > 0 else -1.0  # LONG a gap-up, SHORT a gap-down

    print("\n" + "=" * 70)
    print("GAP-FILL HYPOTHESIS (bets the gap reverses)")
    print("=" * 70)
    _report_combination("Gap-fill, MODERATE gaps (1-2%)", moderate, fill_sign)
    _report_combination("Gap-fill, LARGE gaps (>=2%)", large, fill_sign)

    print("\n" + "=" * 70)
    print("GAP-CONTINUATION HYPOTHESIS (bets the gap extends)")
    print("=" * 70)
    _report_combination("Gap-continuation, MODERATE gaps (1-2%)", moderate, continuation_sign)
    _report_combination("Gap-continuation, LARGE gaps (>=2%)", large, continuation_sign)


if __name__ == "__main__":
    main()
