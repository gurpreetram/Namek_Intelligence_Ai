"""
Standalone evaluation of the "openchart" library (pip install openchart,
MIT license, wraps NSE's own public charting API) as a potential
supplementary/alternative data source to Upstox -- specifically whether
it offers MORE than Upstox's ~30-day 1-minute history limit.

WHY THIS SCRIPT EXISTS SEPARATELY: the same evaluation attempted from the
sandboxed dev environment failed at the very first step -- NSE's own
Akamai CDN returned a hard "403 Access Denied" on the session-cookie-
establishing request to www.nseindia.com, reproducible 3/3 times, which
cascades into broken symbol search (returns a generic index list
regardless of query) and empty historical data for every symbol tried,
including a known-valid token. That looks like Akamai's bot-detection
blocking the sandbox's specific network origin, not a flaw in the
library itself -- this script exists to re-run the identical evaluation
from a real local machine/network to get a genuine answer.

This does NOT modify data/fetcher.py or any part of the main pipeline --
evaluation only, per explicit instruction. Nothing here is wired into
BacktestEngine, storage.py, or any live path.

STEP 1: import/init sanity check
STEP 2: search sanity check (does it actually resolve RELIANCE/TCS/etc.
    to real equity scripcodes, or the broken generic-index-list behavior
    seen in the sandbox?)
STEP 3: 1-minute historical depth test -- try progressively longer
    lookback windows (30/60/90/180/365 days) and report the ACTUAL number
    of days/bars returned at each, and where it starts failing/truncating
STEP 4: data quality comparison against this project's existing Upstox
    pipeline, for whatever date range actually overlaps -- same
    methodology as data/compare_sources.py (mean/max close-price diff %,
    timestamp coverage gaps), not a new invented methodology
STEP 5: universe coverage -- how many of a sample of this project's
    already-cached symbols does openchart's search successfully resolve

Run (from the project root, with its venv active):
    python -m examples.test_openchart
"""
from __future__ import annotations

import random
import time
from datetime import date, datetime, timedelta

import pandas as pd

try:
    from openchart import NSEData
except ImportError:
    raise SystemExit("openchart is not installed -- run: pip install openchart")

from config.settings import STORAGE_ROOT
from data import fetcher, instruments

DEPTH_TEST_DAYS = [30, 60, 90, 180, 365]
SEARCH_TEST_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"]
UNIVERSE_SAMPLE_SIZE = 30
PRICE_DIFF_FLAG_PCT = 0.1
CLOSE_AGREEMENT_PCT = 0.05
MINOR_DISAGREEMENT_PCT = 0.2


def step1_init() -> "NSEData":
    print("=" * 70)
    print("STEP 1: Import / initialization")
    print("=" * 70)
    nse = NSEData()
    print("NSEData() initialized OK\n")
    return nse


def step2_search(nse: "NSEData") -> None:
    print("=" * 70)
    print("STEP 2: Symbol search sanity check")
    print("=" * 70)
    print("(In the sandboxed environment, this returned the SAME generic")
    print(" 20-item NIFTY-index list regardless of query, because NSE's")
    print(" Akamai CDN blocked the session-cookie-establishing request.")
    print(" If that's fixed here, each symbol below should return real")
    print(" equity results with type='Equity', not a repeated index list.)\n")
    for sym in SEARCH_TEST_SYMBOLS:
        try:
            results = nse.search(sym, segment="EQ")
        except Exception as e:
            print(f"{sym}: search raised {type(e).__name__}: {e}")
            continue
        if results.empty:
            print(f"{sym}: search returned EMPTY")
            continue
        types = sorted(set(results["type"]))
        exact = results[results["symbol"].str.upper() == sym.upper()]
        print(f"{sym}: {len(results)} results, types={types}, "
              f"exact match found={not exact.empty}")
        if not exact.empty:
            print(f"    -> {exact.iloc[0].to_dict()}")
    print()


def step3_depth_test(nse: "NSEData", symbol: str = "RELIANCE") -> pd.DataFrame | None:
    print("=" * 70)
    print(f"STEP 3: 1-minute historical depth test ({symbol})")
    print("=" * 70)
    best_df = None
    for days_back in DEPTH_TEST_DAYS:
        end = datetime.now()
        start = end - timedelta(days=days_back)
        t0 = time.time()
        try:
            df = nse.historical(symbol, segment="EQ", start=start, end=end, interval="1m")
        except Exception as e:
            print(f"  requested {days_back:>4d} days back -> ERROR: {type(e).__name__}: {e}")
            continue
        elapsed = time.time() - t0
        if df.empty:
            print(f"  requested {days_back:>4d} days back -> EMPTY (0 rows) in {elapsed:.1f}s")
            continue
        actual_days = (df.index.max() - df.index.min()).days
        distinct_dates = len(set(df.index.date))
        print(f"  requested {days_back:>4d} days back -> {len(df):>6d} bars, "
              f"{distinct_dates} distinct trading days, "
              f"actual range {df.index.min()} to {df.index.max()} "
              f"({actual_days} calendar days) [{elapsed:.1f}s]")
        best_df = df
    print()
    if best_df is None:
        print("No successful fetch at any depth -- cannot proceed to data-quality comparison.\n")
    return best_df


def step4_compare_vs_upstox(openchart_df: pd.DataFrame, symbol: str = "RELIANCE") -> None:
    print("=" * 70)
    print(f"STEP 4: Data quality vs Upstox ({symbol}) -- same methodology as data/compare_sources.py")
    print("=" * 70)

    instrument_key = instruments.resolve_instrument_key(symbol)
    if instrument_key is None:
        print(f"Could not resolve Upstox instrument_key for {symbol} -- skipping comparison.\n")
        return

    oc_dates = sorted(set(openchart_df.index.date))
    # Upstox only has ~30 days of 1-min history -- only request the
    # overlapping tail of openchart's range, not its full depth.
    upstox_start = max(oc_dates[0], date.today() - timedelta(days=29))
    upstox_end = min(oc_dates[-1], date.today() - timedelta(days=1))
    if upstox_start > upstox_end:
        print("No overlapping date range with Upstox's ~30-day window -- skipping comparison.\n")
        return

    upstox_df = fetcher.fetch_historical_1min(instrument_key, upstox_start, upstox_end)
    if upstox_df.empty:
        print(f"Upstox returned no data for {upstox_start}..{upstox_end} -- skipping comparison.\n")
        return

    oc_df = openchart_df.copy()
    if oc_df.index.tz is not None:
        oc_df.index = oc_df.index.tz_localize(None)
    up_df = upstox_df.copy()
    if up_df.index.tz is not None:
        up_df.index = up_df.index.tz_localize(None)

    common = up_df.index.intersection(oc_df.index)
    upstox_only = up_df.index.difference(oc_df.index)
    openchart_only = oc_df.index.difference(up_df.index)

    print(f"Upstox range:    {upstox_start} .. {upstox_end} ({len(up_df)} bars)")
    print(f"openchart range (same window): {len(oc_df.loc[str(upstox_start):str(upstox_end)])} bars")
    print(f"Timestamps only in Upstox:    {len(upstox_only)}")
    print(f"Timestamps only in openchart: {len(openchart_only)}")
    print(f"Overlapping timestamps:       {len(common)}")

    if len(common) == 0:
        print("No overlapping timestamps -- cannot compare prices.\n")
        return

    merged = pd.DataFrame({
        "close_upstox": up_df.loc[common, "close"],
        "close_openchart": oc_df.loc[common, "close"],
    })
    price_diff = (merged["close_upstox"] - merged["close_openchart"]).abs()
    price_diff_pct = price_diff / merged["close_upstox"].abs() * 100
    mean_diff_pct = price_diff_pct.mean()
    max_diff_pct = price_diff_pct.max()
    bars_over_threshold = int((price_diff_pct > PRICE_DIFF_FLAG_PCT).sum())

    print(f"Close price diff: mean={mean_diff_pct:.4f}%, max={max_diff_pct:.4f}%")
    print(f"Bars with diff > {PRICE_DIFF_FLAG_PCT}%: {bars_over_threshold} / {len(common)}")

    if mean_diff_pct < CLOSE_AGREEMENT_PCT:
        verdict = "agrees closely"
    elif mean_diff_pct < MINOR_DISAGREEMENT_PCT:
        verdict = "minor disagreement"
    else:
        verdict = "SIGNIFICANT disagreement"
    print(f"Verdict: {verdict}\n")


def step5_universe_coverage(nse: "NSEData") -> None:
    print("=" * 70)
    print(f"STEP 5: Universe coverage (sample of {UNIVERSE_SAMPLE_SIZE} cached symbols)")
    print("=" * 70)

    if not STORAGE_ROOT.exists():
        print("No cached universe found (storage/candles/ missing) -- skipping.\n")
        return
    all_symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    if not all_symbols:
        print("No cached symbols found -- skipping.\n")
        return

    random.seed(42)  # reproducible sample
    sample = random.sample(all_symbols, min(UNIVERSE_SAMPLE_SIZE, len(all_symbols)))

    resolved, unresolved = [], []
    for sym in sample:
        try:
            results = nse.search(sym, segment="EQ")
        except Exception:
            unresolved.append(sym)
            continue
        if results.empty:
            unresolved.append(sym)
            continue
        exact = results[
            (results["symbol"].str.upper() == sym.upper()) & (results["type"] == "Equity")
        ]
        if exact.empty:
            unresolved.append(sym)
        else:
            resolved.append(sym)

    print(f"Resolved: {len(resolved)}/{len(sample)}")
    print(f"Unresolved: {len(unresolved)}/{len(sample)}")
    if unresolved:
        print(f"  Unresolved symbols: {unresolved}")
    print(f"\nFull cached universe size: {len(all_symbols)} symbols "
          f"(this sample is {len(sample)}/{len(all_symbols)} = "
          f"{len(sample) / len(all_symbols) * 100:.1f}% of it)\n")


def main() -> None:
    nse = step1_init()
    step2_search(nse)
    best_df = step3_depth_test(nse, "RELIANCE")
    if best_df is not None:
        step4_compare_vs_upstox(best_df, "RELIANCE")
    step5_universe_coverage(nse)

    print("=" * 70)
    print("DONE -- paste this full output back for the honest verdict.")
    print("=" * 70)


if __name__ == "__main__":
    main()
