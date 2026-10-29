"""
Compares Upstox vs yfinance 1-minute OHLCV data for the same symbols over
their overlapping time window.

This does NOT assume one source is "correct" and grade the other against
it -- it simply measures how closely the two independent feeds agree
(matching timestamps, close-price deltas, volume deltas) so a human can
judge whether yfinance is a trustworthy stand-in for Upstox while KYC is
pending, or whether the discrepancies are large enough that conclusions
drawn from yfinance data shouldn't be trusted.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from data import fetcher, fetcher_yfinance, instruments

SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK"]

PRICE_DIFF_FLAG_PCT = 0.1
CLOSE_AGREEMENT_PCT = 0.05
MINOR_DISAGREEMENT_PCT = 0.2
MISSING_BARS_FLAG_PCT = 10.0


def compare_symbol(symbol: str, days_back: int = 6) -> None:
    print(f"\n=== {symbol} ===")

    instrument_key = instruments.resolve_instrument_key(symbol)
    if instrument_key is None:
        print(f"{symbol}: could not resolve Upstox instrument_key, skipping")
        return

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back - 1)

    upstox_df = fetcher.fetch_historical_1min(instrument_key, start_date, end_date)
    yf_df = fetcher_yfinance.fetch_recent_1min(symbol)

    if upstox_df.empty or yf_df.empty:
        print(f"{symbol}: one or both sources returned no data "
              f"(upstox={len(upstox_df)} bars, yfinance={len(yf_df)} bars), skipping")
        return

    # Upstox returns tz-aware (+05:30) timestamps; yfinance's are already
    # naive IST-local (see fetcher_yfinance.py). Strip tz so the indexes
    # are directly comparable -- otherwise intersection() silently returns empty.
    if upstox_df.index.tz is not None:
        upstox_df = upstox_df.copy()
        upstox_df.index = upstox_df.index.tz_localize(None)

    upstox_index = upstox_df.index
    yf_index = yf_df.index
    common = upstox_index.intersection(yf_index)
    upstox_only = upstox_index.difference(yf_index)
    yfinance_only = yf_index.difference(upstox_index)

    print(f"Upstox range:   {start_date} .. {end_date} ({len(upstox_df)} bars)")
    print(f"yfinance range: {yf_index.min()} .. {yf_index.max()} ({len(yf_df)} bars)")
    print(f"Timestamps only in Upstox:   {len(upstox_only)}")
    print(f"Timestamps only in yfinance: {len(yfinance_only)}")
    print(f"Overlapping timestamps:      {len(common)}")

    # --- Diagnostic: is this really about today's partial session, or a
    # timestamp alignment issue? Check unique DATES (not full timestamps)
    # covered by each source -- if the same calendar days are covered but
    # full-timestamp overlap is much smaller, that points to a systematic
    # time offset, not missing data.
    upstox_dates = set(upstox_df.index.date)
    yf_dates = set(yf_df.index.date)
    print(f"\nUnique trading days -- Upstox: {sorted(upstox_dates)}")
    print(f"Unique trading days -- yfinance: {sorted(yf_dates)}")
    print(f"Days in both: {len(upstox_dates & yf_dates)}, "
          f"only in Upstox: {sorted(upstox_dates - yf_dates)}, "
          f"only in yfinance: {sorted(yf_dates - upstox_dates)}")

    if len(upstox_only) > 0:
        print(f"First 5 timestamps only in Upstox: {list(upstox_only[:5])}")
    if len(yfinance_only) > 0:
        print(f"First 5 timestamps only in yfinance: {list(yfinance_only[:5])}")

    # Check for a systematic offset: compare the first matching date's
    # timestamps minute-by-minute
    if upstox_dates & yf_dates:
        sample_date = sorted(upstox_dates & yf_dates)[0]
        up_times = sorted(t for t in upstox_df.index if t.date() == sample_date)[:5]
        yf_times = sorted(t for t in yf_df.index if t.date() == sample_date)[:5]
        print(f"\nFirst 5 Upstox timestamps on {sample_date}: {up_times}")
        print(f"First 5 yfinance timestamps on {sample_date}: {yf_times}")

    if len(common) == 0:
        print(f"{symbol}: no overlapping timestamps, cannot compare prices/volume")
        return

    merged = pd.DataFrame({
        "close_upstox": upstox_df.loc[common, "close"],
        "close_yfinance": yf_df.loc[common, "close"],
        "volume_upstox": upstox_df.loc[common, "volume"],
        "volume_yfinance": yf_df.loc[common, "volume"],
    })

    price_diff = (merged["close_upstox"] - merged["close_yfinance"]).abs()
    price_diff_pct = price_diff / merged["close_upstox"].abs() * 100

    mean_price_diff = price_diff.mean()
    max_price_diff = price_diff.max()
    mean_price_diff_pct = price_diff_pct.mean()
    max_price_diff_pct = price_diff_pct.max()
    bars_over_threshold = int((price_diff_pct > PRICE_DIFF_FLAG_PCT).sum())

    print(f"Close price diff: mean=Rs.{mean_price_diff:.4f} ({mean_price_diff_pct:.4f}%), "
          f"max=Rs.{max_price_diff:.4f} ({max_price_diff_pct:.4f}%)")
    print(f"Bars with close diff > {PRICE_DIFF_FLAG_PCT}%: {bars_over_threshold} / {len(common)}")

    volume_valid = merged["volume_upstox"] != 0
    volume_diff_pct = (
        (merged.loc[volume_valid, "volume_upstox"] - merged.loc[volume_valid, "volume_yfinance"]).abs()
        / merged.loc[volume_valid, "volume_upstox"] * 100
    )
    if volume_diff_pct.empty:
        print("Volume diff: no bars with nonzero Upstox volume to compare")
    else:
        print(f"Volume diff %: mean={volume_diff_pct.mean():.2f}%, "
              f"median={volume_diff_pct.median():.2f}%")

    if mean_price_diff_pct < CLOSE_AGREEMENT_PCT:
        verdict = "agrees closely"
    elif mean_price_diff_pct < MINOR_DISAGREEMENT_PCT:
        verdict = "minor disagreement"
    else:
        verdict = "SIGNIFICANT disagreement"
    print(f"Verdict: {verdict}")

    upstox_missing_pct = len(yfinance_only) / len(yf_df) * 100
    yfinance_missing_pct = len(upstox_only) / len(upstox_df) * 100
    if upstox_missing_pct > MISSING_BARS_FLAG_PCT:
        print(f"FLAG: Upstox is missing {upstox_missing_pct:.1f}% of yfinance's bars")
    if yfinance_missing_pct > MISSING_BARS_FLAG_PCT:
        print(f"FLAG: yfinance is missing {yfinance_missing_pct:.1f}% of Upstox's bars")


def main() -> None:
    for symbol in SYMBOLS:
        compare_symbol(symbol)


if __name__ == "__main__":
    main()
