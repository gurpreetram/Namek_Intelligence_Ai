"""
Bulk-fetches 1-minute historical data for many symbols into the local
Parquet cache (data/storage.py), so the system can be backtested/scanned
across the full NSE equity universe instead of a hardcoded handful of
symbols.

Resumable by design: fetch_universe() checks data/storage.py's existing
coverage for each symbol before hitting the network, so re-running this
after a partial/interrupted run only re-fetches symbols that are missing
or whose cached range doesn't cover what was requested. Rate limiting is
NOT duplicated here -- data/fetcher.py already throttles every HTTP
request via its own _throttle()/retry logic; this module just calls it
once per symbol and lets that existing layer pace the requests.

CLI usage:
    python -m data.bulk_fetch                # default: small 50-symbol test batch
    python -m data.bulk_fetch --limit 50      # same, explicit
    python -m data.bulk_fetch --limit all     # full NSE_EQ universe (~2200 symbols)
    python -m data.bulk_fetch --limit 2200    # explicit large number also works
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

from data import fetcher, instruments, storage

DEFAULT_TEST_BATCH_SIZE = 50


@dataclass
class BulkFetchResult:
    succeeded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (symbol, error message)
    total_bars_fetched: int = 0
    elapsed_seconds: float = 0.0


def _has_coverage(symbol: str, start_date: date, end_date: date) -> bool:
    existing = storage.coverage(symbol)
    if existing is None:
        return False
    earliest, latest = existing
    return earliest.date() <= start_date and latest.date() >= end_date


def fetch_universe(symbols: list[str], start_date: date, end_date: date,
                    skip_if_cached: bool = True) -> BulkFetchResult:
    """
    Fetches and caches 1-minute historical data for every symbol in
    `symbols`, one at a time. data/fetcher.py's own throttle/retry logic
    already paces the underlying HTTP requests, so no second throttle
    layer is added here -- this loop is just sequential, not parallel.

    A single symbol's failure (typo, delisted instrument, API error) is
    caught and recorded rather than aborting the whole batch, since one
    bad symbol out of thousands shouldn't kill an otherwise-successful run.
    """
    result = BulkFetchResult()
    total = len(symbols)
    start_time = time.monotonic()

    for i, symbol in enumerate(symbols, start=1):
        done_so_far = i - 1
        if done_so_far > 0:
            avg_per_symbol = (time.monotonic() - start_time) / done_so_far
            eta_minutes = avg_per_symbol * (total - done_so_far) / 60
            eta_str = f", ETA {eta_minutes:.1f}min"
        else:
            eta_str = ""

        print(f"Fetching {i}/{total}: {symbol}...", end=" ", flush=True)

        if skip_if_cached and _has_coverage(symbol, start_date, end_date):
            print(f"already cached{eta_str}")
            result.skipped.append(symbol)
            continue

        try:
            key = instruments.resolve_instrument_key(symbol)
            if key is None:
                raise ValueError("could not resolve instrument_key (typo, delisted, or not NSE_EQ)")
            df = fetcher.fetch_historical_1min(key, start_date, end_date)
            if df.empty:
                raise ValueError("Upstox returned no candles for this range")
            storage.save(symbol, df)
            result.succeeded.append(symbol)
            result.total_bars_fetched += len(df)
            print(f"done ({len(df)} bars){eta_str}")
        except Exception as e:
            result.failed.append((symbol, str(e)))
            print(f"FAILED ({e}){eta_str}")

    result.elapsed_seconds = time.monotonic() - start_time
    return result


def _parse_limit(raw: str | None) -> int | None:
    """None (flag omitted) -> small test batch. 'all' -> no limit. Otherwise an int."""
    if raw is None:
        return DEFAULT_TEST_BATCH_SIZE
    if raw.lower() == "all":
        return None
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-fetch 1-min historical data for the NSE equity universe."
    )
    parser.add_argument(
        "--limit", type=str, default=None,
        help=f"Number of symbols to fetch, or 'all' for the full NSE_EQ universe. "
             f"Defaults to {DEFAULT_TEST_BATCH_SIZE} (a small test batch) if omitted -- "
             f"pass --limit all (or an explicit large number) to run the full universe.",
    )
    args = parser.parse_args()
    limit = _parse_limit(args.limit)

    all_symbols = instruments.list_all_nse_equity_symbols()
    symbols = all_symbols if limit is None else all_symbols[:limit]

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)  # Upstox's actual 1-min data limit

    print(f"Universe: {len(all_symbols)} total NSE_EQ symbols, fetching {len(symbols)}")
    print(f"Date range: {start_date} .. {end_date}\n")

    result = fetch_universe(symbols, start_date, end_date)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Succeeded: {len(result.succeeded)}")
    print(f"Skipped (already cached): {len(result.skipped)}")
    print(f"Failed: {len(result.failed)}")
    if result.failed:
        print("\nFailures:")
        for symbol, err in result.failed:
            print(f"  {symbol}: {err}")
    print(f"\nTotal bars fetched: {result.total_bars_fetched}")
    print(f"Total time: {result.elapsed_seconds / 60:.1f} minutes")


if __name__ == "__main__":
    main()
