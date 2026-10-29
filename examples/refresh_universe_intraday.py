"""
One-off (re-runnable) full-universe refresh of TODAY's intraday bars,
using the same live intraday endpoint + persist pattern paper_trading/
live_poller.py already uses per-symbol for its watchlist (fetch_intraday_
1min_bulk() + storage.save()), just scaled to the full cached universe
instead of a 75-symbol watchlist.

Why this exists: data/bulk_fetch.py deliberately fetches only THROUGH
YESTERDAY (end_date = date.today() - timedelta(days=1), via Upstox's
historical-candle endpoint) -- it backfills the rolling 30-day window for
backtesting and structurally cannot bring in today's session. A live
session running (paper_trading/live_poller.py) only refreshes its own
watchlist (~75 symbols) -- nowhere near enough for /market-breadth or
/sectors, which need the full cached universe / all 180 sector-mapped
symbols respectively to have TODAY's data. This script closes that gap
directly, on demand, without waiting for a scheduled mechanism.

Concurrent (ThreadPoolExecutor via fetch_intraday_1min_bulk(), same
BULK_FETCH_MIN_INTERVAL_SECONDS-throttled rate limiter /sector-momentum-
picks already uses for its smaller 180-symbol version of this same
pattern) -- much faster than data/bulk_fetch.py's sequential historical
fetch, since this is a single intraday snapshot per symbol, not a 30-day
backfill.

Run: python -m examples.refresh_universe_intraday
"""
from __future__ import annotations

import time

from config.settings import STORAGE_ROOT
from data import storage
from data.fetcher import fetch_intraday_1min_bulk
from data.instruments import resolve_many

MAX_WORKERS = 10


def main() -> None:
    symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    print(f"Refreshing today's intraday bars for {len(symbols)} cached symbols...\n")

    instrument_keys = resolve_many(symbols)
    resolved = {s: k for s, k in instrument_keys.items() if k is not None}
    unresolved = len(symbols) - len(resolved)
    if unresolved:
        print(f"  {unresolved} symbols have no instrument_key, skipping those.")

    start = time.monotonic()
    results = fetch_intraday_1min_bulk(resolved, max_workers=MAX_WORKERS)
    elapsed = time.monotonic() - start

    total_bars = 0
    for symbol, df in results.items():
        storage.save(symbol, df)
        total_bars += len(df)

    empty_or_failed = len(resolved) - len(results)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Succeeded: {len(results)}")
    print(f"Empty/failed: {empty_or_failed}")
    print(f"Total bars fetched: {total_bars}")
    print(f"Total time: {elapsed / 60:.1f} minutes ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
