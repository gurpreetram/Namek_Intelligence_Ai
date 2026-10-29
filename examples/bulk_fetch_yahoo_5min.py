"""
Bulk-fetches ~60 days of 5-minute NSE data via data/fetcher_yahoo_5min.py
(a free, unauthenticated, RESEARCH-ONLY supplementary source -- see that
module's docstring for the verification behind it) and saves to
storage/yahoo_5min/, deliberately separate from storage/candles/ (the
primary 1-minute Upstox cache).

Default universe: every symbol currently in the primary cache (matches
what api/main.py's /market-breadth endpoint uses), so the resulting
dataset supports a like-for-like "same breadth calculation, more history"
comparison. Pass --limit to restrict to a smaller/faster subset (e.g. for
testing), or --sector-only to fetch just the 181 sector-mapped symbols.

Resume is trivial here (no JSONL checkpoint apparatus needed, unlike the
trade-level backtest runs): each symbol's own output is a complete,
independent parquet file in storage/yahoo_5min/, so "skip if the file
already exists" is a correct, sufficient resume mechanism on its own --
a partially-completed run just picks up where it left off on re-run.

Concurrency: ThreadPoolExecutor, 5 workers -- verified safe empirically
before this script was written (80 real symbols fetched concurrently at
5 workers, 0 errors, ~0.43s/symbol effective). No published rate limit
exists for this endpoint (unlike Upstox, where a real documented limit
was found and used to calibrate the bulk-fetch speedup earlier) -- 5
workers is a deliberately modest choice given that uncertainty, not
pushed toward the fastest rate that happened to work in one small test.

Run: python -m examples.bulk_fetch_yahoo_5min
    python -m examples.bulk_fetch_yahoo_5min --sector-only
    python -m examples.bulk_fetch_yahoo_5min --limit 200
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage_yahoo_5min
from data.fetcher_yahoo_5min import fetch_5min_60d

LOG_PATH = PROJECT_ROOT / "logs" / "yahoo_5min_bulk_fetch.log"
MAX_WORKERS = 5


def _full_cached_universe() -> list[str]:
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _sector_mapped_universe() -> list[str]:
    mapping_path = PROJECT_ROOT / "config" / "sector_mapping.json"
    return sorted(json.loads(mapping_path.read_text()))


def _fetch_one(symbol: str) -> tuple[str, int]:
    df = fetch_5min_60d(symbol)
    if not df.empty:
        storage_yahoo_5min.save(symbol, df)
    return symbol, len(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector-only", action="store_true",
                         help="Fetch only the 181 sector-mapped symbols instead of the full cached universe.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only fetch the first N symbols (alphabetical).")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()], force=True,
    )
    log = logging.getLogger("yahoo_5min_bulk_fetch")

    universe = _sector_mapped_universe() if args.sector_only else _full_cached_universe()
    if args.limit is not None:
        universe = universe[:args.limit]

    already_done = set(storage_yahoo_5min.cached_symbols())
    remaining = [s for s in universe if s not in already_done]
    log.info(f"Universe: {len(universe)} symbols. Already cached: {len(already_done & set(universe))}. "
              f"Remaining: {len(remaining)}.")

    if not remaining:
        log.info("Nothing to do -- all requested symbols already cached.")
        return

    start_time = time.monotonic()
    done = 0
    empty_count = 0
    progress_every = 100

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in remaining}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _, n_bars = future.result()
            except Exception as e:
                log.warning(f"  {symbol}: fetch raised {type(e).__name__}: {e}")
                n_bars = 0
            if n_bars == 0:
                empty_count += 1
            done += 1
            if done % progress_every == 0 or done == len(remaining):
                elapsed = time.monotonic() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta_min = (len(remaining) - done) / rate / 60 if rate > 0 else float("nan")
                log.info(f"{done}/{len(remaining)} done, {empty_count} empty/failed, ETA {eta_min:.1f} min")

    elapsed = time.monotonic() - start_time
    log.info(f"Batch completed in {elapsed:.1f}s ({elapsed / 60:.1f} min). "
              f"{empty_count}/{len(remaining)} symbols returned no data.")
    final_cached = storage_yahoo_5min.cached_symbols()
    log.info(f"Total symbols now cached in storage/yahoo_5min/: {len(final_cached)}")
    log.info("RUN COMPLETE")


if __name__ == "__main__":
    main()
