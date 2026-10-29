"""
Daily archiving: pulls one completed trading day's 1-minute candles across
the cached universe and appends them to storage/archive/ -- a permanent,
never-pruned store, distinct from storage/candles/ (config.settings.
STORAGE_ROOT), which only ever holds Upstox's own rolling ~30-day intraday
window. Without this, the gap found during the 2026-07-10 backfill work
repeats forever: live-fetched/day-old data eventually rolls out of
Upstox's window and is lost for good, since it was never written to a
permanent cache in the first place.

RUN ONCE DAILY, BY HAND, AFTER MARKET CLOSE (after 16:00 IST, once the
day's data is complete and Upstox's own intraday endpoint has settled):

    python -m examples.daily_archive

Idempotent -- safe to re-run on a day already archived (data/storage.py's
save_archive() merges and de-duplicates by timestamp index), and safe to
run for a past date explicitly via --date if you need to catch up on a
missed day:

    python -m examples.daily_archive --date 2026-07-10

Universe: defaults to the full cached universe (every symbol with a file
in storage/candles/), fetched CONCURRENTLY via data/fetcher.py's
fetch_historical_1min_bulk() (thread pool, the same verified-safe Upstox
rate limit as that module's other bulk fetch -- see fetcher.py's
BULK_FETCH_MIN_INTERVAL_SECONDS docstring for the sourced numbers), which
keeps a full-universe run to a couple of minutes rather than the ~30+
minutes a naive one-symbol-at-a-time loop would take. Pass
--universe sectors to restrict to the smaller sector-mapped universe
(config/sector_mapping.json) instead, if the full universe ever proves
too slow to run daily in practice.

Windows Task Scheduler automation is a natural next step once this has
been run manually a few times to confirm it behaves correctly -- not
wired up yet; run it by hand for now (see the command above).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, timedelta

from config.settings import PROJECT_ROOT, STORAGE_ROOT, is_trading_day, now_ist
from data import storage
from data.fetcher import fetch_historical_1min_bulk
from data.instruments import resolve_many

LOG_PATH = PROJECT_ROOT / "logs" / "daily_archive.log"


def _default_target_date() -> date:
    """
    The most recent PRIOR trading day -- never today, regardless of what
    time this runs. Upstox's historical-candle endpoint has a confirmed
    T+1 settlement pattern: same-day fetches failed 100% of the time
    across 6 attempts spanning 30-95 minutes after close, while fetches
    for the prior trading day succeeded at 99%+ every time. So there's no
    "late enough in the day" that makes today fetchable -- always step
    back at least one day, then further back over weekends/NSE holidays
    until landing on a real trading day.
    """
    d = now_ist().date() - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def _full_cached_universe() -> list[str]:
    if not STORAGE_ROOT.exists():
        return []
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _sector_mapped_universe() -> list[str]:
    mapping_path = PROJECT_ROOT / "config" / "sector_mapping.json"
    if not mapping_path.exists():
        return []
    return sorted(json.loads(mapping_path.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=str, default=None,
                         help="Target date to archive, YYYY-MM-DD. Defaults to the most recent completed trading day.")
    parser.add_argument("--universe", choices=["full", "sectors"], default="full",
                         help="'full' = every symbol in storage/candles/ (default). "
                              "'sectors' = the smaller sector-mapped universe, if full-universe proves too slow.")
    parser.add_argument("--max-workers", type=int, default=10, help="Concurrent fetch threads (default 10).")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("daily_archive")

    target_date = date.fromisoformat(args.date) if args.date else _default_target_date()
    if not is_trading_day(target_date):
        log.info(f"{target_date} is not an NSE trading day (weekend or holiday) -- nothing to archive.")
        return

    universe = _full_cached_universe() if args.universe == "full" else _sector_mapped_universe()
    if not universe:
        log.info(f"No symbols found for universe='{args.universe}' -- nothing to archive.")
        return

    log.info(f"Archiving {target_date} for {len(universe)} symbols "
              f"(universe={args.universe}, max_workers={args.max_workers})...")

    t0 = time.monotonic()
    instrument_keys = resolve_many(universe)
    resolved = {s: k for s, k in instrument_keys.items() if k is not None}
    unresolved_count = len(universe) - len(resolved)
    if unresolved_count:
        log.info(f"{unresolved_count} symbols had no resolvable instrument key -- skipped.")

    fetched = fetch_historical_1min_bulk(resolved, target_date, target_date, max_workers=args.max_workers)
    fetch_elapsed = time.monotonic() - t0

    archived_symbols = 0
    total_new_bars = 0
    for symbol, df in fetched.items():
        before = storage.load_archive(symbol)
        new_bars = len(df.index.difference(before.index))
        storage.save_archive(symbol, df)
        total_new_bars += new_bars
        archived_symbols += 1

    total_elapsed = time.monotonic() - t0
    log.info(
        f"Archived {target_date}: {archived_symbols}/{len(resolved)} symbols fetched successfully "
        f"({len(resolved) - archived_symbols} failed/empty/no data), {total_new_bars} new bars added "
        f"to the archive (fetch took {fetch_elapsed:.1f}s, total {total_elapsed:.1f}s)."
    )


if __name__ == "__main__":
    main()
