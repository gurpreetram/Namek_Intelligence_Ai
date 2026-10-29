"""
Daily full-universe rolling-cache refresh: catches every cached symbol's
storage/candles/ (config.settings.STORAGE_ROOT) up to the latest available
data BEFORE market open, so detectors/sector_momentum.py's
pick_reference_as_of() -- a majority-vote of the last cached DATE across
the whole universe, used by /sectors, /watchlist, and /market-breadth to
decide whether "today's" data is ready -- can actually reach today's date
once the market opens.

THE GAP THIS FIXES: paper_trading/live_poller.py already calls
storage.save() every poll cycle, but only for the ~75-symbol live
watchlist, not the ~2,392-symbol full cache. Confirmed empirically
2026-07-17: 2,220 of 2,392 cached symbols were stuck on a stale date
(2026-07-14) even with a fully healthy, actively-polling live session,
because pick_reference_as_of()'s majority vote is dominated by the
~2,300 symbols the live session never touches. This script closes that
gap by refreshing the FULL (price-filtered) universe once daily, not
just the live watchlist.

PRICE FILTER, applied BEFORE fetching (not just at the strategy-signal
level where detectors/scanner.py's MIN_STOCK_PRICE already applied it):
a symbol whose LAST KNOWN CACHED close is below MIN_STOCK_PRICE is
excluded from the refresh fetch list entirely -- see
_filter_min_price()'s docstring for the exact mechanism and its one real
limitation (re-inclusion requires fresh price data, which an excluded
symbol stops receiving -- flagged there, not hidden).

TWO fetch phases per run:
  1. HISTORICAL catch-up: symbols grouped by their own last-cached date
     (most are on the same handful of dates, e.g. everyone stuck on one
     bulk-fetch day), each group fetched via ONE
     fetch_historical_1min_bulk() call spanning just (last_cached_date+1)
     .. yesterday -- catches up any multi-day backlog without re-fetching
     days already covered. Skipped entirely for a symbol already caught
     up through yesterday.
  2. INTRADAY catch-up: one fetch_intraday_1min_bulk() call across the
     whole filtered universe for TODAY's bars. Before market open this
     legitimately returns nothing yet (Upstox has no candles for a day
     that hasn't started) -- that's expected, not a failure, and is why
     STEP 1 asks how long the HISTORICAL phase alone takes when deciding
     a pre-market schedule time; the intraday phase only starts producing
     real bars once trading begins.

Writes to the ROLLING cache (storage.save()), not the permanent archive
(examples/daily_archive.py's storage.save_archive()) -- these are
deliberately separate concerns: this script's only job is making sure
TODAY's freshness check passes; daily_archive.py's job is never losing a
day once Upstox's ~30-day rolling window rolls it out.

RUN DAILY, BEFORE MARKET OPEN (see README's "Running the backend" section
neighbor -- Task Scheduler task "NAMEK Daily Universe Refresh", same
pattern as "NAMEK Live Session Auto-Start"):

    python -m examples.daily_universe_refresh

Idempotent and resumable, same guarantee as data/bulk_fetch.py: safe to
re-run (storage.save() merges/de-dupes by timestamp), and a symbol
already caught up through yesterday costs nothing extra in the
historical phase (its group is simply empty).
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

from config.settings import MARKET_OPEN, PROJECT_ROOT, STORAGE_ROOT, is_trading_day, now_ist
from data import storage
from data.fetcher import fetch_historical_1min_bulk, fetch_intraday_1min_bulk
from data.instruments import resolve_many
from detectors.scanner import MIN_STOCK_PRICE

LOG_PATH_TEMPLATE = str(PROJECT_ROOT / "logs" / "daily_universe_refresh_{date}.log")


def _full_cached_universe() -> list[str]:
    if not STORAGE_ROOT.exists():
        return []
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _filter_min_price(symbols: list[str], log: logging.Logger) -> list[str]:
    """
    Excludes any symbol whose LAST KNOWN CACHED close is below
    MIN_STOCK_PRICE (detectors/scanner.py's own constant, reused here
    rather than duplicated -- see that module for why Rs.100 was picked).
    Recomputed fresh from each symbol's current cache on every run, NOT a
    persisted blacklist file -- a symbol is excluded or included purely
    based on the most recent price this script (or anything else writing
    to storage.save()) has actually observed for it.

    REAL LIMITATION, deliberately not hidden: once a symbol is excluded,
    this script stops fetching it, which means its cached price can never
    change again through this mechanism -- so a stock that legitimately
    rallies back above Rs.100 after being excluded would NOT be
    automatically re-discovered and re-included, even though this
    function's own logic would happily re-include it if it ever saw a
    fresh price >= Rs.100. The exclusion becomes permanent in practice,
    not by explicit design. A stock trading in the Rs.50-150 range near
    the threshold could be silently dropped for good the first time it
    dips below Rs.100, then never resurface even after recovering.
    Mitigation NOT implemented here (out of scope for this task): a
    periodic (e.g. weekly) lightweight one-bar quote recheck of the
    excluded set specifically, cheap enough to run often since it doesn't
    need a full historical fetch, just one current price per symbol.
    """
    kept = []
    excluded = []
    for symbol in symbols:
        df = storage.load(symbol)
        if df.empty:
            kept.append(symbol)  # no price data yet to judge by -- don't exclude blind
            continue
        last_price = float(df["close"].iloc[-1])
        if last_price < MIN_STOCK_PRICE:
            excluded.append(symbol)
        else:
            kept.append(symbol)
    log.info(
        f"Price filter (MIN_STOCK_PRICE={MIN_STOCK_PRICE:.2f}): {len(excluded)} of {len(symbols)} "
        f"cached symbols excluded (last known price below threshold), {len(kept)} remain in the "
        f"refresh universe."
    )
    return kept


def _group_by_last_cached_date(symbols: list[str]) -> dict[date, list[str]]:
    """{last_cached_date: [symbols stuck on that date]} -- lets the
    historical catch-up phase issue ONE fetch_historical_1min_bulk() call
    per distinct stale date instead of one call per symbol, since in
    practice almost every symbol shares one of a handful of dates (e.g.
    everyone stuck on the same prior bulk-fetch day)."""
    groups: dict[date, list[str]] = {}
    for symbol in symbols:
        cov = storage.coverage(symbol)
        last_date = cov[1].date() if cov is not None else date(2000, 1, 1)
        groups.setdefault(last_date, []).append(symbol)
    return groups


def main() -> None:
    now = now_ist()
    today = now.date()

    date_str = today.isoformat()
    log_path = Path(LOG_PATH_TEMPLATE.format(date=date_str))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("daily_universe_refresh")

    if not is_trading_day(today):
        log.info(f"{today} is not an NSE trading day (weekend or holiday) -- nothing to refresh.")
        return

    log.info("=" * 70)
    log.info(f"DAILY UNIVERSE REFRESH START -- {today}")
    log.info("=" * 70)

    universe = _full_cached_universe()
    if not universe:
        log.info("No symbols found in the local cache -- nothing to refresh.")
        return
    log.info(f"Full cached universe: {len(universe)} symbols")

    filtered = _filter_min_price(universe, log)

    t0 = time.monotonic()
    instrument_keys = resolve_many(filtered)
    resolved = {s: k for s, k in instrument_keys.items() if k is not None}
    unresolved_count = len(filtered) - len(resolved)
    if unresolved_count:
        log.info(f"{unresolved_count} symbols had no resolvable instrument key -- skipped.")

    # --- Phase 1: historical catch-up, grouped by each symbol's own
    # last-cached date, so a symbol already current through yesterday
    # costs nothing extra.
    yesterday = today - timedelta(days=1)
    groups = _group_by_last_cached_date(list(resolved.keys()))
    stale_groups = {d: syms for d, syms in groups.items() if d < yesterday}
    already_current = sum(len(syms) for d, syms in groups.items() if d >= yesterday)
    log.info(
        f"Historical catch-up: {already_current} symbols already current through {yesterday}, "
        f"{sum(len(s) for s in stale_groups.values())} symbols across {len(stale_groups)} distinct "
        f"stale dates need catching up."
    )

    hist_fetched_symbols = 0
    hist_total_bars = 0
    hist_t0 = time.monotonic()
    for last_date, syms in sorted(stale_groups.items()):
        from_date = last_date + timedelta(days=1)
        keys_for_group = {s: resolved[s] for s in syms}
        log.info(f"  Fetching {len(syms)} symbols stuck on {last_date} -> catching up {from_date}..{yesterday}...")
        fetched = fetch_historical_1min_bulk(keys_for_group, from_date, yesterday, max_workers=10)
        for symbol, df in fetched.items():
            if df.empty:
                continue
            storage.save(symbol, df)
            hist_fetched_symbols += 1
            hist_total_bars += len(df)
        log.info(f"    -> {len(fetched)}/{len(syms)} symbols returned data ({sum(len(d) for d in fetched.values())} bars)")
    hist_elapsed = time.monotonic() - hist_t0
    log.info(
        f"Historical catch-up done in {hist_elapsed:.1f}s: {hist_fetched_symbols} symbols updated, "
        f"{hist_total_bars} new bars."
    )

    # --- Phase 2: today's intraday bars, one bulk call across the whole
    # filtered universe. Skipped entirely if run before MARKET_OPEN --
    # Upstox genuinely has no candles yet for a day that hasn't started,
    # so every one of ~1600+ requests would round-trip for an empty
    # result (confirmed costs ~340s for nothing, same as when it succeeds
    # -- request count/network latency dominates, not payload size).
    # Skipping this when scheduled pre-market roughly HALVES the
    # steady-state daily run time; a later on-demand/mid-day trigger (like
    # this script's own 2026-07-17 validation run) still gets full
    # intraday coverage since the time check simply passes by then.
    if now.strftime("%H:%M") < MARKET_OPEN:
        log.info(
            f"Intraday catch-up SKIPPED -- current time {now.strftime('%H:%M')} IST is before "
            f"market open ({MARKET_OPEN}); Upstox has no candles for today yet. The live session "
            f"(paper_trading/live_poller.py) will populate today's bars for its own watchlist once "
            f"trading starts, and tomorrow's pre-market run will catch up the rest via the historical "
            f"phase above."
        )
        intraday_fetched: dict = {}
        intraday_bars = 0
        intraday_elapsed = 0.0
    else:
        intraday_t0 = time.monotonic()
        log.info(f"Intraday catch-up: fetching today ({today}) for all {len(resolved)} symbols...")
        intraday_fetched = fetch_intraday_1min_bulk(resolved, max_workers=10)
        intraday_bars = 0
        for symbol, df in intraday_fetched.items():
            if df.empty:
                continue
            storage.save(symbol, df)
            intraday_bars += len(df)
        intraday_elapsed = time.monotonic() - intraday_t0
        log.info(
            f"Intraday catch-up done in {intraday_elapsed:.1f}s: {len(intraday_fetched)}/{len(resolved)} "
            f"symbols returned today's data ({intraday_bars} bars)."
        )

    total_elapsed = time.monotonic() - t0
    log.info("=" * 70)
    log.info(
        f"DAILY UNIVERSE REFRESH DONE in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min) -- "
        f"{len(resolved)} symbols in refresh universe, {hist_fetched_symbols} historical catch-ups, "
        f"{len(intraday_fetched)} intraday catch-ups."
    )
    log.info("=" * 70)


if __name__ == "__main__":
    main()
