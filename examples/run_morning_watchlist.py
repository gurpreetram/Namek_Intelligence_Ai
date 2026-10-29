"""
Runs Phase A (paper_trading/watchlist.py's morning watchlist build) across
the full cached universe, parallelized via multiprocessing with real
progress visibility (imap_unordered + counter) -- same pattern as
examples/run_universe_backtest.py, not a blocking pool.map() that gives
zero visibility into a multi-minute run.

Each worker handles ONE symbol: computes its ScanResult (confirmed-pick
candidacy, via the exact same single-call fast-scan detectors.scanner.
scan_symbol() daily_picks() uses) AND its near-level candidacy (via
paper_trading.watchlist._check_near_level_for_symbol) in one pass, so a
symbol's data is loaded/resampled once per worker call, not reloaded
across two separate full-universe passes.

Final selection uses paper_trading.watchlist.select_watchlist() -- the
SAME ranking/cap logic build_morning_watchlist()'s single-threaded path
uses, so this parallel run and a small-scale run always make the same
decision given the same per-symbol inputs. No separate/parallel selection
logic is reimplemented here.

Run: python -m examples.run_morning_watchlist
"""
from __future__ import annotations

import multiprocessing
import os
import time

from config.settings import STORAGE_ROOT
from data import storage
from detectors.scanner import ScanResult, scan_symbol
from paper_trading.watchlist import (
    DEFAULT_MAX_WATCHLIST_SIZE,
    DEFAULT_NEAR_LEVEL_ATR_MULT,
    DEFAULT_PICKS_TOP_N,
    WatchlistEntry,
    _check_near_level_for_symbol,
    select_watchlist,
)


def _scan_one(symbol: str) -> tuple[str, ScanResult | None, tuple[float, WatchlistEntry] | None]:
    df = storage.load(symbol)
    if df.empty:
        return symbol, None, None
    as_of = df.index.max()  # each symbol scanned as of its OWN latest bar -- see scanner.py's scan_universe() docstring

    scan_result = None
    try:
        scan_result = scan_symbol(symbol, df, as_of)
    except Exception as e:
        print(f"  {symbol}: scan_symbol error -- {e}")

    near_level = None
    try:
        near_level = _check_near_level_for_symbol(symbol, df, as_of, DEFAULT_NEAR_LEVEL_ATR_MULT)
    except Exception as e:
        print(f"  {symbol}: near-level check error -- {e}")

    return symbol, scan_result, near_level


def build_watchlist_parallel(
    symbols: list[str] | None = None,
    max_size: int = DEFAULT_MAX_WATCHLIST_SIZE,
    picks_top_n: int = DEFAULT_PICKS_TOP_N,
    show_progress: bool = True,
    use_composite_score: bool = True,
) -> list[WatchlistEntry]:
    """
    The reusable core of this script: parallelized Phase A build, callable
    from other code (e.g. examples/run_live_session.py) instead of only
    from this script's own main(), so a caller that needs a fast,
    up-to-date watchlist doesn't have to fall back to
    paper_trading.watchlist.build_morning_watchlist()'s much slower
    single-threaded path just because it isn't running as __main__ here.

    use_composite_score=True (default): after the pool finishes, re-scores
    and re-ranks `confirmed` via detectors.composite_score (see that
    module's docstring for the weights/reasoning) before handing off to
    select_watchlist(). Done here in the MAIN process, not per-worker --
    sector momentum is a universe-wide computation that would be wasteful
    to redo per symbol, and `confirmed` is typically a few tens to low
    hundreds of symbols (however many actually fired), cheap to re-load
    and re-score even though each worker already loaded its own df once.
    """
    if symbols is None:
        symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    if not symbols:
        return []

    cores = os.cpu_count() or 1
    if show_progress:
        print(f"Building Phase A morning watchlist across {len(symbols)} cached symbols, {cores} CPU cores...\n")

    confirmed: list[ScanResult] = []
    near_level: list[tuple[float, WatchlistEntry]] = []
    units_done = 0
    progress_every = 200
    last_bucket = -1
    start_time = time.monotonic()

    # imap_unordered (not map): see run_universe_backtest.py's docstring --
    # map() blocks silently until every task is done, giving zero
    # visibility into a run that could take minutes across 2000+ symbols.
    with multiprocessing.Pool(processes=cores) as pool:
        for symbol, scan_result, nl in pool.imap_unordered(_scan_one, symbols):
            units_done += 1
            if scan_result is not None and scan_result.direction in ("long", "short") and scan_result.entry_price > 0:
                confirmed.append(scan_result)
            if nl is not None:
                near_level.append(nl)

            if show_progress:
                bucket = units_done // progress_every
                if bucket != last_bucket or units_done >= len(symbols):
                    last_bucket = bucket
                    elapsed = time.monotonic() - start_time
                    eta_min = (elapsed / units_done) * (len(symbols) - units_done) / 60
                    print(f"{units_done}/{len(symbols)} done, ETA {eta_min:.1f} min", flush=True)

    if show_progress:
        elapsed = time.monotonic() - start_time
        print(f"\nScan completed in {elapsed:.1f}s ({elapsed / 60:.2f} min)\n")

    if use_composite_score and confirmed:
        if show_progress:
            print(f"Composite-scoring {len(confirmed)} confirmed picks...")
        _apply_composite_ranking_parallel(confirmed, show_progress=show_progress)

    return select_watchlist(confirmed, near_level, max_size, picks_top_n)


def _apply_composite_ranking_parallel(confirmed: list[ScanResult], show_progress: bool = True) -> None:
    """Main-process post-processing: computes composite_score in-place on
    every ScanResult in `confirmed` (re-loading each symbol's cached df --
    workers didn't return theirs, to avoid serializing full DataFrames
    through the Pool for all 2373 symbols just to keep them for the much
    smaller `confirmed` subset), then re-sorts descending. See detectors/
    scanner.py's _apply_composite_ranking() -- same logic, just operating
    on a flat list instead of separate longs/shorts (select_watchlist()
    below does its own direction split and top-N slicing)."""
    import json

    from config.settings import PROJECT_ROOT
    from detectors.composite_score import compute_composite_score
    from detectors.sector_momentum import compute_sector_momentum, pick_reference_as_of

    mapping_path = PROJECT_ROOT / "config" / "sector_mapping.json"
    sector_mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    sector_symbol_data = {s: df for s in sector_mapping if not (df := storage.load(s)).empty}
    sector_as_of = pick_reference_as_of(sector_symbol_data) if sector_symbol_data else None
    sector_results = (
        compute_sector_momentum(sector_mapping, sector_symbol_data, as_of=sector_as_of)
        if sector_as_of is not None else []
    )
    sector_lookup = {r.sector: r for r in sector_results}

    scored = 0
    for r in confirmed:
        df = storage.load(r.symbol)
        if df.empty:
            continue
        cs = compute_composite_score(r, df, df.index.max(), sector_mapping, sector_lookup)
        r.composite_score = cs.total
        r.composite_reasoning = cs.reasoning
        scored += 1

    if show_progress:
        print(f"  Composite-scored {scored}/{len(confirmed)} confirmed picks "
              f"({len(sector_symbol_data)} sector-mapped symbols had usable data for sector_alignment).\n")


def main() -> None:
    watchlist = build_watchlist_parallel()
    if not watchlist:
        print("No cached data found -- run data/bulk_fetch.py first.")
        return

    confirmed_entries = [w for w in watchlist if w.reason == "confirmed_pick"]
    near_level_entries = [w for w in watchlist if w.reason == "near_level"]

    print("=" * 70)
    print("PHASE A MORNING WATCHLIST -- FULL UNIVERSE RESULT")
    print("=" * 70)
    print(f"Final watchlist size: {len(watchlist)}  (cap = {DEFAULT_MAX_WATCHLIST_SIZE})")
    print(f"  Confirmed picks in final watchlist: {len(confirmed_entries)}")
    print(f"  Near-level candidates in final watchlist: {len(near_level_entries)}")

    def _print_side(entries: list[WatchlistEntry]) -> None:
        if not entries:
            print("  (none)")
        for w in entries:
            cs = f"composite={w.composite_score:.3f}" if w.composite_score is not None else "composite=n/a"
            print(f"  {w.symbol:12s} {w.detail:40s} {cs}")
            if w.composite_reasoning:
                for line in w.composite_reasoning:
                    print(f"      - {line}")

    print(f"\n{'-' * 70}\nCONFIRMED PICKS -- BUY (long), ranked by composite score\n{'-' * 70}")
    _print_side([w for w in confirmed_entries if w.direction == "long"])

    print(f"\n{'-' * 70}\nCONFIRMED PICKS -- SELL (short), ranked by composite score\n{'-' * 70}")
    _print_side([w for w in confirmed_entries if w.direction == "short"])

    print(f"\n{'-' * 70}\nNEAR-LEVEL CANDIDATES (closest to level first)\n{'-' * 70}")
    if not near_level_entries:
        print("  (none)")
    for w in near_level_entries:
        print(f"  {w.symbol:12s} [{w.direction:5s}] {w.detail}")


if __name__ == "__main__":
    main()
