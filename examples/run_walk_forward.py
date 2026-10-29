"""
Runnable driver for backtest/walk_forward.py's consistency check -- see
that module's docstring for what this actually tests (out-of-sample
CONSISTENCY, not classic walk-forward optimization) and its current
honest limitation (mechanism test only, until the archive has weeks/
months more history).

Usage:
    python -m examples.run_walk_forward
    python -m examples.run_walk_forward --universe full
    python -m examples.run_walk_forward --start-date 2026-06-08 --end-date 2026-07-10

Defaults to the sector-mapped universe (fast, no network calls -- purely
local backtest computation over already-cached/archived data) and to the
full date range covered by data.storage.load_combined() for a reference
symbol in that universe.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

from backtest.walk_forward import format_report, run_consistency_check
from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage


def _full_cached_universe() -> list[str]:
    if not STORAGE_ROOT.exists():
        return []
    return sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _sector_mapped_universe() -> list[str]:
    mapping_path = PROJECT_ROOT / "config" / "sector_mapping.json"
    if not mapping_path.exists():
        return []
    return sorted(json.loads(mapping_path.read_text()))


def _auto_date_range(symbols: list[str]) -> tuple[date, date]:
    """Full [min, max] date coverage of load_combined() across a handful
    of reference symbols -- not every symbol necessarily shares the exact
    same range, but this project's cache is bulk-fetched in one pass, so
    a small sample is representative."""
    earliest, latest = None, None
    for symbol in symbols[:5]:
        df = storage.load_combined(symbol)
        if df.empty:
            continue
        d_min, d_max = df.index.min().date(), df.index.max().date()
        earliest = d_min if earliest is None else min(earliest, d_min)
        latest = d_max if latest is None else max(latest, d_max)
    if earliest is None or latest is None:
        raise SystemExit("No combined (archive + cache) data found for any sampled symbol -- nothing to test.")
    return earliest, latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", choices=["sectors", "full"], default="sectors")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD, defaults to auto-detected earliest.")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD, defaults to auto-detected latest.")
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--min-reliable-sample", type=int, default=20)
    args = parser.parse_args()

    symbols = _sector_mapped_universe() if args.universe == "sectors" else _full_cached_universe()
    if not symbols:
        raise SystemExit(f"No symbols found for universe='{args.universe}'.")

    if args.start_date and args.end_date:
        start_date, end_date = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    else:
        start_date, end_date = _auto_date_range(symbols)

    print(f"Running consistency check: {len(symbols)} symbols ({args.universe}), "
          f"{start_date} to {end_date}, split_ratio={args.split_ratio}")

    results, mechanism_test_only = run_consistency_check(
        symbols, start_date, end_date,
        split_ratio=args.split_ratio, min_reliable_sample=args.min_reliable_sample,
    )
    print()
    print(format_report(results, mechanism_test_only))


if __name__ == "__main__":
    main()
