"""
Validates detectors/scanner.py's daily_picks() against whatever symbols are
currently cached -- prints the top 5 buy + top 5 sell setups with entry,
stop-loss, display target, score, and reasoning, so results can be
sanity-checked before wiring this into a dashboard tab.

Run: python -m examples.run_daily_picks
    python -m examples.run_daily_picks --limit 400   # bounded batch, faster
"""
from __future__ import annotations

import argparse
import time

from config.settings import STORAGE_ROOT
from data import storage
from detectors.scanner import daily_picks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    if args.limit is not None:
        symbols = symbols[:args.limit]
    if not symbols:
        print("No cached data found -- run data/bulk_fetch.py first.")
        return

    print(f"Loading {len(symbols)} cached symbols...")
    symbol_data = {symbol: storage.load(symbol) for symbol in symbols}
    symbol_data = {s: df for s, df in symbol_data.items() if not df.empty}
    last_dates = sorted({df.index.max().date() for df in symbol_data.values()})
    print(f"Scanning {len(symbol_data)} symbols, each as of its OWN most recent "
          f"cached bar (last-cached dates present in this set: {last_dates})...")

    start = time.monotonic()
    picks = daily_picks(symbol_data, as_of=None, top_n=args.top_n)
    elapsed = time.monotonic() - start
    print(f"Scan completed in {elapsed:.1f}s\n")

    def show(label: str, side: str) -> None:
        print(f"{'=' * 70}\n{label}\n{'=' * 70}")
        rows = picks[side]
        if not rows:
            print("No qualifying setups.")
            return
        for p in rows:
            print(f"\n{p.symbol}  [{p.setup_type}]  score={p.score:.2f}")
            print(f"  entry={p.entry_price:.2f}  stop_loss={p.stop_loss:.2f}  target={p.target_price:.2f}")
            for r in p.reasoning:
                print(f"    - {r}")

    show("TOP BUY (long) PICKS", "buy")
    print()
    show("TOP SELL (short) PICKS", "sell")


if __name__ == "__main__":
    main()
