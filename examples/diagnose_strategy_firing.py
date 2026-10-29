"""
Read-only diagnostic: mirrors detectors/scanner.py's scan_symbol()
(live_mode=True, the exact mechanism daily_picks()/paper_trading/
watchlist.py use for Daily Picks' "Confirmed" section) across the full
cached universe, but tallies RAW fired signals AND post-verification
survivors PER STRATEGY -- scan_symbol() itself only returns the single
BEST-scoring candidate per symbol, which hides how many of the other 6
strategies fired-but-lost or fired-but-got-rejected on that same symbol.

Does not modify or replace scan_symbol()/daily_picks() -- reuses
_fresh_strategies(), _apply_verification(), analyze_liquidity(),
MIN_STOCK_PRICE, and multi_timeframe_view() exactly as scan_symbol()
does, evaluating each of the 7 strategies once per liquid, price-
qualifying symbol at its own latest cached bar.

Run: python -m examples.diagnose_strategy_firing
    python -m examples.diagnose_strategy_firing --limit 500
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import pandas as pd

from config.settings import STORAGE_ROOT
from data import storage
from data.resample import multi_timeframe_view
from detectors.liquidity_analysis import analyze_liquidity
from detectors.pick_verification import VERIFIED_STRATEGIES
from detectors.scanner import MIN_STOCK_PRICE, _apply_verification, _fresh_strategies
from backtest.strategy_base import SignalDirection

RECENT_DAYS_FOR_MTF = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    if args.limit is not None:
        symbols = symbols[:args.limit]

    print(f"Loading {len(symbols)} cached symbols into memory...")
    t0 = time.monotonic()
    symbol_data: dict[str, pd.DataFrame] = {}
    for i, s in enumerate(symbols, 1):
        df = storage.load(s)
        if not df.empty:
            symbol_data[s] = df
        if i % 500 == 0:
            print(f"  loaded {i}/{len(symbols)}...", flush=True)
    print(f"Loaded {len(symbol_data)}/{len(symbols)} symbols with usable data in {time.monotonic() - t0:.1f}s\n")

    raw_fired = Counter()
    verified_survived = Counter()
    verified_rejected = Counter()
    scanned = 0
    below_price = 0
    illiquid = 0
    not_enough_bars = 0
    errored = Counter()

    t0 = time.monotonic()
    for symbol, df in symbol_data.items():
        as_of = df.index.max()
        today = as_of.date()
        truncated = df.loc[:as_of]
        today_bar_count = int((truncated.index.date == today).sum())
        if today_bar_count < 25:
            not_enough_bars += 1
            continue

        current_price = float(truncated["close"].iloc[-1])
        if current_price < MIN_STOCK_PRICE:
            below_price += 1
            continue

        liquidity = analyze_liquidity(truncated, min_avg_traded_value=500_000.0)
        if liquidity is not None and not liquidity.is_liquid:
            illiquid += 1
            continue

        scanned += 1
        recent_dates = sorted({d for d in truncated.index.date})[-RECENT_DAYS_FOR_MTF:]
        bounded = truncated[truncated.index.date >= recent_dates[0]]
        try:
            mtf_view = multi_timeframe_view(bounded, [1, 5, 15])
        except Exception as e:
            errored["__resample__"] += 1
            continue
        current_time = truncated.index[-1]

        strategies = _fresh_strategies()
        fired: dict[str, "Signal"] = {}
        for strategy in strategies:
            try:
                signal = strategy.evaluate(symbol, mtf_view, current_time)
            except Exception as e:
                errored[strategy.name] += 1
                continue
            if signal is not None and signal.direction != SignalDirection.NONE:
                fired[strategy.name] = signal
                raw_fired[strategy.name] += 1

        if not fired:
            continue

        today_df = truncated[truncated.index.date == today]
        from detectors.scanner import ScanResult
        for name, signal in fired.items():
            candidate = ScanResult(symbol, signal.confidence, name, signal.direction.value,
                                    list(signal.reasoning), entry_price=current_price,
                                    stop_loss=signal.stop_loss, target=signal.target)
            if name not in VERIFIED_STRATEGIES:
                verified_survived[name] += 1  # not subject to verification -- fires directly
                continue
            updated, note = _apply_verification(candidate, signal, today_df, current_time)
            if updated is None:
                verified_rejected[name] += 1
            else:
                verified_survived[name] += 1

    elapsed = time.monotonic() - t0
    print(f"Scanned {scanned} symbols in {elapsed:.1f}s "
          f"(excluded: {below_price} below min price, {illiquid} illiquid, {not_enough_bars} not enough bars today)\n")

    ALL_STRATEGIES = ["opening_range_breakout", "validated_breakout", "trend_continuation",
                       "reversal_detection", "vwap_reclaim", "smart_money", "fibonacci_retracement"]

    print("=" * 100)
    print(f"{'Strategy':<26s} {'Verified?':<10s} {'Raw fired':>10s} {'Verified OK':>12s} {'Rejected':>10s}  Note")
    print("=" * 100)
    for name in ALL_STRATEGIES:
        is_verified = name in VERIFIED_STRATEGIES
        raw = raw_fired[name]
        ok = verified_survived[name]
        rej = verified_rejected[name]
        note = ""
        if name in ("validated_breakout", "reversal_detection"):
            note = "2-step state machine -- single cold call can only ever set pending state, never confirm (see scan_symbol() docstring)"
        elif not is_verified:
            note = "fires directly, no verification gate"
        print(f"{name:<26s} {'yes' if is_verified else 'no':<10s} {raw:>10d} {ok:>12d} {rej:>10d}  {note}")

    if errored:
        print(f"\nEvaluation errors (exceptions caught, not counted as fires): {dict(errored)}")


if __name__ == "__main__":
    main()
