"""
Backtests the new pre-entry verification layer (1.0% min-move + volume
z-score >=1.0 + 3-bar no-pullback confirmation, detectors/pick_verification.py)
COMBINED with the fixed 12-point target + 0.75 min R:R filter
(detectors.scanner.FIXED_TARGET_POINTS/MIN_FIXED_TARGET_RR), against the
same 4 strategies it's live-wired to (opening_range_breakout, vwap_reclaim,
smart_money, fibonacci_retracement) -- via backtest.engine.run_multi()'s
verification_config parameter, which uses the RIGOROUS bar-by-bar wait-state
version of verification (the same logic paper_trading/engine.py's
PaperTradingEngine live_mode uses), NOT detectors/scanner.py's fast-scan
approximation -- see run_multi()'s own docstring for why that distinction
matters here.

Runs TWO variants per symbol in the SAME worker call, sharing the one-time
disk load: a BASELINE pass (original strategy behavior: real 2x-risk-reward
target, no verification -- reproduces exactly what examples/
run_universe_backtest.py already reports) and a VERIFIED pass
(verification_config=True for all 4 names), so the comparison is a true
apples-to-apples same-symbols-same-session comparison rather than pulling
possibly-stale numbers from an earlier, differently-scoped run.

Run: python -m examples.run_verification_backtest --limit 400
    python -m examples.run_verification_backtest --limit 400 --strategy smart_money
        (isolated single-strategy run -- e.g. to test verification alone,
        without the other 3 strategies' trades diluting the picture)
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from datetime import date, timedelta

from backtest.engine import Trade, run_multi
from backtest.metrics import compute
from config.settings import STORAGE_ROOT
from detectors.fibonacci_strategy import FibonacciStrategy
from detectors.smc_strategy import SmartMoneyStrategy
from detectors.vwap_strategy import VWAPStrategy
from examples.opening_range_breakout import OpeningRangeBreakout

STRATEGY_BUILDERS = {
    "opening_range_breakout": (lambda: OpeningRangeBreakout(range_minutes=15, volume_multiplier=1.5), [1, 5, 15]),
    "vwap_reclaim": (lambda: VWAPStrategy(volume_lookback_bars=20, volume_multiplier=1.3, stop_lookback_bars=15), [1]),
    "smart_money": (lambda: SmartMoneyStrategy(order_block_lookback_bars=50, move_atr_mult=1.5), [1]),
    "fibonacci_retracement": (lambda: FibonacciStrategy(swing_lookback_bars=50, min_leg_atr_mult=1.5), [1]),
}


def _cached_symbols() -> list[str]:
    return sorted(p.name.removesuffix("_1min.parquet") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _fresh_specs(names: tuple[str, ...]) -> dict:
    """New Strategy instances every call -- run_multi() being called
    TWICE per symbol here (baseline + verified) absolutely requires this:
    reusing the same instances would leak _traded_today state from the
    first call into the second, silently suppressing every signal in the
    second run (confirmed the hard way while building this script).
    `names`: which STRATEGY_BUILDERS entries to include -- passed through
    the task tuple (not a module-global filter) so this works correctly
    under Windows' multiprocessing 'spawn' start method, where each worker
    process re-imports this module fresh and would silently ignore a
    global mutated only in the main process after argparse."""
    return {name: (STRATEGY_BUILDERS[name][0](), STRATEGY_BUILDERS[name][1]) for name in names}


def _run_one(task: tuple[str, date, date, tuple[str, ...]]) -> dict:
    """Worker: runs BOTH variants against ONE symbol, sharing nothing
    across the two run_multi() calls except the symbol's on-disk data
    (each call reloads it independently, same as run_universe_backtest.py's
    per-symbol cost -- reloading twice here is a deliberate simplicity
    tradeoff, not an oversight, since correctness > shaving a second disk
    read at this scale)."""
    symbol, start_date, end_date, names = task
    try:
        baseline = run_multi(symbol, _fresh_specs(names), start_date, end_date)
        verified = run_multi(symbol, _fresh_specs(names), start_date, end_date,
                              verification_config={name: True for name in names})
    except Exception as e:
        print(f"  {symbol}: {e}")
        return {"symbol": symbol, "baseline": {n: [] for n in names},
                "verified": {n: [] for n in names}, "rejected": {n: 0 for n in names}}
    return {
        "symbol": symbol,
        "baseline": {name: result.trades for name, result in baseline.items()},
        "verified": {name: result.trades for name, result in verified.items()},
        "rejected": {name: result.rejected_signal_count for name, result in verified.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=400, help="Number of cached symbols to test (default 400).")
    parser.add_argument("--strategy", type=str, default=None, choices=sorted(STRATEGY_BUILDERS),
                         help="Restrict to ONE strategy (default: all 4) -- isolates that strategy's "
                              "verification-vs-baseline comparison without the other 3 diluting the picture.")
    args = parser.parse_args()

    names = (args.strategy,) if args.strategy else tuple(STRATEGY_BUILDERS)

    symbols = _cached_symbols()
    if not symbols:
        print("No cached data found -- run data/bulk_fetch.py first.")
        return
    symbols = symbols[:args.limit]

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)

    cores = os.cpu_count() or 1
    tasks = [(symbol, start_date, end_date, names) for symbol in symbols]
    print(f"Backtesting {len(symbols)} symbols x {len(names)} strateg{'y' if len(names)==1 else 'ies'} "
          f"({', '.join(names)}) x 2 variants (baseline + verified), {cores} CPU cores...\n")

    raw_results = []
    start_time = time.monotonic()
    with multiprocessing.Pool(processes=cores) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, tasks), start=1):
            raw_results.append(r)
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.monotonic() - start_time
                eta_min = (elapsed / i) * (len(tasks) - i) / 60
                print(f"{i}/{len(tasks)} symbols done, ETA {eta_min:.1f} min", flush=True)

    baseline_trades: dict[str, list[Trade]] = {name: [] for name in names}
    verified_trades: dict[str, list[Trade]] = {name: [] for name in names}
    rejected_counts: dict[str, int] = {name: 0 for name in names}
    for r in raw_results:
        for name in names:
            baseline_trades[name].extend(r["baseline"][name])
            verified_trades[name].extend(r["verified"][name])
            rejected_counts[name] += r["rejected"][name]

    print(f"\n{'=' * 100}")
    print(f"{'Strategy':26s} {'Variant':10s} {'Trades':>7s} {'WinRate':>8s} {'PF':>7s} {'AvgR':>7s} "
          f"{'Sharpe':>7s} {'MaxDD':>9s} {'Rejected':>9s}")
    print("=" * 100)

    combined_baseline: list[Trade] = []
    combined_verified: list[Trade] = []
    for name in names:
        b = compute(baseline_trades[name])
        v = compute(verified_trades[name])
        combined_baseline.extend(baseline_trades[name])
        combined_verified.extend(verified_trades[name])
        pf_b = f"{b.profit_factor:.2f}" if b.profit_factor != float("inf") else "inf"
        pf_v = f"{v.profit_factor:.2f}" if v.profit_factor != float("inf") else "inf"
        print(f"{name:26s} {'baseline':10s} {b.total_trades:7d} {b.win_rate:7.1%}  {pf_b:>6s} "
              f"{b.avg_r_multiple:7.2f} {b.sharpe_ratio:7.2f} {b.max_drawdown_points:9.1f} {'--':>9s}")
        print(f"{'':26s} {'verified':10s} {v.total_trades:7d} {v.win_rate:7.1%}  {pf_v:>6s} "
              f"{v.avg_r_multiple:7.2f} {v.sharpe_ratio:7.2f} {v.max_drawdown_points:9.1f} "
              f"{rejected_counts[name]:9d}")

    if len(names) > 1:
        print("=" * 100)
        cb = compute(combined_baseline)
        cv = compute(combined_verified)
        pf_cb = f"{cb.profit_factor:.2f}" if cb.profit_factor != float("inf") else "inf"
        pf_cv = f"{cv.profit_factor:.2f}" if cv.profit_factor != float("inf") else "inf"
        total_rejected = sum(rejected_counts.values())
        print(f"{'COMBINED (' + str(len(names)) + ')':26s} {'baseline':10s} {cb.total_trades:7d} "
              f"{cb.win_rate:7.1%}  {pf_cb:>6s} {cb.avg_r_multiple:7.2f} {cb.sharpe_ratio:7.2f} "
              f"{cb.max_drawdown_points:9.1f} {'--':>9s}")
        print(f"{'':26s} {'verified':10s} {cv.total_trades:7d} {cv.win_rate:7.1%}  {pf_cv:>6s} "
              f"{cv.avg_r_multiple:7.2f} {cv.sharpe_ratio:7.2f} {cv.max_drawdown_points:9.1f} "
              f"{total_rejected:9d}")
    print("=" * 100)

    print(f"\nTotal elapsed: {(time.monotonic() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
