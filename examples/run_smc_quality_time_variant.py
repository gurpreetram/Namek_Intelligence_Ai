"""
TEST 1 follow-up to examples/run_strategy_optimization.py: Smart Money
Concepts with quality_filter (move_atr_mult=2.0) + time_filter
(entry_time_window=("09:30","13:00")) combined, EXPLICITLY WITHOUT the
stop_refinement (fixed_stop_atr_mult) change -- the combination flagged
in that script's report as likely beating all 4 originally-tested
variants (since stop_refinement alone was the one component that hurt),
but not actually tested there.

Same harness, same 400-symbol / 30-day window convention as
run_strategy_optimization.py, for direct comparability against its
already-reported baseline/quality_filter/time_filter/combined rows.

Run: python -m examples.run_smc_quality_time_variant
"""
from __future__ import annotations

import multiprocessing
import os
import time
from datetime import date, timedelta

from backtest.engine import Trade, run_multi
from backtest.metrics import compute
from config.settings import STORAGE_ROOT
from detectors.smc_strategy import SmartMoneyStrategy

VARIANT_KWARGS = dict(order_block_lookback_bars=50, move_atr_mult=2.0, entry_time_window=("09:30", "13:00"))


def _cached_symbols() -> list[str]:
    return sorted(p.name.removesuffix("_1min.parquet") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _run_one(task: tuple[str, date, date]) -> list[Trade]:
    symbol, start_date, end_date = task
    try:
        result = run_multi(symbol, {"smc__quality_time_only": (SmartMoneyStrategy(**VARIANT_KWARGS), [1])},
                            start_date, end_date)
        return result["smc__quality_time_only"].trades
    except Exception as e:
        print(f"  {symbol}: {e}")
        return []


def main() -> None:
    symbols = _cached_symbols()[:400]
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)

    cores = os.cpu_count() or 1
    tasks = [(symbol, start_date, end_date) for symbol in symbols]
    print(f"SMC quality_filter (2x ATR) + time_filter (09:30-13:00) ONLY -- NO stop_refinement")
    print(f"Backtesting {len(symbols)} symbols, {cores} CPU cores...\n")

    all_trades: list[Trade] = []
    start_time = time.monotonic()
    with multiprocessing.Pool(processes=cores) as pool:
        for i, trades in enumerate(pool.imap_unordered(_run_one, tasks), start=1):
            all_trades.extend(trades)
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.monotonic() - start_time
                eta_min = (elapsed / i) * (len(tasks) - i) / 60
                print(f"{i}/{len(tasks)} symbols done, ETA {eta_min:.1f} min", flush=True)

    elapsed = time.monotonic() - start_time
    print(f"\nDone in {elapsed / 60:.1f} min\n")

    report = compute(all_trades)
    pf = f"{report.profit_factor:.2f}" if report.profit_factor != float("inf") else "inf"
    print("=" * 70)
    print(f"quality_filter + time_filter ONLY (no stop_refinement) -- {len(symbols)}-symbol comparison")
    print("=" * 70)
    print(f"Trades: {report.total_trades}  WinRate: {report.win_rate:.1%}  PF: {pf}  "
          f"AvgR: {report.avg_r_multiple:.2f}  Sharpe: {report.sharpe_ratio:.2f}  MaxDD: {report.max_drawdown_points:.1f}")
    print()
    print("REFERENCE (same 400-symbol run, from run_strategy_optimization.py's report):")
    print("  baseline          2319 trades  WR 33.9%  PF 1.01  AvgR  0.00  MaxDD 777.6")
    print("  stop_refinement   2062 trades  WR 32.1%  PF 0.86  AvgR -0.06  MaxDD 641.0")
    print("  quality_filter    2126 trades  WR 34.3%  PF 1.08  AvgR  0.02  MaxDD 601.3")
    print("  time_filter       1941 trades  WR 33.1%  PF 1.05  AvgR  0.01  MaxDD 781.9")
    print("  combined (all 3)  1615 trades  WR 32.6%  PF 0.96  AvgR -0.04  MaxDD 200.4")


if __name__ == "__main__":
    main()
