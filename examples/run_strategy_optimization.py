"""
Deep-optimizes TWO strategies -- SmartMoneyStrategy and ValidatedBreakout
-- leaving the other 5 strategies completely untouched (no shared code
path was modified in a way that changes their behavior; every new
parameter added to detectors/smc_strategy.py, detectors/
validated_breakout_strategy.py, and detectors/breakout_validator.py
defaults to reproducing the EXACT original behavior, so the live system
and every other strategy/script is unaffected unless it explicitly opts
into a new parameter).

Same harness as examples/run_verification_backtest.py: backtest.engine's
run_multi() (which already applies the liquidity gate and gap-risk guard
identically for every variant -- see backtest/engine.py's own docstring,
nothing extra needed here) and backtest.metrics.compute(), over the same
400-symbol / 30-day window convention as every other "standard
comparison test" in this project. Each of the 10 variants below (5 SMC +
5 ValidatedBreakout) is evaluated on the SAME symbols/session in one
pass per symbol, sharing the one-time disk load -- a true apples-to-
apples comparison, not separately-scoped runs.

SMART MONEY variants (baseline matches the always-live default exactly):
  baseline          -- move_atr_mult=1.5, no fixed stop, no time filter (unchanged)
  stop_refinement   -- fixed_stop_atr_mult=0.75 (tight ATR-only stop, replacing the zone-based one)
  quality_filter    -- move_atr_mult=2.0 (stronger pre-block move required)
  time_filter       -- entry_time_window=("09:30","13:00")
  combined          -- all three together

VALIDATED BREAKOUT variants (baseline matches the always-live default exactly):
  baseline             -- required_confirmation_bars=1, no distance gate, min_volume_zscore=0.5 (unchanged)
  confirmation_bars    -- required_confirmation_bars=2
  distance_filter      -- min_distance_atr_mult=1.5
  volume_filter        -- min_volume_zscore=1.5
  combined             -- all three together

Run: python -m examples.run_strategy_optimization
    python -m examples.run_strategy_optimization --limit 400
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
from detectors.validated_breakout_strategy import ValidatedBreakout

SMC_VARIANTS = {
    "baseline": dict(order_block_lookback_bars=50, move_atr_mult=1.5),
    "stop_refinement": dict(order_block_lookback_bars=50, move_atr_mult=1.5, fixed_stop_atr_mult=0.75),
    "quality_filter": dict(order_block_lookback_bars=50, move_atr_mult=2.0),
    "time_filter": dict(order_block_lookback_bars=50, move_atr_mult=1.5, entry_time_window=("09:30", "13:00")),
    "combined": dict(order_block_lookback_bars=50, move_atr_mult=2.0, fixed_stop_atr_mult=0.75,
                      entry_time_window=("09:30", "13:00")),
}

VB_VARIANTS = {
    "baseline": dict(),
    "confirmation_bars": dict(required_confirmation_bars=2),
    "distance_filter": dict(min_distance_atr_mult=1.5),
    "volume_filter": dict(min_volume_zscore=1.5),
    "combined": dict(required_confirmation_bars=2, min_distance_atr_mult=1.5, min_volume_zscore=1.5),
}

# name -> (timeframes,) -- SMC only ever reads 1min data; ValidatedBreakout
# needs 5min/15min too (level detection + MTF alignment gate), matching
# live_poller.py's STRATEGY_BUILDERS timeframe lists for these two exactly.
ALL_VARIANT_NAMES = (
    [f"smc__{v}" for v in SMC_VARIANTS] + [f"vb__{v}" for v in VB_VARIANTS]
)


def _build_specs() -> dict:
    specs = {}
    for variant, kwargs in SMC_VARIANTS.items():
        specs[f"smc__{variant}"] = (SmartMoneyStrategy(**kwargs), [1])
    for variant, kwargs in VB_VARIANTS.items():
        specs[f"vb__{variant}"] = (ValidatedBreakout(**kwargs), [1, 5, 15])
    return specs


def _cached_symbols() -> list[str]:
    return sorted(p.name.removesuffix("_1min.parquet") for p in STORAGE_ROOT.glob("*_1min.parquet"))


def _run_one(task: tuple[str, date, date]) -> dict:
    symbol, start_date, end_date = task
    try:
        results = run_multi(symbol, _build_specs(), start_date, end_date)
    except Exception as e:
        print(f"  {symbol}: {e}")
        return {"symbol": symbol, "trades": {name: [] for name in ALL_VARIANT_NAMES}}
    return {"symbol": symbol, "trades": {name: r.trades for name, r in results.items()}}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=400, help="Number of cached symbols to test (default 400).")
    args = parser.parse_args()

    symbols = _cached_symbols()
    if not symbols:
        print("No cached data found -- run data/bulk_fetch.py first.")
        return
    symbols = symbols[:args.limit]

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)

    cores = os.cpu_count() or 1
    tasks = [(symbol, start_date, end_date) for symbol in symbols]
    print(f"Backtesting {len(symbols)} symbols x {len(ALL_VARIANT_NAMES)} variants "
          f"(5 SmartMoney + 5 ValidatedBreakout), {cores} CPU cores...\n")

    all_trades: dict[str, list[Trade]] = {name: [] for name in ALL_VARIANT_NAMES}
    start_time = time.monotonic()
    with multiprocessing.Pool(processes=cores) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, tasks), start=1):
            for name in ALL_VARIANT_NAMES:
                all_trades[name].extend(r["trades"][name])
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.monotonic() - start_time
                eta_min = (elapsed / i) * (len(tasks) - i) / 60
                print(f"{i}/{len(tasks)} symbols done, ETA {eta_min:.1f} min", flush=True)

    elapsed = time.monotonic() - start_time
    print(f"\nDone in {elapsed / 60:.1f} min\n")

    def _print_table(title: str, variants: dict, prefix: str) -> None:
        print("=" * 100)
        print(title)
        print("=" * 100)
        print(f"{'Variant':22s} {'Trades':>7s} {'WinRate':>8s} {'PF':>7s} {'AvgR':>7s} {'Sharpe':>7s} {'MaxDD':>9s}")
        print("-" * 100)
        baseline_report = None
        for variant in variants:
            trades = all_trades[f"{prefix}__{variant}"]
            report = compute(trades)
            pf = f"{report.profit_factor:.2f}" if report.profit_factor != float("inf") else "inf"
            marker = ""
            if variant == "baseline":
                baseline_report = report
            elif baseline_report is not None:
                worse = []
                if report.profit_factor < baseline_report.profit_factor:
                    worse.append("PF down")
                if report.win_rate < baseline_report.win_rate:
                    worse.append("WR down")
                if report.avg_r_multiple < baseline_report.avg_r_multiple:
                    worse.append("AvgR down")
                marker = f"  <-- WORSE than baseline ({', '.join(worse)})" if worse else ""
            print(f"{variant:22s} {report.total_trades:7d} {report.win_rate:7.1%}  {pf:>6s} "
                  f"{report.avg_r_multiple:7.2f} {report.sharpe_ratio:7.2f} {report.max_drawdown_points:9.1f}{marker}")
        print()

    _print_table(f"SMART MONEY CONCEPTS -- {len(symbols)}-symbol comparison", SMC_VARIANTS, "smc")
    _print_table(f"VALIDATED BREAKOUT -- {len(symbols)}-symbol comparison", VB_VARIANTS, "vb")

    print("=" * 100)
    print("REFERENCE (as given, full-universe numbers from a DIFFERENT sample -- NOT directly comparable "
          "to the same-symbols-same-session table above, shown only as prior context):")
    print("  Smart Money Concepts (full universe): PF 1.07, WR 33.0%, avg R -0.03")
    print("  Validated Breakout (full universe):   PF 1.03, WR 32.9%, avg R -0.04")
    print("=" * 100)


if __name__ == "__main__":
    main()
