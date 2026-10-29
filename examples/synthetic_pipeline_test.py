"""
Generates synthetic but structurally realistic 1-minute OHLCV data
(random walk with an engineered support/resistance level and a mix of
genuine breakouts and fakeouts) purely to prove detectors/support_resistance.py,
detectors/breakout_validator.py, and detectors/validated_breakout_strategy.py
run correctly end-to-end through the backtest engine.

This is a code-correctness check, NOT a strategy validation -- synthetic
data with an engineered level obviously isn't evidence the strategy works
on real markets. Delete or ignore this once you have real Upstox/Groww
data flowing; it exists only so you (and I) can see the pipeline execute
without waiting on a data source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import compute
from data import storage
from detectors.validated_breakout_strategy import ValidatedBreakout


def generate_synthetic_day(trade_date: str, base_price: float = 1000.0,
                            level_price: float = 1010.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{trade_date} 09:15", f"{trade_date} 15:29", freq="1min")
    n = len(times)

    prices = np.zeros(n)
    prices[0] = base_price
    for i in range(1, n):
        drift = 0.0
        # Engineer reactions at the level: push back down twice, then
        # break through with volume on the third approach.
        if abs(prices[i - 1] - level_price) < 2 and i < n * 0.7:
            drift = -1.5  # rejection
        elif i > n * 0.75:
            drift = 0.8  # genuine breakout drift upward through the level
        prices[i] = prices[i - 1] + drift + rng.normal(0, 1.2)

    opens = prices + rng.normal(0, 0.3, n)
    closes = prices + rng.normal(0, 0.3, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.8, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.8, n))
    volumes = rng.integers(5000, 15000, n).astype(float)
    # Volume surge on the genuine breakout portion.
    volumes[int(n * 0.75):int(n * 0.8)] *= 3

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=times,
    )


def main() -> None:
    symbol = "SYNTHTEST"
    days = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]
    full_df = pd.concat([
        generate_synthetic_day(d, level_price=1010.0, seed=i) for i, d in enumerate(days)
    ])
    storage.save(symbol, full_df)
    print(f"Generated {len(full_df)} synthetic 1-min bars across {len(days)} days")

    strategy = ValidatedBreakout(level_timeframe=15, min_touches=2)
    engine = BacktestEngine(strategy=strategy, timeframes=[1, 5, 15])

    from datetime import date
    result = engine.run(symbol, date(2026, 6, 1), date(2026, 6, 5))

    print(f"\nTrades generated: {len(result.trades)}")
    for t in result.trades:
        print(f"  {t.entry_time} {t.direction.value} @ {t.entry_price:.2f} -> "
              f"{t.exit_reason} @ {t.exit_price:.2f} (R={t.r_multiple:.2f})")
        for r in t.reasoning:
            print(f"      - {r}")

    if result.trades:
        print("\n" + "=" * 50)
        print(compute(result.trades))
    else:
        print("\nNo trades fired. With only 5 synthetic days this can happen "
              "depending on whether min_touches=2 found a qualifying level -- "
              "not necessarily a bug. Check the level detection directly if "
              "curious (detectors/support_resistance.find_levels).")


if __name__ == "__main__":
    main()
