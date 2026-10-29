"""
Same purpose as synthetic_pipeline_test.py but for Phase 3: engineers a
clean uptrend with a shallow, low-volume pullback (should classify as
continuation) followed by resumption, to prove pullback_classifier.py
and trend_continuation_strategy.py run correctly end-to-end. Mechanics
check only -- not evidence of real-market profitability.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import compute
from data import storage
from detectors.trend_continuation_strategy import TrendContinuation


def generate_trend_with_pullback_day(trade_date: str, base_price: float = 1000.0,
                                      seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{trade_date} 09:15", f"{trade_date} 15:29", freq="1min")
    n = len(times)
    prices = np.zeros(n)
    prices[0] = base_price
    volumes = np.zeros(n)

    for i in range(1, n):
        pct = i / n
        if 0.15 < pct < 0.35:
            drift, vol = 1.2, rng.integers(15000, 22000)   # strong trend leg up
        elif 0.35 <= pct < 0.42:
            drift, vol = -0.5, rng.integers(4000, 7000)    # shallow, low-volume pullback
        elif 0.42 <= pct < 0.7:
            drift, vol = 0.9, rng.integers(12000, 18000)   # continuation resumes
        else:
            drift, vol = rng.normal(0, 0.3), rng.integers(6000, 10000)  # drift/chop
        prices[i] = prices[i - 1] + drift + rng.normal(0, 0.8)
        volumes[i] = vol

    opens = prices + rng.normal(0, 0.2, n)
    closes = prices + rng.normal(0, 0.2, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.5, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.5, n))
    volumes[0] = rng.integers(8000, 12000)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=times,
    )


def main() -> None:
    symbol = "SYNTHTREND"
    days = [f"2026-06-{d:02d}" for d in range(1, 6)]
    full_df = pd.concat([
        generate_trend_with_pullback_day(d, seed=i) for i, d in enumerate(days)
    ])
    storage.save(symbol, full_df)
    print(f"Generated {len(full_df)} synthetic 1-min bars across {len(days)} days")

    strategy = TrendContinuation(trend_lookback_bars=20, pullback_bars=2, min_confidence=0.5)
    engine = BacktestEngine(strategy=strategy, timeframes=[1, 15])
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
        print("\nNo trades fired -- check min_confidence threshold or that the "
              "engineered pullback pattern actually falls within the lookback window.")


if __name__ == "__main__":
    main()
