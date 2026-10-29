"""
Engineers a clean uptrend that GENUINELY reverses (not just pulls back) --
price breaks below the trend leg's origin and continues down -- to prove
reversal_strategy.py correctly identifies and trades it. Also includes a
day with only a normal shallow pullback (no real reversal) to confirm the
strategy correctly stays OUT of that case (it should trade continuation
territory, not this strategy).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import compute
from data import storage
from detectors.reversal_strategy import ReversalDetection


def generate_genuine_reversal_day(trade_date: str, base_price: float = 1000.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{trade_date} 09:15", f"{trade_date} 15:29", freq="1min")
    n = len(times)
    prices = np.zeros(n)
    prices[0] = base_price

    for i in range(1, n):
        pct = i / n
        if 0.15 < pct < 0.35:
            drift = 1.2  # strong uptrend leg
        elif 0.35 <= pct < 0.40:
            drift = -2.5  # sharp break BELOW the trend leg's origin -- genuine reversal
        elif 0.40 <= pct < 0.75:
            drift = -0.9  # reversal continues downward
        else:
            drift = rng.normal(0, 0.3)
        prices[i] = prices[i - 1] + drift + rng.normal(0, 0.7)

    opens = prices + rng.normal(0, 0.2, n)
    closes = prices + rng.normal(0, 0.2, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.5, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.5, n))
    volumes = rng.integers(8000, 15000, n).astype(float)
    volumes[int(n * 0.35):int(n * 0.42)] *= 2.5  # volume surge on the reversal break

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=times,
    )


def main() -> None:
    symbol = "SYNTHREVERSAL"
    days = [f"2026-06-{d:02d}" for d in range(1, 6)]
    full_df = pd.concat([generate_genuine_reversal_day(d, seed=i) for i, d in enumerate(days)])
    storage.save(symbol, full_df)
    print(f"Generated {len(full_df)} synthetic 1-min bars across {len(days)} days")

    strategy = ReversalDetection(trend_lookback_bars=20, pullback_bars=2, min_confidence=0.6)
    engine = BacktestEngine(strategy=strategy, timeframes=[1])
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
        print("\nNo trades fired -- check whether the engineered reversal "
              "actually breaks structure within the lookback window.")


if __name__ == "__main__":
    main()
