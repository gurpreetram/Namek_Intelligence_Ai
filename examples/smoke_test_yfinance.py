"""
Smoke test: proves the backtest engine runs end-to-end using free
yfinance data (last ~7 days only). This does NOT validate whether the
Opening Range Breakout strategy is actually good -- 7 days is far too
small a sample to conclude anything about win rate or profitability.
It only proves the engine's mechanics (no-lookahead slicing, resampling,
trade fills, exit logic, metrics calculation) don't crash or produce
nonsense on real market data.

Run: python -m examples.smoke_test_yfinance

Once your Upstox account clears KYC, switch to
examples/run_backtest_example.py for real validation with proper history.
"""
from __future__ import annotations

from datetime import date, timedelta

from backtest.engine import BacktestEngine
from backtest.metrics import compute
from data import fetcher_yfinance, storage
from examples.opening_range_breakout import OpeningRangeBreakout

SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK"]


def main() -> None:
    print("Fetching free yfinance data (last ~7 days, 1-min bars)...")
    print("NOTE: this is a plumbing test only -- not a real backtest sample.\n")

    for symbol in SYMBOLS:
        df = fetcher_yfinance.fetch_recent_1min(symbol)
        if df.empty:
            print(f"{symbol}: no data returned -- check the symbol or your network")
            continue
        storage.save(symbol, df)
        print(f"{symbol}: cached {len(df)} 1-min bars "
              f"({df.index.min()} to {df.index.max()})")

    strategy = OpeningRangeBreakout(range_minutes=15, volume_multiplier=1.5)
    engine = BacktestEngine(strategy=strategy, timeframes=[1, 5, 15])

    end_date = date.today()
    start_date = end_date - timedelta(days=10)

    all_trades = []
    for symbol in SYMBOLS:
        coverage = storage.coverage(symbol)
        if coverage is None:
            continue
        try:
            result = engine.run(symbol, start_date, end_date)
        except ValueError as e:
            print(f"{symbol}: {e}")
            continue
        print(f"{symbol}: {len(result.trades)} trades")
        all_trades.extend(result.trades)

    print("\n" + "=" * 50)
    print("SMOKE TEST REPORT (engine mechanics check, not a real backtest)")
    print("=" * 50)
    if not all_trades:
        print("No trades generated. This can be normal with only ~7 days of "
              "data and 3 symbols -- ORB doesn't fire every day. If you want "
              "to confirm the engine itself is working, check that candles "
              "were cached above and no exceptions were thrown.")
        return

    report = compute(all_trades)
    print(report)
    print("\nIf this ran without errors and produced a report (even with few "
          "trades), the engine's mechanics are sound. Move to Upstox once "
          "KYC clears for a real validation run.")


if __name__ == "__main__":
    main()
