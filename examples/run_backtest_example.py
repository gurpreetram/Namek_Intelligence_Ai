"""
End-to-end validation run: fetch real NSE data via Upstox, cache it,
run the Opening Range Breakout strategy through the backtest engine, and
print a performance report.

Before running:
  1. python -m config.auth login          (open URL, log in, get code)
  2. python -m config.auth exchange <code>
  3. pip install -r requirements.txt

Then:
  python -m examples.run_backtest_example
"""
from __future__ import annotations

from datetime import date, timedelta

from backtest.engine import BacktestEngine
from backtest.metrics import compute
from data import fetcher, instruments, storage
from examples.opening_range_breakout import OpeningRangeBreakout

# Small, liquid universe to start with -- expand once the engine is validated.
SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

# Stay within the confirmed data-availability window (Jan 2022 onward) and
# use a range recent enough to be representative of current market behavior.
END_DATE = date.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=90)


def ensure_data_cached(symbol: str) -> None:
    key = instruments.resolve_instrument_key(symbol)
    if key is None:
        raise ValueError(
            f"Could not resolve instrument_key for '{symbol}'. Check the "
            f"symbol is correct, or that Upstox's field names in the "
            f"instrument master match what data/instruments.py expects "
            f"(verify against a fresh download if this fails)."
        )

    existing = storage.coverage(symbol)
    if existing and existing[0].date() <= START_DATE and existing[1].date() >= END_DATE:
        print(f"{symbol}: already cached for {START_DATE}..{END_DATE}")
        return

    print(f"{symbol}: fetching {START_DATE}..{END_DATE} from Upstox...")
    df = fetcher.fetch_historical_1min(key, START_DATE, END_DATE)
    storage.save(symbol, df)
    print(f"{symbol}: cached {len(df)} 1-min bars")


def main() -> None:
    strategy = OpeningRangeBreakout(range_minutes=15, volume_multiplier=1.5)
    engine = BacktestEngine(strategy=strategy, timeframes=[1, 5, 15])

    all_trades = []
    for symbol in SYMBOLS:
        ensure_data_cached(symbol)
        result = engine.run(symbol, START_DATE, END_DATE)
        print(f"\n{symbol}: {len(result.trades)} trades")
        all_trades.extend(result.trades)

    print("\n" + "=" * 50)
    print(f"COMBINED REPORT ({len(SYMBOLS)} symbols, {START_DATE} to {END_DATE})")
    print("=" * 50)
    report = compute(all_trades)
    print(report)

    if all_trades:
        print("\nSample trade with reasoning:")
        sample = all_trades[0]
        print(f"  {sample.symbol} {sample.direction.value} @ {sample.entry_price} "
              f"-> {sample.exit_reason} @ {sample.exit_price} (R={sample.r_multiple:.2f})")
        for reason in sample.reasoning:
            print(f"    - {reason}")


if __name__ == "__main__":
    main()
