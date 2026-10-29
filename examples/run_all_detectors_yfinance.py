"""
Runs ALL FOUR phases against real (but limited) free yfinance data:
  1. Backtest engine + Opening Range Breakout (baseline sanity check)
  2. Validated Breakout strategy (support/resistance + fakeout rejection)
  3. Trend Continuation strategy (pullback classifier)
  4. Scanner (live-style scan across the universe as of the most recent bar)

Results are printed to the terminal AND written to an HTML report that
auto-opens in your browser -- easier to read than scrolling terminal output.

This is the first time any of this runs against REAL market data instead
of synthetic/engineered data. Two honest limits to keep in mind reading
the output:
  - yfinance only gives ~7 days of 1-minute NSE data -- far too small a
    sample to conclude anything about real profitability. Treat every
    metric here as "does this look sane," not "is this strategy good."
  - Real data will surface things synthetic data can't: actual gaps,
    illiquid minutes, unusual volume spikes around news, etc. If
    something looks weird in the output, paste it back to me exactly
    as printed -- that's more useful than summarizing it.

Run: python -m examples.run_all_detectors_yfinance
"""
from __future__ import annotations

import webbrowser
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import compute
from config.settings import PROJECT_ROOT
from data import fetcher_yfinance, storage
from detectors.scanner import scan_universe
from detectors.validated_breakout_strategy import ValidatedBreakout
from detectors.trend_continuation_strategy import TrendContinuation
from detectors.reversal_strategy import ReversalDetection
from examples.opening_range_breakout import OpeningRangeBreakout
from reporting.html_report import generate_report

SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]


def fetch_and_cache_all() -> dict[str, pd.DataFrame]:
    data = {}
    for symbol in SYMBOLS:
        print(f"Fetching {symbol}...")
        df = fetcher_yfinance.fetch_recent_1min(symbol)
        if df.empty:
            print(f"  {symbol}: no data returned, skipping")
            continue
        storage.save(symbol, df)
        data[symbol] = df
        print(f"  {symbol}: {len(df)} bars, {df.index.min()} to {df.index.max()}")
    return data


def run_strategy(name: str, strategy, timeframes: list[int],
                  start_date: date, end_date: date) -> tuple[list, object]:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    engine = BacktestEngine(strategy=strategy, timeframes=timeframes)
    all_trades = []
    for symbol in SYMBOLS:
        try:
            result = engine.run(symbol, start_date, end_date)
        except ValueError as e:
            print(f"{symbol}: {e}")
            continue
        if result.trades:
            print(f"{symbol}: {len(result.trades)} trades")
            for t in result.trades:
                print(f"    {t.entry_time} {t.direction.value} @ {t.entry_price:.2f} -> "
                      f"{t.exit_reason} @ {t.exit_price:.2f} (R={t.r_multiple:.2f})")
        all_trades.extend(result.trades)

    report = compute(all_trades)
    if all_trades:
        print(f"\n{report}")
    else:
        print("No trades generated -- with only ~7 days of data this can be "
              "normal, especially for the stricter validated_breakout and "
              "trend_continuation strategies, which are designed to skip "
              "more often than they trade.")
    return all_trades, report


def main() -> None:
    print("Fetching free yfinance data (last ~7 days, 1-min bars)...\n")
    data = fetch_and_cache_all()
    if not data:
        print("No data fetched at all -- check your internet connection or "
              "whether yfinance needs updating (pip install --upgrade yfinance).")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=10)

    strategy_results = {}

    trades, report = run_strategy(
        "Opening Range Breakout (baseline)",
        OpeningRangeBreakout(range_minutes=15, volume_multiplier=1.5),
        [1, 5, 15], start_date, end_date)
    strategy_results["Opening Range Breakout (baseline)"] = (trades, report)

    trades, report = run_strategy(
        "Validated Breakout",
        ValidatedBreakout(level_timeframe=15, min_touches=2),
        [1, 5, 15], start_date, end_date)
    strategy_results["Validated Breakout"] = (trades, report)

    trades, report = run_strategy(
        "Trend Continuation",
        TrendContinuation(trend_lookback_bars=20, pullback_bars=2, min_confidence=0.75),
        [1, 15], start_date, end_date)
    strategy_results["Trend Continuation"] = (trades, report)

    trades, report = run_strategy(
        "Reversal Detection",
        ReversalDetection(trend_lookback_bars=20, pullback_bars=2, min_confidence=0.6),
        [1], start_date, end_date)
    strategy_results["Reversal Detection"] = (trades, report)

    print(f"\n{'=' * 60}\nPHASE 4: Scanner (live-style scan as of the most recent bar)\n{'=' * 60}")
    most_recent = max(df.index.max() for df in data.values())
    scanner_results = scan_universe(data, most_recent, min_score=0.5)
    if not scanner_results:
        print(f"No qualifying setups as of {most_recent} -- normal, setups "
              f"aren't constant. Try running this again at a different time "
              f"of day for more scan snapshots.")
    for r in scanner_results:
        print(f"{r.symbol:10s} score={r.score:.2f}  setup={r.setup_type:20s} direction={r.direction}")
        for reason in r.reasoning:
            print(f"           - {reason}")

    report_path = PROJECT_ROOT / "reporting" / "latest_report.html"
    generate_report(report_path, strategy_results, scanner_results)
    print(f"\nHTML report written to: {report_path}")
    webbrowser.open(f"file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
