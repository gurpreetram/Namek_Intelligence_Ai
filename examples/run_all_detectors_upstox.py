"""
Runs ALL FOUR strategies + scanner against REAL Upstox historical data --
a full year by default, not yfinance's ~7-day limit. This is the first
time this system will be tested against a sample size large enough to
draw any real conclusions from.

Honest note up front: even 1 year of 5 stocks is still a modest sample
for statistical confidence (especially per-symbol or per-strategy
breakdowns) -- treat this as a first real look, not a final verdict.
More symbols and longer history (both possible now that Upstox is
working) will matter more the further you go.

Run: python -m examples.run_all_detectors_upstox
"""
from __future__ import annotations

import webbrowser
from datetime import date, timedelta

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import compute
from config.settings import PROJECT_ROOT
from data import fetcher, instruments, storage
from detectors.scanner import scan_universe
from detectors.validated_breakout_strategy import ValidatedBreakout
from detectors.trend_continuation_strategy import TrendContinuation
from detectors.reversal_strategy import ReversalDetection
from examples.opening_range_breakout import OpeningRangeBreakout
from reporting.html_report import generate_report

SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

# Upstox only serves 1-minute historical data for the trailing ~30 days.
END_DATE = date.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=29)


def fetch_and_cache_all() -> dict[str, pd.DataFrame]:
    data = {}
    for symbol in SYMBOLS:
        key = instruments.resolve_instrument_key(symbol)
        if key is None:
            print(f"{symbol}: could not resolve instrument_key, skipping")
            continue

        existing = storage.coverage(symbol)
        if existing and existing[0].date() <= START_DATE and existing[1].date() >= END_DATE:
            print(f"{symbol}: already cached for {START_DATE}..{END_DATE}")
            data[symbol] = storage.load(symbol)
            continue

        print(f"{symbol}: fetching {START_DATE}..{END_DATE} from Upstox "
              f"(this can take a bit for a full year of 1-min data)...")
        df = fetcher.fetch_historical_1min(key, START_DATE, END_DATE)
        storage.save(symbol, df)
        data[symbol] = df
        print(f"{symbol}: cached {len(df)} 1-min bars, "
              f"{df.index.min() if not df.empty else 'N/A'} to "
              f"{df.index.max() if not df.empty else 'N/A'}")
    return data


def run_strategy(name: str, strategy, timeframes: list[int]) -> tuple[list, object]:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    engine = BacktestEngine(strategy=strategy, timeframes=timeframes)
    all_trades = []
    for symbol in SYMBOLS:
        try:
            result = engine.run(symbol, START_DATE, END_DATE)
        except ValueError as e:
            print(f"{symbol}: {e}")
            continue
        if result.trades:
            print(f"{symbol}: {len(result.trades)} trades")
        all_trades.extend(result.trades)

    report = compute(all_trades)
    if all_trades:
        print(f"\n{report}")
    else:
        print("No trades generated across the full year for this strategy.")
    return all_trades, report


def main() -> None:
    print(f"Fetching real Upstox data: {START_DATE} to {END_DATE}\n")
    data = fetch_and_cache_all()
    if not data:
        print("No data fetched -- check your UPSTOX_ANALYTICS_TOKEN in .env "
              "and that segments are active on your account.")
        return

    strategy_results = {}

    trades, report = run_strategy(
        "Opening Range Breakout (baseline)",
        OpeningRangeBreakout(range_minutes=15, volume_multiplier=1.5), [1, 5, 15])
    strategy_results["Opening Range Breakout (baseline)"] = (trades, report)

    trades, report = run_strategy(
        "Validated Breakout",
        ValidatedBreakout(level_timeframe=15, min_touches=2), [1, 5, 15])
    strategy_results["Validated Breakout"] = (trades, report)

    trades, report = run_strategy(
        "Trend Continuation",
        TrendContinuation(trend_lookback_bars=20, pullback_bars=2, min_confidence=0.75), [1, 15])
    strategy_results["Trend Continuation"] = (trades, report)

    trades, report = run_strategy(
        "Reversal Detection",
        ReversalDetection(trend_lookback_bars=20, pullback_bars=2, min_confidence=0.6), [1])
    strategy_results["Reversal Detection"] = (trades, report)

    print(f"\n{'=' * 60}\nScanner (live-style scan as of the most recent bar)\n{'=' * 60}")
    most_recent = max(df.index.max() for df in data.values() if not df.empty)
    scanner_results = scan_universe(data, most_recent, min_score=0.5)
    if not scanner_results:
        print(f"No qualifying setups as of {most_recent}.")
    for r in scanner_results:
        print(f"{r.symbol:10s} score={r.score:.2f}  setup={r.setup_type:20s} direction={r.direction}")

    report_path = PROJECT_ROOT / "reporting" / "latest_report.html"
    generate_report(
        report_path, strategy_results, scanner_results,
        data_source_note=f"Real Upstox data, {START_DATE} to {END_DATE} "
                          f"({(END_DATE - START_DATE).days} days, {len(SYMBOLS)} symbols).",
    )
    print(f"\nHTML report written to: {report_path}")
    webbrowser.open(f"file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
