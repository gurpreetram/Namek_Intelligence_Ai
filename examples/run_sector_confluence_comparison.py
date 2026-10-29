"""
Compares TrendContinuation against SectorConfluenceTrend (TrendContinuation
+ sector-trend agreement gate, see detectors/sector_confluence_trend_strategy.py)
across the same 400-symbol cached test set used elsewhere in this project,
to see directly whether the sector filter improves trade quality or just
throws away trades without improving the average outcome.

Honest framing up front: config/sector_mapping.json only covers ~4% of the
cached universe (built from the top-10-by-weight constituents of 11 NSE
sectoral index factsheets, not full official constituent lists -- see that
file's own generation notes). SectorConfluenceTrend can only possibly fire
on symbols with a sector label, so its trade count will be far smaller than
TrendContinuation's by construction, not necessarily because the filter is
"working" or "not working" -- read the per-trade averages (win rate, PF,
avg R), not just the raw trade count, to judge the filter itself.

Run: python -m examples.run_sector_confluence_comparison
"""
from __future__ import annotations

import json
import multiprocessing
from datetime import date, timedelta

from backtest.engine import BacktestEngine, Trade
from backtest.metrics import compute
from config.settings import PROJECT_ROOT, STORAGE_ROOT
from detectors.sector_confluence_trend_strategy import SectorConfluenceTrend
from detectors.sector_trend import load_sector_price_data
from detectors.trend_continuation_strategy import TrendContinuation

TREND_CONTINUATION_KWARGS = dict(trend_lookback_bars=20, pullback_bars=2, min_confidence=0.75)

_sector_mapping: dict[str, str] = {}
_sector_price_data: dict[str, dict] = {}


def _init_worker(sector_mapping: dict[str, str]) -> None:
    """Pool initializer: loads sector price data ONCE per worker process,
    not once per task -- avoids re-pickling every symbol's full price
    history through the task queue for every one of the 400 tasks."""
    global _sector_mapping, _sector_price_data
    _sector_mapping = sector_mapping
    _sector_price_data = load_sector_price_data(sector_mapping)


def _run_one(task: tuple[str, date, date]) -> tuple[str, list[Trade], list[Trade]]:
    symbol, start_date, end_date = task
    tc_engine = BacktestEngine(strategy=TrendContinuation(**TREND_CONTINUATION_KWARGS), timeframes=[1, 15])
    sct_engine = BacktestEngine(
        strategy=SectorConfluenceTrend(_sector_mapping, _sector_price_data, **TREND_CONTINUATION_KWARGS),
        timeframes=[1, 15],
    )
    try:
        tc_trades = tc_engine.run(symbol, start_date, end_date).trades
    except Exception as e:
        print(f"  [trend_continuation] {symbol}: {e}")
        tc_trades = []
    try:
        sct_trades = sct_engine.run(symbol, start_date, end_date).trades
    except Exception as e:
        print(f"  [sector_confluence_trend] {symbol}: {e}")
        sct_trades = []
    return symbol, tc_trades, sct_trades


def main() -> None:
    sector_mapping = json.loads((PROJECT_ROOT / "config" / "sector_mapping.json").read_text())
    symbols = sorted(p.name.removesuffix("_1min.parquet") for p in STORAGE_ROOT.glob("*_1min.parquet"))[:400]
    covered = [s for s in symbols if s in sector_mapping]
    print(f"400-symbol test set: {len(covered)}/{len(symbols)} have a sector label "
          f"({len(covered)/len(symbols)*100:.1f}%)")
    print(f"Sector-labeled symbols in this set: {covered}\n")

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)
    tasks = [(symbol, start_date, end_date) for symbol in symbols]
    cores = 8

    with multiprocessing.Pool(processes=cores, initializer=_init_worker, initargs=(sector_mapping,)) as pool:
        results = pool.map(_run_one, tasks)

    tc_all: list[Trade] = []
    sct_all: list[Trade] = []
    for symbol, tc_trades, sct_trades in results:
        tc_all.extend(tc_trades)
        sct_all.extend(sct_trades)

    tc_report = compute(tc_all)
    sct_report = compute(sct_all)

    print("=" * 60)
    print("TrendContinuation (baseline)")
    print("=" * 60)
    print(tc_report)

    print()
    print("=" * 60)
    print("SectorConfluenceTrend (TrendContinuation + sector agreement gate)")
    print("=" * 60)
    print(sct_report)

    print()
    print("=" * 60)
    print("SIDE BY SIDE")
    print("=" * 60)
    print(f"{'Metric':22s} {'TrendContinuation':>20s} {'SectorConfluenceTrend':>22s}")
    print(f"{'Trades':22s} {tc_report.total_trades:20d} {sct_report.total_trades:22d}")
    print(f"{'Win rate':22s} {tc_report.win_rate:19.1%} {sct_report.win_rate:21.1%}")
    pf_tc = f"{tc_report.profit_factor:.2f}" if tc_report.profit_factor != float("inf") else "inf"
    pf_sct = f"{sct_report.profit_factor:.2f}" if sct_report.profit_factor != float("inf") else "inf"
    print(f"{'Profit factor':22s} {pf_tc:>20s} {pf_sct:>22s}")
    print(f"{'Avg R-multiple':22s} {tc_report.avg_r_multiple:20.2f} {sct_report.avg_r_multiple:22.2f}")
    print(f"{'Sharpe':22s} {tc_report.sharpe_ratio:20.2f} {sct_report.sharpe_ratio:22.2f}")


if __name__ == "__main__":
    main()
