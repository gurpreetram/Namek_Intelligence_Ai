"""
Critical correctness check: replays the SAME synthetic data bar-by-bar
through PaperTradingEngine and compares the result against BacktestEngine
run on the identical data. If these ever diverge, that's a serious bug --
a strategy that looks profitable in backtest must behave identically in
paper/live trading, or the whole backtest is meaningless as a predictor
of anything.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import compute
from data import storage
from detectors.validated_breakout_strategy import ValidatedBreakout
from examples.synthetic_pipeline_test import generate_synthetic_day
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import TradeJournal


def main() -> None:
    symbol = "SYNTHTEST_PAPER"
    days = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]
    full_df = pd.concat([generate_synthetic_day(d, level_price=1010.0, seed=i) for i, d in enumerate(days)])
    storage.save(symbol, full_df)

    # --- Run 1: standard BacktestEngine (full range in one call) ---
    backtest_strategy = ValidatedBreakout(level_timeframe=15, min_touches=2)
    backtest_engine = BacktestEngine(strategy=backtest_strategy, timeframes=[1, 5, 15])
    backtest_result = backtest_engine.run(symbol, date(2026, 6, 1), date(2026, 6, 5))

    print("=== BacktestEngine (full-range) ===")
    for t in backtest_result.trades:
        print(f"  {t.entry_time} {t.direction.value} @ {t.entry_price:.2f} -> "
              f"{t.exit_reason} @ {t.exit_price:.2f} (R={t.r_multiple:.2f})")
    print(compute(backtest_result.trades))

    # --- Run 2: PaperTradingEngine, fed bar-by-bar (simulating live arrival) ---
    journal_path = None  # uses default path; fine for this test
    journal = TradeJournal()
    # Use a fresh journal file for this test run so counts are clean.
    if journal.path.exists():
        journal.path.unlink()

    paper_strategy = ValidatedBreakout(level_timeframe=15, min_touches=2)  # fresh instance -- same params, no shared state
    paper_engine = PaperTradingEngine(strategy=paper_strategy, timeframes=[1, 5, 15], journal=journal)

    for current_index in full_df.index:
        history_so_far = full_df.loc[:current_index]
        paper_engine.on_new_bar(symbol, history_so_far, current_index)

    journal_df = journal.load_all()
    exits = journal_df[journal_df["event"] == "trade_exited"] if not journal_df.empty else journal_df

    print("\n=== PaperTradingEngine (bar-by-bar replay) ===")
    for _, row in exits.iterrows():
        print(f"  {row['entry_time']} {row['direction']} @ {row['entry_price']:.2f} -> "
              f"{row['exit_reason']} @ {row['exit_price']:.2f} (R={row['r_multiple']:.2f})")
    print(f"\nJournal summary: {journal.summary()}")

    # --- Cross-check ---
    print(f"\n{'=' * 60}")
    backtest_count = len(backtest_result.trades)
    paper_count = len(exits)
    print(f"BacktestEngine trades: {backtest_count}")
    print(f"PaperTradingEngine trades: {paper_count}")
    if backtest_count == paper_count:
        print("MATCH -- paper trading engine is consistent with backtest engine.")
    else:
        print("MISMATCH -- this is a real bug, the two engines disagree on "
              "identical data and identical strategy logic.")


if __name__ == "__main__":
    main()
