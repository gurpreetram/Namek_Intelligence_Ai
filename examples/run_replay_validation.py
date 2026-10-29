"""
Phase B validation -- MUST pass before ever testing the live poller against
real market hours. Feeds one full historical trading day into
LivePoller.feed_bar() bar-by-bar, in a tight loop: simulates an entire
day of live polling in seconds instead of waiting for real time to pass.
This calls the EXACT same feed_bar() method poll_once() uses live, so a
pass here genuinely proves what live polling will do -- it isn't a
parallel/simplified implementation being tested instead of the real one.

Correctness check: compares the resulting journal's trade_entered/
trade_exited events against a direct backtest/engine.py run_multi() call
for the SAME symbols/day/strategies. If PaperTradingEngine (fed
incrementally, bar by bar, exactly as live polling will) doesn't produce
the exact same trades as BacktestEngine (fed the whole day at once),
something in the incremental state management is wrong -- most likely
either the IncrementalResampler fix just added to PaperTradingEngine, or
the day-rollover bootstrap logic around it.

Warm-up parity: ValidatedBreakout caches its detected S/R levels and a
"bars since last recompute" counter PER SYMBOL, not per symbol+day (see
detectors/validated_breakout_strategy.py) -- that state carries over
across day boundaries within one strategy instance's lifetime. That means
the EXACT bar-of-day the 30-bar recompute cadence lands on depends on how
many total evaluate() calls the instance has made since its first-ever
bar, not just "since today started". For the replay and the backtest
comparison to land on identical levels (and therefore identical trades),
BOTH sides' strategy instances must start from the exact same first bar --
first found the hard way: an earlier version of this script fed the
replay only the single target day while comparing against run_multi()
seeded from each symbol's ENTIRE cached history, and 3 of 38 trades
mismatched, all in validated_breakout, all traceable to this exact cause.
Both sides now use the identical WARMUP_TRADING_DAYS-day window ending on
replay_date.

Run: python -m examples.run_replay_validation
"""
from __future__ import annotations

import pandas as pd

from backtest.engine import run_multi
from config.settings import PROJECT_ROOT
from data import storage
from paper_trading.journal import TradeJournal
from paper_trading.live_poller import STRATEGY_BUILDERS, LivePoller
from paper_trading.watchlist import WatchlistEntry

REPLAY_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "AXISBANK",
    "SBIN", "KOTAKBANK", "BAJFINANCE", "BHARTIARTL", "ITC", "LT",
    "HINDUNILVR", "MARUTI", "TITAN", "SUNPHARMA", "ASIANPAINT",
    "ULTRACEMCO", "WIPRO", "TATAMOTORS",
]

REPLAY_JOURNAL_PATH = PROJECT_ROOT / "paper_trading" / "replay_validation_journal.jsonl"

TRADE_FIELDS = ["entry_time", "entry_price", "exit_time", "exit_price", "exit_reason", "pnl_points"]

# Both the replay feed and the run_multi() comparison start from the same
# per-symbol date, WARMUP_TRADING_DAYS trading days before replay_date --
# see the module docstring's "Warm-up parity" note for why this must match
# exactly, not just be "long enough".
WARMUP_TRADING_DAYS = 10


def _pick_replay_date(symbol_data: dict[str, pd.DataFrame]):
    """A date present in EVERY test symbol's cache, with at least
    WARMUP_TRADING_DAYS of cached history before it (for warm-up) and not
    itself any symbol's single most-recent cached day (which could be a
    partially-fetched/incomplete session)."""
    common_dates = None
    for df in symbol_data.values():
        dates = set(df.index.date)
        common_dates = dates if common_dates is None else (common_dates & dates)
    if not common_dates:
        raise RuntimeError("No date is common to all replay symbols' cached data.")
    sorted_dates = sorted(common_dates)
    if len(sorted_dates) < WARMUP_TRADING_DAYS + 2:
        raise RuntimeError(
            f"Only {len(sorted_dates)} common dates -- need at least "
            f"{WARMUP_TRADING_DAYS + 2} for a safe pick with warm-up."
        )
    return sorted_dates[-2]


def _warmup_start_date(df: pd.DataFrame, replay_date):
    """The date WARMUP_TRADING_DAYS trading days before replay_date, in
    THIS symbol's own cached date list -- used identically by both the
    replay feed and the run_multi() comparison (see module docstring)."""
    dates = sorted(set(df.index.date))
    idx = dates.index(replay_date)
    start_idx = max(0, idx - WARMUP_TRADING_DAYS)
    return dates[start_idx]


def main() -> None:
    print(f"Loading cached data for {len(REPLAY_SYMBOLS)} replay symbols...")
    symbol_data = {s: storage.load(s) for s in REPLAY_SYMBOLS}
    symbol_data = {s: df for s, df in symbol_data.items() if not df.empty}
    missing = set(REPLAY_SYMBOLS) - set(symbol_data)
    if missing:
        print(f"WARNING: no cached data for {sorted(missing)} -- excluded from this run.")

    replay_date = _pick_replay_date(symbol_data)
    print(f"Replay date: {replay_date}\n")

    # Fresh journal for this run -- REPLAY_JOURNAL_PATH is a throwaway
    # file, separate from paper_trading/journal.jsonl, so this never
    # pollutes the real trade journal.
    if REPLAY_JOURNAL_PATH.exists():
        REPLAY_JOURNAL_PATH.unlink()
    journal = TradeJournal(path=REPLAY_JOURNAL_PATH)

    watchlist = [
        WatchlistEntry(symbol=s, reason="replay_test", direction="watch", detail="")
        for s in symbol_data
    ]
    # notifier=None -- this is a correctness test, not a live drill; no
    # Telegram/console noise wanted here.
    poller = LivePoller(watchlist, notifier=None, journal=journal, poll_interval_seconds=0)

    print(f"Replaying bar-by-bar through LivePoller.feed_bar() (the exact "
          f"method poll_once() uses live), {WARMUP_TRADING_DAYS} warm-up "
          f"trading days plus the replay day...")
    warmup_start: dict[str, object] = {}
    for symbol, df in symbol_data.items():
        start_date = _warmup_start_date(df, replay_date)
        warmup_start[symbol] = start_date
        replay_window = df[(df.index.date >= start_date) & (df.index.date <= replay_date)]
        if replay_window.empty:
            continue
        for current_index in replay_window.index:
            # Truncate explicitly to current_index on every call, even
            # though on_new_bar() already never reads beyond it internally
            # -- belt-and-suspenders against lookahead bias, matching this
            # project's standing discipline everywhere else.
            poller.feed_bar(symbol, replay_window.loc[:current_index], current_index)
    print("Replay complete.\n")

    # --- Compare journal entries against a direct backtest for the same day ---
    print("Running direct backtest/engine.py run_multi() for comparison...")
    backtest_trades: dict[tuple[str, str], list] = {}
    for symbol, df in symbol_data.items():
        specs = {name: (build(), tfs) for name, (build, tfs) in STRATEGY_BUILDERS.items()}
        start_date = warmup_start[symbol]  # identical per-symbol warm-up window as the replay feed above
        results = run_multi(symbol, specs, start_date=start_date, end_date=replay_date)
        for strategy_name, result in results.items():
            day_trades = [t for t in result.trades if t.entry_time.date() == replay_date]
            backtest_trades[(symbol, strategy_name)] = day_trades

    journal_df = journal.load_all()
    print(f"\nJournal events recorded: {len(journal_df)}\n")
    if not journal_df.empty:
        # The journal now spans the full warm-up window (WARMUP_TRADING_DAYS
        # days plus replay_date), not just replay_date -- must filter down
        # to replay_date's own entries before comparing, same as
        # backtest_trades already does via day_trades above. entry_time is
        # stored as a JSON string (see journal.py's json.dumps(default=str)),
        # so it needs parsing back to a date, not a direct == compare.
        journal_df = journal_df[pd.to_datetime(journal_df["entry_time"]).dt.date == replay_date]

    total_compared = 0
    total_matched = 0
    mismatches = []

    for (symbol, strategy_name), bt_trades in backtest_trades.items():
        if journal_df.empty:
            journal_exits = []
        else:
            journal_exits = journal_df[
                (journal_df["event"] == "trade_exited")
                & (journal_df["symbol"] == symbol)
                & (journal_df["strategy_name"] == strategy_name)
            ].to_dict("records")

        bt_sorted = sorted(bt_trades, key=lambda t: str(t.entry_time))
        journal_sorted = sorted(journal_exits, key=lambda r: str(r["entry_time"]))

        if len(bt_sorted) != len(journal_sorted):
            mismatches.append(
                f"{symbol}/{strategy_name}: trade COUNT differs -- "
                f"backtest={len(bt_sorted)} replay={len(journal_sorted)}"
            )
            total_compared += max(len(bt_sorted), len(journal_sorted))
            continue

        for bt_trade, j_rec in zip(bt_sorted, journal_sorted):
            total_compared += 1
            bt_view = {
                "entry_time": str(bt_trade.entry_time), "entry_price": round(bt_trade.entry_price, 4),
                "exit_time": str(bt_trade.exit_time), "exit_price": round(bt_trade.exit_price, 4),
                "exit_reason": bt_trade.exit_reason, "pnl_points": round(bt_trade.pnl_points, 4),
            }
            j_view = {
                "entry_time": str(j_rec["entry_time"]), "entry_price": round(j_rec["entry_price"], 4),
                "exit_time": str(j_rec["exit_time"]), "exit_price": round(j_rec["exit_price"], 4),
                "exit_reason": j_rec["exit_reason"], "pnl_points": round(j_rec["pnl_points"], 4),
            }
            if bt_view == j_view:
                total_matched += 1
            else:
                mismatches.append(f"{symbol}/{strategy_name}: MISMATCH\n  backtest={bt_view}\n  replay  ={j_view}")

    print("=" * 70)
    print("REPLAY VALIDATION RESULT")
    print("=" * 70)
    print(f"Symbols tested: {len(symbol_data)}")
    print(f"Replay date: {replay_date}")
    print(f"Trades compared: {total_compared}")
    print(f"Matched exactly: {total_matched}")
    print(f"Mismatched: {len(mismatches)}")
    if mismatches:
        print("\nMismatch details:")
        for m in mismatches:
            print(f"  - {m}")
        print("\nVERDICT: FAIL -- do not proceed to live testing until resolved.")
    else:
        print("\nVERDICT: PASS -- PaperTradingEngine/LivePoller reproduces "
              "backtest/engine.py's trades exactly for this replay day.")


if __name__ == "__main__":
    main()
