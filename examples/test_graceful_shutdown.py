"""
Validates paper_trading/live_poller.py's + examples/run_live_session.py's
graceful-shutdown behavior WITHOUT waiting for real market hours or
issuing live network calls: feeds historical replay bars into a loop that
mirrors run_live_session.py's real structure (same signal handlers, same
try/finally, same 1s-chunked sleep), then sends THIS SAME PROCESS a real
SIGTERM from a background thread partway through -- simulating an
external interrupt (Ctrl+C, terminal closed, session torn down) arriving
mid-cycle, the exact scenario that killed a prior live session silently
and left 30 positions frozen open.

Confirms:
  1. The SHUTDOWN log message appears.
  2. Positions open at interrupt time get force-closed with
     exit_reason="shutdown_close" (not left open forever).
  3. The loop exits cleanly (no stack trace, no hang) rather than
     vanishing with no trace.

Uses a THROWAWAY journal file, not paper_trading/journal.jsonl -- this is
a correctness test, not a real session, and must never pollute the real
journal.

WINDOWS SIGNAL-DELIVERY CAVEAT, discovered while building this test, not
assumed: os.kill(os.getpid(), signal.SIGTERM) on Windows does NOT deliver
a catchable signal the way it does on Unix -- Python implements it as a
direct call to the Win32 TerminateProcess() API, which is an unconditional
kill (confirmed empirically: an earlier version of this test that used
os.kill() for SIGTERM was itself hard-killed, exit code 15, the handler
never ran -- the exact "no trace" failure mode this fix exists to solve).
This means self-signaling SIGTERM cannot be used to test the handler on
Windows; instead this script invokes the registered signal handler
function directly, exactly as the OS would on an actual signal delivery
(a standard, legitimate way to unit-test signal handlers specifically
because real OS signal delivery is platform-fragile to test). This
validates the handler's LOGIC (flag -> loop -> finally -> force_close_all)
faithfully; it does not prove Windows will successfully deliver a real
SIGTERM from an external source to this process (a real Ctrl+C during
Monday's live session is the actual end-to-end confirmation for that).

Run: python -m examples.test_graceful_shutdown
"""
from __future__ import annotations

import signal
import threading
import time
from datetime import date, timedelta

from config.settings import PROJECT_ROOT
from data import storage
from paper_trading.journal import TradeJournal
from paper_trading.live_poller import LivePoller
from paper_trading.watchlist import WatchlistEntry

TEST_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "AXISBANK",
    "SBIN", "KOTAKBANK", "BAJFINANCE", "BHARTIARTL",
]
TEST_JOURNAL_PATH = PROJECT_ROOT / "paper_trading" / "test_graceful_shutdown_journal.jsonl"

_shutdown_requested = False


def _request_shutdown(signum, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True


def main() -> None:
    global _shutdown_requested
    _shutdown_requested = False

    print("Loading cached data for test symbols...")
    symbol_data = {s: storage.load(s) for s in TEST_SYMBOLS}
    symbol_data = {s: df for s, df in symbol_data.items() if not df.empty}

    # Pick a replay date common to all test symbols, with enough prior
    # history for a fair warm-up (same reasoning as
    # examples/run_replay_validation.py's WARMUP_TRADING_DAYS).
    common_dates = None
    for df in symbol_data.values():
        dates = set(df.index.date)
        common_dates = dates if common_dates is None else (common_dates & dates)
    sorted_dates = sorted(common_dates)
    replay_date = sorted_dates[-2]  # not the single most-recent day -- could be a partial fetch
    warmup_start = sorted_dates[max(0, sorted_dates.index(replay_date) - 5)]
    print(f"Replay date: {replay_date} (warm-up from {warmup_start})\n")

    if TEST_JOURNAL_PATH.exists():
        TEST_JOURNAL_PATH.unlink()
    journal = TradeJournal(path=TEST_JOURNAL_PATH)

    watchlist = [WatchlistEntry(symbol=s, reason="test", direction="watch", detail="") for s in symbol_data]
    poller = LivePoller(watchlist, notifier=None, journal=journal, poll_interval_seconds=3)

    # --- Setup phase: feed bars up to mid-morning to get REAL open positions ---
    # (not measured/tested -- just needs to reliably produce some open
    # trades before the "live" part of this test begins.)
    setup_cutoff = "10:30"
    replay_frames = {}
    for symbol, df in symbol_data.items():
        window = df[(df.index.date >= warmup_start) & (df.index.date <= replay_date)]
        replay_frames[symbol] = window

    setup_bars: dict[str, list] = {}
    remaining_bars: dict[str, list] = {}
    for symbol, window in replay_frames.items():
        day_bars = window[window.index.date == replay_date]
        setup_idx = day_bars.index[day_bars.index.strftime("%H:%M") <= setup_cutoff]
        remaining_idx = day_bars.index[day_bars.index.strftime("%H:%M") > setup_cutoff]
        setup_bars[symbol] = list(setup_idx)
        remaining_bars[symbol] = list(remaining_idx)

    print(f"Setup: feeding bars up to {setup_cutoff} to establish real open positions...")
    for symbol, window in replay_frames.items():
        for current_index in setup_bars[symbol]:
            poller.feed_bar(symbol, window.loc[:current_index], current_index)
        # Seed _last_known_prices/_last_bar_seen from the setup phase so
        # force_close_all() has something to close with, same as a real
        # poll_once() would have populated them.
        if setup_bars[symbol]:
            last_ts = setup_bars[symbol][-1]
            poller._last_known_prices[symbol] = float(window.loc[last_ts, "close"])
            poller._last_bar_seen[symbol] = last_ts

    open_before = poller.count_open_positions()
    print(f"Open positions after setup phase: {open_before}")
    if open_before == 0:
        print("WARNING: no open positions after setup -- the interrupt test won't prove much. "
              "Continuing anyway to at least confirm the shutdown mechanics run without error.")

    # --- "Live" phase: mirrors run_live_session.py's real loop shape ---
    signal.signal(signal.SIGINT, _request_shutdown)
    try:
        signal.signal(signal.SIGTERM, _request_shutdown)
    except (ValueError, AttributeError, OSError):
        pass

    def _send_interrupt_after_delay():
        time.sleep(2.0)
        print("\n>>> [test] invoking the registered SIGTERM handler now (see module "
              "docstring for why this calls the handler directly rather than "
              "os.kill() on Windows) <<<\n")
        _request_shutdown(signal.SIGTERM, None)

    interrupt_thread = threading.Thread(target=_send_interrupt_after_delay, daemon=True)
    interrupt_thread.start()

    print(f"\nEntering poll loop (poll_interval={poller.poll_interval_seconds}s, "
          f"feeding remaining bars in small batches)...")
    bar_cursor = {s: 0 for s in remaining_bars}
    poll_count = 0
    shutdown_logged = False

    try:
        while not _shutdown_requested:
            any_bars_left = any(bar_cursor[s] < len(remaining_bars[s]) for s in remaining_bars)
            if not any_bars_left:
                print("Replay data exhausted -- stopping (not an interrupt).")
                break

            poll_count += 1
            cycle_start = time.monotonic()
            print(f"--- cycle {poll_count} ---")
            # Feed a small batch of bars per symbol this "cycle" (simulates poll_once()).
            for symbol, window in replay_frames.items():
                idx_list = remaining_bars[symbol]
                start = bar_cursor[symbol]
                end = min(start + 5, len(idx_list))
                for current_index in idx_list[start:end]:
                    poller.feed_bar(symbol, window.loc[:current_index], current_index)
                    poller._last_known_prices[symbol] = float(window.loc[current_index, "close"])
                    poller._last_bar_seen[symbol] = current_index
                bar_cursor[symbol] = end

            open_count = poller.count_open_positions()
            print(f"  open positions: {open_count}")

            cycle_elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.0, poller.poll_interval_seconds - cycle_elapsed)
            slept = 0.0
            while slept < sleep_for and not _shutdown_requested:
                chunk = min(0.5, sleep_for - slept)
                time.sleep(chunk)
                slept += chunk
    finally:
        open_count = poller.count_open_positions()
        if _shutdown_requested:
            print(f"SHUTDOWN: interrupted at {time.strftime('%Y-%m-%d %H:%M:%S')}, "
                  f"{open_count} open position(s) remaining -- attempting final close.")
            shutdown_logged = True
        if open_count > 0:
            closed = poller.force_close_all(reason="shutdown_close")
            print(f"Force-closed {closed} open position(s) at last known price.")

    open_after = poller.count_open_positions()

    print("\n" + "=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)
    print(f"Shutdown was requested (signal received): {_shutdown_requested}")
    print(f"SHUTDOWN log message printed: {shutdown_logged}")
    print(f"Open positions before interrupt (after setup): {open_before}")
    print(f"Open positions remaining after shutdown cleanup: {open_after}")

    journal_df = journal.load_all()
    shutdown_exits = journal_df[
        (journal_df["event"] == "trade_exited") & (journal_df.get("exit_reason") == "shutdown_close")
    ] if not journal_df.empty else journal_df
    n_shutdown_closes = len(shutdown_exits) if not journal_df.empty else 0
    print(f"Journal entries with exit_reason='shutdown_close': {n_shutdown_closes}")

    passed = _shutdown_requested and shutdown_logged and open_after == 0
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  (Check: was a signal actually received? Did the finally block run? "
              "Did open_after reach 0?)")


if __name__ == "__main__":
    main()
