"""
Phase B LIVE session -- OBSERVATION MODE ONLY. Builds today's Phase A
watchlist, then runs the validated LivePoller against it, live, polling
every 60s until market close (or interruption). Every confirmed entry/exit
is journaled (paper_trading/journal.jsonl) AND pushed through
NotificationEngine (Console + FileLog always; Telegram if
TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are set).

This does NOT place real orders. PaperTradingEngine only journals and
notifies what it WOULD have done -- exactly the same fill/exit rules
verified against backtest/engine.py in examples/run_replay_validation.py
(37/37 matched). This script's only new responsibility is sourcing live
bars and driving the poll loop; no trading logic lives here.

Every poll cycle, and the watchlist itself, is logged with timestamps to
logs/live_session_<today>.log via Python's logging module (not just the
journal/notifications, which only cover signals and trades) -- so a
session can be reconstructed and audited afterward even if nothing fired.

Run: python -m examples.run_live_session
   (or examples/run_with_watchdog.py, which runs this AS a subprocess and
   auto-restarts it on a crash/stale heartbeat -- see that script)
Stop: Ctrl+C, SIGTERM (e.g. a normal `kill`/terminal close), a watchdog-
issued CTRL_BREAK_EVENT (see SIGBREAK registration below), or wait for
market close (15:30 IST) -- the loop checks the current time every cycle
and exits on its own if you let it run.

GRACEFUL SHUTDOWN: this session died silently once before -- process torn
down with no graceful shutdown, no EOD square-off, 30 positions left
frozen open in the journal indefinitely. Fixed with three pieces:
  1. SIGINT/SIGTERM/SIGBREAK handlers set a cooperative shutdown flag
     (checked at the top of every loop iteration and during sleep, in ~1s
     increments) instead of relying on Python's default SIGINT->
     KeyboardInterrupt behavior, which a custom handler replaces anyway.
     SIGBREAK (Windows CTRL_BREAK_EVENT) was added 2026-08-19 specifically
     so examples/run_with_watchdog.py can request a real graceful stop
     from a SEPARATE process -- see its registration below for why a real
     SIGTERM isn't deliverable cross-process on Windows at all.
  2. A try/finally around the whole loop ALWAYS runs cleanup -- not just
     on the two "expected" exits (market close, shutdown signal), but on
     any other unexpected exception too, since the whole point is "never
     die silently, no matter why the loop stopped." Cleanup logs a
     SHUTDOWN message and force-closes any still-open positions at their
     last known price (exit_reason="shutdown_close", kept honestly
     distinct from a real "eod_square_off" in the journal).
  3. Separately, LivePoller.poll_once() itself now also enforces EOD
     square-off from actual wall-clock time every cycle (not just each
     bar's own timestamp) -- see paper_trading/live_poller.py's
     _enforce_eod_safeguard() docstring for why the per-bar-only version
     was the root cause of the 30 frozen positions in the first place.
KNOWN LIMITATION: a hard/forced kill (e.g. `taskkill /F` on Windows,
`kill -9` on Unix) bypasses ALL signal handlers on ANY platform -- no
amount of application-level handling can catch that. These fixes cover
graceful termination (Ctrl+C, a normal terminate request, terminal
closing normally); they cannot cover the process being forcibly killed.
"""
from __future__ import annotations

import logging
import os
import signal
import time
from datetime import date, datetime

from config.settings import PROJECT_ROOT, now_ist
from examples.run_morning_watchlist import build_watchlist_parallel
from notifications.notification_engine import ConsoleChannel, FileLogChannel, NotificationEngine
from notifications.send_ops_alert import send_ops_alert
from paper_trading.journal import TradeJournal
from paper_trading.live_poller import MARKET_CLOSE_TIME, LivePoller

POLL_INTERVAL_SECONDS = 60

_shutdown_requested = False


def _request_shutdown(signum, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True


def _build_notifier(log: logging.Logger) -> NotificationEngine:
    channels: list = [ConsoleChannel(), FileLogChannel()]
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        from notifications.notification_engine import TelegramChannel
        channels.append(TelegramChannel(bot_token, chat_id))
        log.info("Telegram configured -- alerts will also be sent to Telegram.")
    else:
        log.info("Telegram NOT configured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set in "
                  "environment) -- alerts will go to Console + FileLog only.")
    return NotificationEngine(channels=channels)


def main() -> None:
    today = date.today()
    log_path = PROJECT_ROOT / "logs" / f"live_session_{today}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("live_session")

    log.info("=" * 70)
    log.info("LIVE SESSION START -- OBSERVATION MODE ONLY, NO REAL EXECUTION")
    log.info("=" * 70)

    log.info("Building Phase A watchlist (parallelized, ~15-20s)...")
    watchlist = build_watchlist_parallel(show_progress=False)
    n_confirmed = sum(1 for w in watchlist if w.reason == "confirmed_pick")
    n_near_level = sum(1 for w in watchlist if w.reason == "near_level")
    log.info(f"Watchlist built: {len(watchlist)} symbols ({n_confirmed} confirmed_pick, "
              f"{n_near_level} near_level)")
    for w in watchlist:
        log.info(f"  WATCHLIST: {w.symbol:12s} [{w.reason:14s}] [{w.direction:5s}] {w.detail}")

    notifier = _build_notifier(log)
    journal = TradeJournal()  # the real journal (paper_trading/journal.jsonl), not a replay/test one
    poller = LivePoller(watchlist, notifier=notifier, journal=journal,
                         poll_interval_seconds=POLL_INTERVAL_SECONDS,
                         # Reads the module global dynamically each call (a plain
                         # lambda closes over the name, not its value at this
                         # point) -- correctly sees _shutdown_requested flip to
                         # True once the signal handlers below are registered
                         # and fire, even though they're registered AFTER this
                         # constructor call. See LivePoller.__init__'s should_stop
                         # docstring for why poll_once() needs this now.
                         should_stop=lambda: _shutdown_requested)

    # Custom SIGINT handler REPLACES Python's default SIGINT->KeyboardInterrupt
    # behavior -- Ctrl+C now sets _shutdown_requested via this handler
    # instead of raising an exception, so shutdown is driven entirely by
    # the cooperative flag check below (one mechanism, not two competing
    # ones). SIGTERM registration can fail on some platforms/environments
    # -- logged, not fatal, since SIGINT handling alone still covers the
    # most common interactive case (Ctrl+C).
    signal.signal(signal.SIGINT, _request_shutdown)
    try:
        signal.signal(signal.SIGTERM, _request_shutdown)
    except (ValueError, AttributeError, OSError) as e:
        log.info(f"SIGTERM handler could not be registered ({e}) -- SIGINT (Ctrl+C) is still "
                 f"handled. Note: a hard/forced kill (taskkill /F, kill -9) bypasses ALL signal "
                 f"handlers on any platform regardless -- these fixes cover graceful termination only.")
    try:
        # SIGBREAK (Windows-only, CTRL_BREAK_EVENT) -- added 2026-08-19 for
        # examples/run_with_watchdog.py: on Windows, os.kill(pid, SIGTERM)
        # and taskkill/Stop-Process all map to a hard TerminateProcess that
        # bypasses every signal handler above (confirmed directly earlier
        # this project -- there is no way to deliver a real SIGTERM on
        # Windows from another process). CTRL_BREAK_EVENT, sent via
        # Popen.send_signal(signal.CTRL_BREAK_EVENT) to a child started
        # with creationflags=CREATE_NEW_PROCESS_GROUP, IS deliverable from
        # another process and IS routed through Python's signal module as
        # SIGBREAK -- but only if something registers a handler for it;
        # unhandled, the OS's default console control action terminates
        # the process just as abruptly as a hard kill. Registering it here,
        # reusing the SAME _request_shutdown callback as SIGINT/SIGTERM, is
        # what lets the watchdog request a REAL graceful shutdown (closes
        # open positions properly, see the finally block below) instead of
        # being forced to hard-kill on every restart.
        signal.signal(signal.SIGBREAK, _request_shutdown)
    except (ValueError, AttributeError, OSError) as e:
        log.info(f"SIGBREAK handler could not be registered ({e}) -- a watchdog-issued "
                 f"CTRL_BREAK_EVENT will not be caught gracefully; Ctrl+C (SIGINT) is still handled.")

    log.info(f"Starting live poll loop: {len(poller.symbols)} watchlist symbols, "
              f"{POLL_INTERVAL_SECONDS}s interval; until {MARKET_CLOSE_TIME} IST or interrupted. "
              f"(Full-universe scanning removed 2026-08-24 -- see paper_trading/live_poller.py's "
              f"module docstring; use the Scanner tab for an on-demand full-universe pass.)")
    send_ops_alert(
        title="NAMEK session started",
        message=f"Watching {len(poller.symbols)} symbols.",
    )

    poll_count = 0
    try:
        while not _shutdown_requested:
            now = now_ist()
            if now.strftime("%H:%M") >= MARKET_CLOSE_TIME:
                log.info("Market closed -- stopping poll loop.")
                break

            poll_count += 1
            cycle_start = time.monotonic()
            log.info(f"--- Poll cycle {poll_count} at {now.strftime('%H:%M:%S')} ---")
            try:
                poller.poll_once()
            except Exception:
                log.exception(f"poll_once() raised an unexpected error in cycle {poll_count}")

            open_count = poller.count_open_positions()
            cycle_elapsed = time.monotonic() - cycle_start
            log.info(f"Cycle {poll_count} done in {cycle_elapsed:.1f}s -- "
                      f"{open_count} open paper positions across {len(poller.symbols)} tracked symbols.")

            # Sleep in ~1s increments, re-checking the shutdown flag each
            # time, instead of one long time.sleep(sleep_for) -- a signal
            # that arrives mid-sleep is noticed within ~1s instead of
            # only at the next full poll_interval_seconds boundary.
            sleep_for = max(0.0, POLL_INTERVAL_SECONDS - cycle_elapsed)
            slept = 0.0
            while slept < sleep_for and not _shutdown_requested:
                chunk = min(1.0, sleep_for - slept)
                time.sleep(chunk)
                slept += chunk
    finally:
        # ALWAYS runs -- on the two expected exits (market close,
        # shutdown signal) AND on any unexpected exception escaping the
        # loop, since the whole point of this fix is "never die silently,
        # no matter why the loop stopped." force_close_all() is a no-op
        # if nothing's open (the normal case: EOD safeguards inside
        # poll_once() already closed everything by market close).
        open_count = poller.count_open_positions()
        if _shutdown_requested:
            log.info(f"SHUTDOWN: interrupted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                      f"{open_count} open position(s) remaining -- attempting final close.")
        elif open_count > 0:
            log.info(f"Loop exited with {open_count} open position(s) still remaining "
                      f"(unexpected -- EOD safeguards should have closed these already) -- "
                      f"attempting final close.")
        if open_count > 0:
            closed = poller.force_close_all(reason="shutdown_close")
            log.info(f"Force-closed {closed} open position(s) at last known price.")

        log.info("=" * 70)
        log.info("LIVE SESSION END")
        log.info("=" * 70)
        summary = journal.summary()
        log.info(f"Session summary (full journal, not just today): {summary}")

        # TODAY's trade count specifically (summary() above is the full,
        # all-time journal) -- for the Telegram end-of-session alert. This
        # finally block runs on EVERY exit path (genuine market close,
        # Ctrl+C, OR a watchdog-triggered restart-kill), so "ended" here
        # means "this process instance stopped," not strictly "15:30 IST
        # reached" -- an intentional, honest simplification: a restart
        # mid-day is still a real end for THIS run, and the trade count
        # is still accurate (it's read from the shared journal file, not
        # this process's own in-memory state).
        try:
            today_str = date.today().isoformat()
            journal_df = journal.load_all()
            trades_today = 0
            if not journal_df.empty and "logged_at" in journal_df.columns:
                trades_today = int((
                    (journal_df["event"] == "trade_entered")
                    & journal_df["logged_at"].astype(str).str.startswith(today_str)
                ).sum())
            send_ops_alert(title="NAMEK session ended", message=f"{trades_today} trade(s) entered today.")
        except Exception as e:
            log.warning(f"Could not compute/send end-of-session alert ({type(e).__name__}: {e}) -- "
                        f"non-fatal, the session itself already ended cleanly above.")


if __name__ == "__main__":
    main()
