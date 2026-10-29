"""
Automatic daily launcher for the live trading session -- lets Windows Task
Scheduler (or any other daily trigger) start the session unattended every
trading day, without a human manually typing the run command each morning.

As of 2026-08-19, this launches examples/run_with_watchdog.py (which
starts examples/run_live_session.py itself as a subprocess and auto-
restarts it on a crash/stale heartbeat or sustained resource pressure),
not run_live_session.py directly -- see that module's docstring. The two
guards below are UNCHANGED and still run first, deciding whether to start
anything at all today; they are specific to "should an unattended daily
trigger fire right now" and are independent of the watchdog's own (later,
narrower) is_trading_day() check.

Two guards, BOTH required before the live session actually starts:
  1. Trading day check (config.settings.is_trading_day()) -- skips
     weekends and NSE holidays. Reuses the same verified holiday
     calendar every other market-hours check in this project already
     uses, not a second, potentially-diverging one.
  2. Start-window check -- only starts if invoked within
     START_WINDOW_START..START_WINDOW_END IST (default 09:14-09:20).
     This exists specifically so a LATE-firing trigger (Task Scheduler
     woke up late, the laptop was asleep and just resumed well after
     09:15, etc.) doesn't silently start the session hours into the
     trading day against a stale/partial watchlist -- if the window has
     already passed, this exits without starting, same as "not a
     trading day." It deliberately does NOT just check "is the market
     currently open" (09:15-15:30) -- that would happily start a fresh
     session at, say, 13:00 off a late trigger, which is exactly the
     silently-wrong behavior this guards against.

HONEST LIMITATION, not solved by this script OR Task Scheduler: none of
this fires at all if the laptop itself is asleep at 09:15 IST. Confirmed
directly earlier the same day this script was built: a real Windows
sleep/resume event killed an already-running live session mid-day with
no exception, no crash record (see paper_trading/live_poller.py's
history for that incident) -- the same sleep behavior would just as
easily prevent THIS launcher from ever running at all if the laptop is
asleep at trigger time; Task Scheduler cannot wake a sleeping machine on
its own (that requires a separate "wake timer" configuration, which is
an OS/BIOS-level power setting, not something this script or a scheduled
task can do from inside Windows). This script removes the need to
manually type the run command every morning; it does NOT make the
session survive (or even start during) a sleeping machine -- the user is
still responsible for keeping the laptop powered on and awake through
market hours.

SEPARATE REMINDER printed every time this actually starts a session (not
just in this docstring): the dashboard BACKEND should run WITHOUT
--reload on a live trading day (run_backend_live.bat), not with it
(run_backend_dev.bat) -- see README.md's "Running the backend" section.
This script does not start or manage the backend itself; the printed
reminder is for whoever's watching this log.

Run manually: python -m examples.auto_start_live_session
    python -m examples.auto_start_live_session --force   # skip the
        trading-day/start-window checks, for manual testing outside
        09:14-09:20 or on a non-trading day
Run via Task Scheduler: see the project's setup notes for the exact
`schtasks /create` command -- triggered daily at 09:15 IST, pointing at
this project's venv Python interpreter running this script as a module.
"""
from __future__ import annotations

import argparse
import sys

from config.settings import is_trading_day, now_ist

START_WINDOW_START = "09:14"
START_WINDOW_END = "09:20"


def _within_start_window(now) -> bool:
    current = now.strftime("%H:%M")
    return START_WINDOW_START <= current <= START_WINDOW_END


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the trading-day and start-window checks and start the live session immediately "
             "regardless of date/time -- for manual testing only, never for the scheduled task.",
    )
    args = parser.parse_args()

    now = now_ist()
    print(f"[auto_start_live_session] Checked at {now.isoformat()}")

    if not args.force:
        if not is_trading_day(now.date()):
            print(f"[auto_start_live_session] {now.date()} is not an NSE trading day "
                  f"(weekend or holiday) -- exiting without starting anything.")
            sys.exit(0)

        if not _within_start_window(now):
            print(f"[auto_start_live_session] Current time {now.strftime('%H:%M')} IST is outside "
                  f"the {START_WINDOW_START}-{START_WINDOW_END} start window -- exiting without "
                  f"starting anything (this is what stops a late-firing trigger from starting the "
                  f"session hours into the day off a stale watchlist).")
            sys.exit(0)

        print(f"[auto_start_live_session] {now.date()} is a trading day and "
              f"{now.strftime('%H:%M')} IST is within the start window -- starting live session.")
    else:
        print("[auto_start_live_session] --force passed -- skipping trading-day/start-window checks.")

    print("[auto_start_live_session] REMINDER: on a live trading day, run the dashboard backend "
          "WITHOUT --reload (run_backend_live.bat / `uvicorn api.main:app --port 8000`), not "
          "run_backend_dev.bat. Three separate backend freezes were traced back to --reload's file "
          "watcher (a sleep-interrupted reload state, a CPU-starved reload event loop, and an "
          "I/O-bound StatReload sweep of storage/'s ~7,100+ cached files) -- see README.md's "
          "'Running the backend' section. This script does not start or manage the backend itself "
          "(it only starts the live trading session) -- this is a reminder for whoever's watching "
          "this log, not an automated check.")

    # Runs the WATCHDOG (examples/run_with_watchdog.py), not run_live_
    # session.py directly, as of 2026-08-19 -- the watchdog starts
    # run_live_session.py itself as a subprocess and auto-restarts it if
    # it dies/goes stale or the machine comes under sustained resource
    # pressure, none of which this launcher's own trading-day/start-
    # window checks (above) have any way to detect. Those checks still
    # matter and still run first: they decide WHETHER to start anything
    # at all today; the watchdog's own is_trading_day() check is a second,
    # defensive check for the case where it's invoked some other way
    # (e.g. run_all.bat, run manually) that skips this launcher entirely.
    from examples.run_with_watchdog import main as run_watchdog_main
    run_watchdog_main()


if __name__ == "__main__":
    main()
