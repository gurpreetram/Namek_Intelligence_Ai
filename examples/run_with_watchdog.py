"""
Watchdog for examples/run_live_session.py, added 2026-08-19 for this
project's 4-core/7.8GB-RAM machine, which has repeatedly shown resource-
pressure symptoms (growing cycle times, the system "hanging") when the
backend, live session, and (previously) a separate npm dev server all ran
at once. This script does NOT replace any of the fixes already made for
that (api/main.py serving frontend/dist directly instead of needing npm;
paper_trading/live_poller.py's memory-pressure fixes, most significantly
removing automatic full-universe scanning entirely on 2026-08-24 -- see
that module's docstring) -- it is the next layer: if the live session
dies or goes stale ANYWAY, or the whole machine comes under sustained
pressure, this notices and reacts automatically instead of requiring a
human to catch it (the way the 2026-08-18 "hung session" investigation
had to be done by hand, mid-day, from a description of frozen-looking
symptoms).

WHAT IT DOES, every WATCHDOG_CHECK_INTERVAL_SECONDS (60s):
  1. Reads system-wide RAM/CPU usage via psutil.
  2. Checks whether the run_live_session.py subprocess is still alive AND
     its heartbeat (paper_trading/live_poller.py's HEARTBEAT_PATH) is
     recent -- same staleness definition api/main.py's
     _live_session_is_fresh() uses elsewhere in this project, not a
     second one.
  3. If the process is dead, the heartbeat is stale, OR CPU/RAM has been
     above their thresholds for *_CONSECUTIVE_CHECKS_THRESHOLD checks in
     a row: stops the old subprocess GRACEFULLY (see _stop_gracefully()
     -- this matters a lot, see below), waits RESTART_WAIT_SECONDS, and
     starts a fresh one.
  4. If RAM is above MEMORY_PCT_THRESHOLD at restart time, waits an extra
     MEMORY_GUARD_EXTRA_WAIT_SECONDS first and logs a warning, giving the
     OS a chance to reclaim memory from the just-stopped process before
     spawning a new one.
  Removed 2026-08-24: a "reduce load and restart with a smaller dynamic-
  symbol cap" step used to live here (set NAMEK_MAX_DYNAMIC_SYMBOLS in the
  fresh subprocess's environment under sustained CPU/RAM pressure). That
  mechanism only had meaning while live_poller.py tracked a dynamic,
  growable symbol set on top of the fixed watchlist; now that automatic
  full-universe scanning (and the dynamic set it fed) is gone entirely,
  there is nothing left for it to shrink -- restarting alone is the
  remaining lever.

IMPORTANT, HONEST CONSEQUENCE OF AUTOMATED RESTARTS: every restart --
whether from a dead process, a stale heartbeat, or the CPU guard -- runs
run_live_session.py's normal graceful-shutdown path, which force-closes
ALL open paper positions at their last known price (exit_reason=
"shutdown_close") before the fresh process starts flat and rebuilds its
watchlist from scratch. This is NOT position-preserving -- a session
restarted mid-day loses track of whatever was open and starts over. A
watchdog that restarts often will materially distort today's paper-
trading performance data (many "shutdown_close" exits instead of natural
target/stop/EOD ones). This script does not attempt to solve that (state
reconstruction from the journal on startup is a much bigger feature than
"restart on staleness") -- it only does what was asked: keep a live
session RUNNING, automatically, even if imperfectly, rather than have it
silently sit dead/stale until a human notices.

GRACEFUL SUBPROCESS CONTROL (why this isn't just subprocess.kill()): on
Windows, there is no way to deliver a real SIGTERM to another process --
os.kill()/taskkill/Stop-Process all map to a hard TerminateProcess call
that bypasses every signal handler (confirmed directly earlier this
project when a live session needed a manual restart -- see that
conversation for the full investigation). The one thing that DOES work
cross-process on Windows is CTRL_BREAK_EVENT, sent via Popen.send_signal()
to a child started with creationflags=CREATE_NEW_PROCESS_GROUP, which
Python's signal module surfaces as SIGBREAK -- but only if the child
registers a handler for it. run_live_session.py was updated 2026-08-19
(same day as this script) to do exactly that, reusing its existing
_request_shutdown cooperative-flag mechanism. _stop_gracefully() below
sends CTRL_BREAK_EVENT and waits up to GRACEFUL_STOP_TIMEOUT_SECONDS for
the child to exit on its own (properly closing positions, writing its
SHUTDOWN log line) before escalating to a hard kill as a last resort --
same "graceful first, forced only if truly necessary" pattern
run_live_session.py's own shutdown handling already uses.

SINGLE-INSTANCE GUARD: writes this process's PID to logs/watchdog.pid at
startup. A second watchdog invocation checks that file, confirms the PID
is actually alive AND actually looks like a watchdog (not just a reused
PID belonging to an unrelated process -- see _pid_is_our_watchdog()), and
exits immediately if so. A stale pidfile (process not running, or not
recognizably a watchdog) is treated as leftover, not a lock -- overwritten
and startup proceeds normally.

Run manually: python -m examples.run_with_watchdog
Run via run_all.bat: started in its own console window alongside the
    backend -- see that script.
Run via Task Scheduler: examples/auto_start_live_session.py now calls
    this module's main() instead of run_live_session.py's directly (after
    its own trading-day/start-window checks) -- see that script.
Stop: Ctrl+C in the watchdog's own window (SIGINT), or a CTRL_BREAK_EVENT
    from an outer process that spawned this with creationflags=
    CREATE_NEW_PROCESS_GROUP (SIGBREAK, same mechanism this script uses
    on its own child -- see _stop_gracefully()) -- either way gracefully
    stops the live session subprocess first, then exits. Same Windows
    hard-kill limitation as run_live_session.py applies here too (a
    forced kill of the WATCHDOG doesn't run its cleanup, so the child
    could be orphaned -- if that happens, the orphaned run_live_session.py
    process is still itself perfectly capable of running normally / being
    Ctrl+C'd directly in its own console, exactly as before this script
    existed).
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import date, datetime

import psutil

from config.settings import MARKET_CLOSE, PROJECT_ROOT, is_trading_day, now_ist
from notifications.send_ops_alert import send_ops_alert
from paper_trading.live_poller import HEARTBEAT_PATH

WATCHDOG_CHECK_INTERVAL_SECONDS = 60
HEARTBEAT_STALE_THRESHOLD_SECONDS = 120
# Generous on purpose: building today's watchlist + the first poll cycle
# has taken 400-500s+ in real observed sessions (from the era before
# 2026-08-24's removal of automatic full-universe scanning -- a fixed
# ~50-symbol watchlist alone should be considerably faster, but this
# grace period is deliberately still sized for the historical worst case
# rather than re-tuned down without new real timing data), and a fresh
# subprocess writes NO heartbeat at all until
# its first poll_once() completes. Checking staleness before this grace
# period would restart a perfectly healthy, still-starting session. Used
# ONLY for the very first spawn of the day -- see RESTART_HEARTBEAT_
# STARTUP_GRACE_SECONDS below for every subsequent restart.
HEARTBEAT_STARTUP_GRACE_SECONDS = 600
# Added 2026-08-20: a restart still has to pay the SAME watchlist-build
# cost as a first spawn (examples/run_live_session.py's main() always
# calls build_watchlist_parallel() fresh -- there is no "reuse the
# previous instance's watchlist" path), so this is NOT actually as safe
# as "the session already has a watchlist cached" might suggest -- it is
# shorter than HEARTBEAT_STARTUP_GRACE_SECONDS specifically to shrink the
# window a genuinely stuck restart can hide in, not because a restart is
# cheaper. If a restart's first poll cycle happens to be slow (e.g. it
# lands on a full-universe scan, or hits network issues), this CAN still
# produce a false "stale" read before the new session's first heartbeat
# lands -- a real, known tradeoff of prioritizing a faster stale-restart
# loop over zero false positives, not an oversight. See this module's
# validation notes for the real observed restart timing.
RESTART_HEARTBEAT_STARTUP_GRACE_SECONDS = 120
# Reduced from 10 to 2 on 2026-08-20 (was per-spec "wait 10 seconds", now
# "start immediately") -- a small non-zero buffer, not literally 0s,
# still gives Windows a moment to fully release the old process's port/
# file handles before the new one starts, without meaningfully adding to
# the stale-session window.
RESTART_WAIT_SECONDS = 2
# How long to wait for CTRL_BREAK_EVENT to produce a clean child exit
# before escalating to a hard kill. Raised from 30 to 50 on 2026-08-19
# after real validation: paper_trading/live_poller.py's poll_once() now
# checks the shutdown flag between EACH symbol (confirmed correct and
# near-instant in isolation -- a unit-level test breaks after exactly 2
# should_stop() checks, 0.04s), but a SINGLE symbol's blocking fetch_
# intraday_1min() call can itself still take up to ~14s worst-case if it
# hits Upstox rate-limiting and burns through data/fetcher.py's MAX_
# RETRIES=4 growing 2/4/6/8s backoff before the loop can even return to
# check the flag again -- confirmed as the real cause of a 30s timeout in
# testing (not a bug in the per-symbol check itself, which passed its own
# isolated test cleanly). Two such symbols in a row under real system
# load can approach 30s alone. 50s gives real margin above that while
# still comfortably inside the ~60-500s+ a full UNINTERRUPTED cycle used
# to take before any of today's fixes.
GRACEFUL_STOP_TIMEOUT_SECONDS = 50
MEMORY_PCT_THRESHOLD = 85.0
# Reduced from 30 to 10 on 2026-08-20 -- real measured RAM at 93.4% showed
# this delay was a meaningful fraction of the total stale-session window;
# 10s still gives the OS a real (if smaller) chance to reclaim the just-
# stopped process's memory before the new one starts.
MEMORY_GUARD_EXTRA_WAIT_SECONDS = 10
# Added 2026-08-19: RAM > MEMORY_PCT_THRESHOLD for MEMORY_CONSECUTIVE_
# CHECKS_THRESHOLD consecutive checks is now its OWN restart trigger (not
# just a pre-restart delay tacked onto an already-decided restart, which
# is all it was before) -- real validation the same day measured this
# machine at 91% RAM under normal multi-process load, well past a safe
# margin. Requires 3 consecutive high readings, same reasoning as the CPU
# guard: a single brief spike shouldn't force a resource-intensive
# restart. MEMORY_GUARD_EXTRA_WAIT_SECONDS above still separately applies
# as a delay whenever ANY restart happens while RAM is high, regardless
# of which condition actually triggered that restart.
MEMORY_CONSECUTIVE_CHECKS_THRESHOLD = 3
CPU_PCT_THRESHOLD = 90.0
CPU_CONSECUTIVE_CHECKS_THRESHOLD = 3
# Added 2026-08-19: if MAX_CONSECUTIVE_FAILED_RESTARTS restarts in a row
# each fail to produce a healthy session (dead/stale/CPU-tripped/mem-
# tripped again before ANY intervening healthy check), the watchdog gives
# up -- sends a high-urgency Telegram alert and stops the loop entirely,
# rather than burning CPU/network/RAM in an endless restart cycle against
# a problem restarting clearly isn't fixing (a genuinely broken network
# path, a corrupted cache, Upstox itself down, etc.). "Failed" is
# deliberately simple: the counter increments every time a restart is
# TRIGGERED and resets to 0 on any single healthy check -- so 3 restarts
# with no healthy check between any of them counts, however many actual
# minutes that spans.
MAX_CONSECUTIVE_FAILED_RESTARTS = 3

PID_FILE = PROJECT_ROOT / "logs" / "watchdog.pid"
LOG_FILE = PROJECT_ROOT / "logs" / "watchdog.log"

_shutdown_requested = False


def _request_shutdown(signum, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True


def _setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("watchdog")


def _pid_is_our_watchdog(pid: int) -> bool:
    """True if `pid` is a live process that looks like a run_with_watchdog
    invocation -- guards against the rare PID-reuse case where a stale
    pidfile's PID has since been reassigned to an unrelated process,
    which would otherwise falsely block every future watchdog start."""
    if not psutil.pid_exists(pid):
        return False
    try:
        cmdline = " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return "run_with_watchdog" in cmdline


def _check_single_instance(log: logging.Logger) -> bool:
    """True if it's safe to proceed (no other live watchdog holds the
    PID file); False if this process should exit immediately instead."""
    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            existing_pid = None
        if existing_pid is not None and _pid_is_our_watchdog(existing_pid):
            log.error(f"Another watchdog is already running (PID {existing_pid}, per {PID_FILE}) -- "
                      f"exiting immediately to avoid a double-watchdog scenario (two watchdogs would "
                      f"each try to manage their own run_live_session.py subprocess independently).")
            return False
        log.info(f"Found {PID_FILE} but PID {existing_pid} is not a live watchdog -- treating as "
                 f"stale and continuing.")
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    return True


def _release_pid_file(log: logging.Logger) -> None:
    try:
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()
    except OSError as e:
        log.warning(f"Could not remove {PID_FILE} on exit ({e}) -- next startup will find it stale "
                    f"(PID {os.getpid()} won't be running) and clean it up itself.")


def _heartbeat_timestamp() -> datetime | None:
    """Raw parsed heartbeat timestamp (naive local datetime.now(), matching
    exactly what live_poller.py's _write_heartbeat() writes -- not
    now_ist()/aware). None if the file is missing/unparseable OR its date
    isn't today (a leftover heartbeat from yesterday should read as
    infinitely stale, not compared numerically)."""
    if not HEARTBEAT_PATH.exists():
        return None
    try:
        payload = json.loads(HEARTBEAT_PATH.read_text())
        last_heartbeat = datetime.fromisoformat(payload["timestamp"])
    except (OSError, ValueError, KeyError):
        return None
    if last_heartbeat.date() != date.today():
        return None
    return last_heartbeat


def _heartbeat_age_seconds() -> float | None:
    """Same definition as api/main.py's _live_session_is_fresh(). None if
    there's no usable (today-dated, parsable) heartbeat at all."""
    ts = _heartbeat_timestamp()
    if ts is None:
        return None
    return (datetime.now() - ts).total_seconds()


def _heartbeat_symbols_polled() -> int | None:
    """Reads the symbols_polled field straight from the heartbeat file --
    used by the post-restart confirmation below so the "Session restored"
    alert reports a real, current number instead of guessing."""
    if not HEARTBEAT_PATH.exists():
        return None
    try:
        payload = json.loads(HEARTBEAT_PATH.read_text())
        return int(payload["symbols_polled"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


# Added 2026-08-20 (FIX 2d): after spawning a RESTART (not the initial
# spawn of the day -- see _run()'s call site), poll the heartbeat file at
# this cadence, up to POST_RESTART_CONFIRM_TIMEOUT_SECONDS, waiting for it
# to show a timestamp newer than the restart itself -- i.e., proof the new
# process actually completed a real poll_once() cycle, not just that it
# started. Only once confirmed does the "Session restored" alert fire, so
# the user's phone notification means what it says instead of firing the
# instant the process is merely spawned (which could still fail before
# ever producing a heartbeat).
POST_RESTART_CONFIRM_POLL_SECONDS = 10
# Bounded generously -- a restart pays the full watchlist-build cost
# again (see RESTART_HEARTBEAT_STARTUP_GRACE_SECONDS's docstring), so
# this needs real headroom above that, not just above the grace period
# itself. Timing out here does NOT alert a failure -- the main loop's own
# staleness check (using RESTART_HEARTBEAT_STARTUP_GRACE_SECONDS) is what
# catches a restart that never actually recovers; this confirmation is
# purely for the POSITIVE "it's back" signal, silence on timeout is
# intentional (the normal check loop already covers the negative case).
POST_RESTART_CONFIRM_TIMEOUT_SECONDS = 420


def _confirm_restored_and_alert(log: logging.Logger, spawned_at_wall_clock: datetime) -> None:
    """spawned_at_wall_clock: datetime.now() captured right when the NEW
    subprocess was spawned. REQUIRES the heartbeat's own timestamp to be
    AFTER this moment, not merely "recent" -- fixed 2026-08-26 after a
    real false-positive caught in live operation: the OLD (about-to-be-
    replaced) process's own last heartbeat, written seconds before the
    restart decision, was still well within HEARTBEAT_STALE_THRESHOLD_
    SECONDS by the time this ran, so the previous version (comparing only
    generic staleness, not "is this newer than the restart") declared
    "Restart confirmed" and sent the Telegram alert while the NEW process
    was still building its watchlist, seconds after spawning -- a stale
    signal reported as a fresh confirmation. Comparing the heartbeat's OWN
    timestamp directly against spawned_at_wall_clock is unambiguous: only
    a heartbeat the NEW process actually wrote can satisfy it."""
    deadline = time.monotonic() + POST_RESTART_CONFIRM_TIMEOUT_SECONDS
    while time.monotonic() < deadline and not _shutdown_requested:
        ts = _heartbeat_timestamp()
        if ts is not None and ts > spawned_at_wall_clock:
            symbols = _heartbeat_symbols_polled()
            age = (datetime.now() - ts).total_seconds()
            log.info(f"Restart confirmed -- heartbeat from {ts.isoformat()} ({age:.0f}s old, "
                     f"written after the {spawned_at_wall_clock.isoformat()} spawn), {symbols} symbols.")
            send_ops_alert(
                title="NAMEK session restored",
                message=f"✅ Session restored. Watching {symbols if symbols is not None else '?'} symbols.",
            )
            return
        _sleep_interruptible(POST_RESTART_CONFIRM_POLL_SECONDS)
    if not _shutdown_requested:
        log.warning(f"Restart NOT confirmed within {POST_RESTART_CONFIRM_TIMEOUT_SECONDS}s -- no "
                    f"'Session restored' alert sent. The main check loop's own staleness detection "
                    f"(using RESTART_HEARTBEAT_STARTUP_GRACE_SECONDS) will catch this on its next tick "
                    f"if the session genuinely hasn't recovered.")


def _spawn_live_session(log: logging.Logger) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "examples.run_live_session"],
        cwd=str(PROJECT_ROOT),
        # Required for _stop_gracefully()'s CTRL_BREAK_EVENT to target
        # ONLY this child (and not also hit the watchdog itself, which
        # shares this console otherwise) -- see this module's docstring.
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    log.info(f"Spawned run_live_session.py (PID {proc.pid}).")
    return proc


def _stop_gracefully(proc: subprocess.Popen, log: logging.Logger, reason: str) -> None:
    if proc.poll() is not None:
        log.info(f"Subprocess (PID {proc.pid}) had already exited (code {proc.returncode}) -- nothing to stop.")
        return
    log.info(f"Stopping subprocess (PID {proc.pid}) gracefully -- reason: {reason}. Sending CTRL_BREAK_EVENT...")
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    except (OSError, ValueError) as e:
        log.warning(f"CTRL_BREAK_EVENT failed ({e}) -- escalating straight to terminate().")
        proc.terminate()
    try:
        proc.wait(timeout=GRACEFUL_STOP_TIMEOUT_SECONDS)
        log.info(f"Subprocess (PID {proc.pid}) exited gracefully (code {proc.returncode}).")
        return
    except subprocess.TimeoutExpired:
        pass
    log.warning(f"Subprocess (PID {proc.pid}) did not exit within {GRACEFUL_STOP_TIMEOUT_SECONDS}s of "
                f"CTRL_BREAK_EVENT -- escalating to a hard kill. Any open paper positions will NOT be "
                f"closed cleanly by this stop (same Windows hard-kill limitation run_live_session.py's "
                f"own docstring documents).")
    proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        log.error(f"Subprocess (PID {proc.pid}) still not confirmed dead 10s after kill() -- giving up "
                  f"waiting and proceeding anyway.")


def _sleep_interruptible(seconds: float) -> None:
    slept = 0.0
    while slept < seconds and not _shutdown_requested:
        chunk = min(1.0, seconds - slept)
        time.sleep(chunk)
        slept += chunk


def _run(log: logging.Logger) -> bool:
    """Returns True if the watchdog gave up after MAX_CONSECUTIVE_FAILED_
    RESTARTS (caller should exit nonzero -- see main()), False on every
    other exit path (market close, shutdown signal -- both normal)."""
    today = now_ist().date()
    if not is_trading_day(today):
        log.info(f"{today} is not an NSE trading day (weekend or holiday) -- exiting immediately, "
                 f"nothing started.")
        return False

    if now_ist().strftime("%H:%M") >= MARKET_CLOSE:
        log.info(f"Current time is already past market close ({MARKET_CLOSE} IST) -- exiting "
                 f"immediately without starting anything.")
        return False

    consecutive_high_cpu = 0
    consecutive_high_mem = 0
    consecutive_failed_restarts = 0
    gave_up = False
    # First spawn of the day gets the full 600s grace (see that constant's
    # docstring); every restart after that uses the shorter RESTART_
    # HEARTBEAT_STARTUP_GRACE_SECONDS (120s) -- FIX 4.
    current_startup_grace_seconds = HEARTBEAT_STARTUP_GRACE_SECONDS

    proc = _spawn_live_session(log)
    spawned_at = time.monotonic()

    try:
        while not _shutdown_requested:
            _sleep_interruptible(WATCHDOG_CHECK_INTERVAL_SECONDS)
            if _shutdown_requested:
                break

            now = now_ist()
            if now.strftime("%H:%M") >= MARKET_CLOSE:
                log.info(f"Market closed ({MARKET_CLOSE} IST) -- stopping watchdog and live session.")
                break

            mem_pct = psutil.virtual_memory().percent
            cpu_pct = psutil.cpu_percent(interval=None)  # since the LAST call -- non-blocking
            child_alive = proc.poll() is None
            heartbeat_age = _heartbeat_age_seconds()
            elapsed_since_spawn = time.monotonic() - spawned_at

            if cpu_pct > CPU_PCT_THRESHOLD:
                consecutive_high_cpu += 1
            else:
                consecutive_high_cpu = 0
            if mem_pct > MEMORY_PCT_THRESHOLD:
                consecutive_high_mem += 1
            else:
                consecutive_high_mem = 0

            log.info(f"Check: mem={mem_pct:.1f}% cpu={cpu_pct:.1f}% child_alive={child_alive} "
                     f"heartbeat_age={'n/a' if heartbeat_age is None else f'{heartbeat_age:.0f}s'} "
                     f"elapsed_since_spawn={elapsed_since_spawn:.0f}s "
                     f"consecutive_high_cpu={consecutive_high_cpu} consecutive_high_mem={consecutive_high_mem} "
                     f"consecutive_failed_restarts={consecutive_failed_restarts}")

            dead = not child_alive
            past_startup_grace = elapsed_since_spawn > current_startup_grace_seconds
            stale = past_startup_grace and (heartbeat_age is None or heartbeat_age > HEARTBEAT_STALE_THRESHOLD_SECONDS)
            cpu_tripped = consecutive_high_cpu >= CPU_CONSECUTIVE_CHECKS_THRESHOLD
            mem_tripped = consecutive_high_mem >= MEMORY_CONSECUTIVE_CHECKS_THRESHOLD

            # consecutive_failed_restarts is reset on a RAW single-reading
            # "is this ACTUALLY fine right now" check (mem_pct/cpu_pct
            # compared directly against threshold), deliberately NOT on
            # cpu_tripped/mem_tripped (the multi-check accumulated flags).
            # Those two get reset to 0 on every restart (see below), so
            # tying the give-up counter's reset to THEM would let sustained
            # pressure hide behind that reset window forever: each restart
            # buys ~MEMORY_CONSECUTIVE_CHECKS_THRESHOLD checks of "not
            # tripped yet" even though the underlying pressure never
            # actually went away, so consecutive_failed_restarts would keep
            # getting reset before ever reaching MAX_CONSECUTIVE_FAILED_
            # RESTARTS -- confirmed as a REAL bug via an isolated test
            # (sustained 91% RAM restarted forever, never gave up) before
            # this fix, not a hypothetical concern.
            this_check_actually_healthy = (
                not dead and not stale
                and mem_pct <= MEMORY_PCT_THRESHOLD and cpu_pct <= CPU_PCT_THRESHOLD
            )
            if this_check_actually_healthy:
                consecutive_failed_restarts = 0

            if not (dead or stale or cpu_tripped or mem_tripped):
                continue

            if dead:
                reason = f"process died (exit code {proc.returncode})"
            elif stale:
                age_desc = "no heartbeat file for today" if heartbeat_age is None else f"heartbeat {heartbeat_age:.0f}s old"
                reason = f"{age_desc} (threshold {HEARTBEAT_STALE_THRESHOLD_SECONDS}s, past {current_startup_grace_seconds:.0f}s startup grace)"
            elif cpu_tripped:
                reason = f"sustained high CPU ({consecutive_high_cpu} consecutive checks > {CPU_PCT_THRESHOLD:.0f}%)"
            else:
                reason = f"sustained high RAM ({consecutive_high_mem} consecutive checks > {MEMORY_PCT_THRESHOLD:.0f}%)"

            if mem_pct > MEMORY_PCT_THRESHOLD:
                log.warning(f"Memory guard: system RAM at {mem_pct:.1f}% (> {MEMORY_PCT_THRESHOLD:.0f}%) "
                            f"-- waiting an extra {MEMORY_GUARD_EXTRA_WAIT_SECONDS}s before restarting, "
                            f"to give the OS time to free memory.")
                _sleep_interruptible(MEMORY_GUARD_EXTRA_WAIT_SECONDS)
                if _shutdown_requested:
                    break

            consecutive_failed_restarts += 1
            if consecutive_failed_restarts >= MAX_CONSECUTIVE_FAILED_RESTARTS:
                log.error(f"GIVING UP -- {consecutive_failed_restarts} consecutive restarts in a row "
                          f"with no healthy check in between (latest reason: {reason}). Restarting "
                          f"clearly isn't fixing this -- stopping instead of burning resources in an "
                          f"endless loop. Manual intervention needed.")
                send_ops_alert(
                    title="NAMEK live session watchdog giving up",
                    message=(f"{consecutive_failed_restarts} consecutive restarts all failed "
                              f"(latest reason: {reason}). Stopped retrying to avoid burning resources. "
                              f"The live session is NOT running and will NOT be auto-restarted again "
                              f"today -- needs manual attention."),
                    urgency="high",
                )
                gave_up = True
                _stop_gracefully(proc, log, reason="giving_up_after_failed_restarts")
                break

            log.warning(f"RESTARTING live session -- reason: {reason}")
            # FIX 2c: sent BEFORE the stop/restart, not after, so the user
            # knows a gap is coming and roughly why -- "Back in ~60s" is a
            # rough estimate for the common case, not a guarantee (see
            # RESTART_HEARTBEAT_STARTUP_GRACE_SECONDS's docstring: a
            # restart still pays the full watchlist-build cost, which by
            # itself has been observed at 30-40s, before the first poll
            # cycle even starts).
            send_ops_alert(
                title="NAMEK live session restarting",
                message=f"⚠️ Restarting session (RAM: {mem_pct:.1f}%). Back in ~60s. Reason: {reason}",
            )
            _stop_gracefully(proc, log, reason=reason)
            _sleep_interruptible(RESTART_WAIT_SECONDS)
            if _shutdown_requested:
                break
            spawned_at_wall_clock = datetime.now()  # for _confirm_restored_and_alert() -- see its docstring
            proc = _spawn_live_session(log)
            spawned_at = time.monotonic()
            current_startup_grace_seconds = RESTART_HEARTBEAT_STARTUP_GRACE_SECONDS
            consecutive_high_cpu = 0
            consecutive_high_mem = 0
            # FIX 2d: confirms via the heartbeat itself (not just "the
            # process started") before telling the user it's back -- see
            # _confirm_restored_and_alert()'s docstring. Bounded, so this
            # can't hang the watchdog's own loop indefinitely if the
            # restart never actually recovers.
            _confirm_restored_and_alert(log, spawned_at_wall_clock=spawned_at_wall_clock)
    finally:
        if _shutdown_requested:
            log.info("Watchdog interrupted -- stopping live session gracefully before exiting.")
        if not gave_up:  # _stop_gracefully() already called in the give-up path above
            _stop_gracefully(proc, log, reason="watchdog_shutdown")
        log.info("Watchdog exiting.")
    return gave_up


def main() -> None:
    log = _setup_logging()
    signal.signal(signal.SIGINT, _request_shutdown)
    try:
        signal.signal(signal.SIGTERM, _request_shutdown)
    except (ValueError, AttributeError, OSError):
        pass  # best-effort, same as run_live_session.py -- SIGINT (Ctrl+C) alone still covers the interactive case
    try:
        # SIGBREAK, same reasoning as run_live_session.py's own registration
        # (see that module's docstring) -- lets the WATCHDOG itself be
        # stopped gracefully via CTRL_BREAK_EVENT by an outer process that
        # spawned it with creationflags=CREATE_NEW_PROCESS_GROUP, not just
        # by an interactive Ctrl+C in its own console window. Confirmed
        # necessary, not speculative: an earlier validation run of this
        # script sent CTRL_BREAK_EVENT to a watchdog instance without this
        # handler registered, which fell through to Windows' default
        # console-control action (abrupt termination) instead of the
        # cooperative flag below -- the finally block's _stop_gracefully()
        # call on the CHILD never ran, orphaning it. Registering this
        # closes that gap for any future external caller, not just this
        # one test.
        signal.signal(signal.SIGBREAK, _request_shutdown)
    except (ValueError, AttributeError, OSError):
        pass

    if not _check_single_instance(log):
        sys.exit(1)
    gave_up = False
    try:
        gave_up = _run(log)
    finally:
        _release_pid_file(log)
    if gave_up:
        # Nonzero so run_auto_start.bat's errorlevel check ALSO fires an
        # alert (harmless double-alert -- see that .bat's comment) and so
        # Task Scheduler's own LastTaskResult correctly reflects that this
        # run did not end normally.
        sys.exit(1)


if __name__ == "__main__":
    main()
