"""
Watchdog for the FastAPI backend (api/main.py, served via uvicorn) --
the backend-side counterpart to examples/run_with_watchdog.py's live-
session watchdog, added 2026-08-19 as part of making the whole system
self-healing enough to run unattended (see that module's docstring for
the broader context: this machine has repeatedly shown resource-pressure
symptoms, and previously nothing restarted a crashed/frozen backend at
all).

Much simpler than the live-session watchdog, deliberately: the backend is
a STATELESS read API (data/storage.py's local cache in, JSON responses
out) with no open-position/journal-integrity concerns the way a live
trading session has -- there is nothing here that needs a graceful
CTRL_BREAK_EVENT dance before stopping. A plain terminate()/kill() is
fine and simpler.

WHAT IT DOES, every BACKEND_CHECK_INTERVAL_SECONDS (60s):
  1. GETs http://localhost:{BACKEND_PORT}/health.
  2. If that fails (non-200, timeout, connection refused) BACKEND_
     FAILURE_THRESHOLD (3) times in a row, OR the uvicorn subprocess has
     died outright: kills it and starts a fresh one, logs the restart to
     logs/backend_watchdog.log, and sends a Telegram alert (see
     notifications/send_ops_alert.py) -- same "every restart gets
     announced" convention the live-session watchdog uses.
  3. If MAX_CONSECUTIVE_FAILED_RESTARTS restarts in a row all still fail
     health checks (with no healthy check in between), gives up: sends a
     high-urgency alert and stops, rather than looping forever -- same
     give-up semantics as run_with_watchdog.py's _run(), same reasoning
     (a backend that won't come back up after several restarts has a
     real problem restarting alone won't fix -- port already in use by
     something else, a code error on import, etc.).

SINGLE-INSTANCE GUARD: same PID-file pattern as run_with_watchdog.py,
using its own separate logs/backend_watchdog.pid so the two watchdogs
never interfere with each other's locks.

Run manually: python -m examples.run_backend_watchdog
Run via run_all.bat: started in its own console window alongside the
    live-session watchdog -- see that script. Replaces starting uvicorn
    directly.
Stop: Ctrl+C in its own console window (SIGINT), or CTRL_BREAK_EVENT from
    an outer process that spawned it with creationflags=
    CREATE_NEW_PROCESS_GROUP (SIGBREAK) -- either way stops the backend
    subprocess (a plain terminate(), no graceful-shutdown complexity
    needed -- see module docstring) before exiting.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time

import psutil
import requests

from config.settings import PROJECT_ROOT
from notifications.send_ops_alert import send_ops_alert

BACKEND_PORT = 8000
BACKEND_CHECK_INTERVAL_SECONDS = 60
BACKEND_FAILURE_THRESHOLD = 3  # consecutive /health failures before restarting
HEALTH_CHECK_TIMEOUT_SECONDS = 10
# Generous startup grace before the FIRST health check counts against the
# failure threshold -- uvicorn + FastAPI's own startup (module imports,
# route registration) is fast (a few seconds) but not instant, and a
# check landing before the server's even listening shouldn't count as a
# "failure."
STARTUP_GRACE_SECONDS = 30
STOP_TIMEOUT_SECONDS = 15  # terminate() -> wait -> escalate to kill()
MAX_CONSECUTIVE_FAILED_RESTARTS = 3  # same give-up semantics as run_with_watchdog.py

PID_FILE = PROJECT_ROOT / "logs" / "backend_watchdog.pid"
LOG_FILE = PROJECT_ROOT / "logs" / "backend_watchdog.log"

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
    return logging.getLogger("backend_watchdog")


def _pid_is_our_watchdog(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return False
    try:
        cmdline = " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return "run_backend_watchdog" in cmdline


def _check_single_instance(log: logging.Logger) -> bool:
    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            existing_pid = None
        if existing_pid is not None and _pid_is_our_watchdog(existing_pid):
            log.error(f"Another backend watchdog is already running (PID {existing_pid}, per {PID_FILE}) "
                      f"-- exiting immediately to avoid two uvicorn instances fighting over port "
                      f"{BACKEND_PORT}.")
            return False
        log.info(f"Found {PID_FILE} but PID {existing_pid} is not a live backend watchdog -- treating "
                 f"as stale and continuing.")
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    return True


def _release_pid_file(log: logging.Logger) -> None:
    try:
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()
    except OSError as e:
        log.warning(f"Could not remove {PID_FILE} on exit ({e}) -- next startup will find it stale and "
                    f"clean it up itself.")


def _spawn_backend(log: logging.Logger) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(BACKEND_PORT)],
        cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    log.info(f"Spawned uvicorn (PID {proc.pid}) on port {BACKEND_PORT}.")
    return proc


def _stop_backend(proc: subprocess.Popen, log: logging.Logger, reason: str) -> None:
    if proc.poll() is not None:
        log.info(f"uvicorn (PID {proc.pid}) had already exited (code {proc.returncode}) -- nothing to stop.")
        return
    log.info(f"Stopping uvicorn (PID {proc.pid}) -- reason: {reason}.")
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_SECONDS)
        log.info(f"uvicorn (PID {proc.pid}) exited (code {proc.returncode}).")
        return
    except subprocess.TimeoutExpired:
        pass
    log.warning(f"uvicorn (PID {proc.pid}) did not exit within {STOP_TIMEOUT_SECONDS}s -- killing.")
    proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        log.error(f"uvicorn (PID {proc.pid}) still not confirmed dead 10s after kill() -- giving up waiting.")


def _health_ok() -> bool:
    try:
        resp = requests.get(f"http://localhost:{BACKEND_PORT}/health", timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _sleep_interruptible(seconds: float) -> None:
    slept = 0.0
    while slept < seconds and not _shutdown_requested:
        chunk = min(1.0, seconds - slept)
        time.sleep(chunk)
        slept += chunk


def _run(log: logging.Logger) -> bool:
    """Returns True if the watchdog gave up (caller exits nonzero)."""
    consecutive_failures = 0
    consecutive_failed_restarts = 0
    gave_up = False

    proc = _spawn_backend(log)
    spawned_at = time.monotonic()

    try:
        while not _shutdown_requested:
            _sleep_interruptible(BACKEND_CHECK_INTERVAL_SECONDS)
            if _shutdown_requested:
                break

            elapsed_since_spawn = time.monotonic() - spawned_at
            dead = proc.poll() is not None
            past_startup_grace = elapsed_since_spawn > STARTUP_GRACE_SECONDS
            healthy = (not dead) and (not past_startup_grace or _health_ok())

            if healthy:
                if consecutive_failures > 0:
                    log.info(f"/health OK again after {consecutive_failures} failure(s) -- streak reset.")
                consecutive_failures = 0
                if past_startup_grace:
                    consecutive_failed_restarts = 0
                continue

            if dead:
                consecutive_failures = BACKEND_FAILURE_THRESHOLD  # dead is an immediate restart, no need to accumulate
                reason = f"process died (exit code {proc.returncode})"
            else:
                consecutive_failures += 1
                reason = f"/health failed {consecutive_failures} consecutive time(s)"
                log.warning(f"Check: {reason} (threshold {BACKEND_FAILURE_THRESHOLD}).")
                if consecutive_failures < BACKEND_FAILURE_THRESHOLD:
                    continue

            consecutive_failed_restarts += 1
            if consecutive_failed_restarts >= MAX_CONSECUTIVE_FAILED_RESTARTS:
                log.error(f"GIVING UP -- {consecutive_failed_restarts} consecutive backend restarts in a "
                          f"row all failed (latest reason: {reason}). Stopping instead of looping forever.")
                send_ops_alert(
                    title="NAMEK backend watchdog giving up",
                    message=(f"{consecutive_failed_restarts} consecutive backend restarts all failed "
                              f"(latest reason: {reason}). The dashboard/API is NOT running and will NOT "
                              f"be auto-restarted again -- needs manual attention."),
                    urgency="high",
                )
                gave_up = True
                _stop_backend(proc, log, reason="giving_up_after_failed_restarts")
                break

            log.warning(f"RESTARTING backend -- reason: {reason}")
            send_ops_alert(title="NAMEK backend restarted", message=f"Reason: {reason}")
            _stop_backend(proc, log, reason=reason)
            _sleep_interruptible(5)
            if _shutdown_requested:
                break
            proc = _spawn_backend(log)
            spawned_at = time.monotonic()
            consecutive_failures = 0
    finally:
        if _shutdown_requested:
            log.info("Backend watchdog interrupted -- stopping backend before exiting.")
        if not gave_up:
            _stop_backend(proc, log, reason="watchdog_shutdown")
        log.info("Backend watchdog exiting.")
    return gave_up


def main() -> None:
    log = _setup_logging()
    signal.signal(signal.SIGINT, _request_shutdown)
    try:
        signal.signal(signal.SIGTERM, _request_shutdown)
    except (ValueError, AttributeError, OSError):
        pass
    try:
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
        sys.exit(1)


if __name__ == "__main__":
    main()
