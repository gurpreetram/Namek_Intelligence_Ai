"""
Daily self-diagnosing health check, added 2026-08-19 as part of making
the whole system safe to run unattended for weeks -- see examples/
run_with_watchdog.py's and examples/run_backend_watchdog.py's docstrings
for the broader "autonomous system" context. Scheduled via Task Scheduler
at 08:30 IST, BEFORE the 08:45 universe refresh / 09:15 live session /
16:30 archive tasks -- this is the one signal a user without Claude Code
access has that "everything is fine" or "something needs attention"
without opening the laptop.

Six checks, each wrapped so ONE failing check can never silently prevent
the daily summary from being sent (a health check that itself crashes
before sending anything would be worse than useless -- the user would get
NO signal at all on a day something's actually wrong):
  1. Python/venv sane -- core dependencies (pandas, pyarrow, requests)
     importable. Trivially true if this script is running at all via the
     venv interpreter, but checked explicitly anyway (see check_python_venv())
     rather than assumed, so a broken venv reports as a WARNING instead of
     this script just never running at all with no explanation.
  2. storage/candles/ (data.storage's STORAGE_ROOT) has recent data --
     newest file mtime across the cache should be within the last couple
     of trading days (accounts for weekends: Monday's check is fine
     seeing Friday's data).
  3. storage/archive/ (data.storage's ARCHIVE_ROOT) is up to date -- same
     freshness check, against the PERMANENT archive instead of the
     rolling ~30-day cache.
  4. All scheduled tasks are enabled (not disabled) in Windows Task
     Scheduler -- checked via PowerShell's Get-ScheduledTask (no pywin32
     dependency needed).
  5. Telegram bot reachable -- calls the bot's own getMe API. Ironic edge
     case, handled explicitly: if THIS check fails, the alert reporting
     it obviously can't be sent via Telegram either -- see main()'s
     handling of that specific case.
  6. (Implicit -- not a separate numbered check) The alert itself sending
     successfully is the final proof the whole pipeline works end to end.

Run manually: python -m examples.health_check
Run via Task Scheduler: run_health_check.bat, daily at 08:30 IST -- see
    that wrapper and the project's Task Scheduler setup notes.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from config.settings import ARCHIVE_ROOT, PROJECT_ROOT, STORAGE_ROOT, is_trading_day, now_ist
from notifications.send_ops_alert import send_ops_alert

LOG_PATH = PROJECT_ROOT / "logs" / "health_check.log"

# Scheduled task names this check verifies are enabled -- kept as a plain
# list (not auto-discovered) so a task being MISSING entirely (not just
# disabled) is also caught, same as a disabled one -- see check_scheduled_
# tasks()'s "not found" handling.
SCHEDULED_TASK_NAMES = [
    "NAMEK Health Check",
    "NAMEK Daily Universe Refresh",
    "NAMEK Live Session Auto-Start",
    "NAMEK Daily Archive",
]

# How many trading days back "recent" tolerates -- 2 covers the normal
# Monday-sees-Friday weekend gap with one extra day of slack for a
# single missed run, without masking a REAL multi-day staleness problem.
MAX_STALE_TRADING_DAYS = 2


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {message}\n")
    print(message)


def _newest_mtime(directory: Path) -> datetime | None:
    if not directory.exists():
        return None
    newest: float | None = None
    for p in directory.glob("*.parquet"):
        mtime = p.stat().st_mtime
        if newest is None or mtime > newest:
            newest = mtime
    return datetime.fromtimestamp(newest) if newest is not None else None


def _trading_days_since(dt: datetime, as_of: date) -> int:
    """How many TRADING days (weekdays, ignoring holidays for simplicity
    here -- this is a coarse freshness signal, not a market-hours-exact
    one) separate dt's date from as_of. 0 if dt's date == as_of."""
    d = dt.date()
    count = 0
    cur = as_of
    while cur > d:
        cur -= timedelta(days=1)
        if is_trading_day(cur):
            count += 1
    return count


def check_python_venv() -> CheckResult:
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
        import requests as _r  # noqa: F401
        venv_marker = "venv" in sys.executable.lower()
        if not venv_marker:
            return CheckResult("Python/venv", False, f"Running from an unexpected interpreter: {sys.executable}")
        return CheckResult("Python/venv", True, f"OK ({sys.executable})")
    except Exception as e:
        return CheckResult("Python/venv", False, f"Core dependency import failed: {type(e).__name__}: {e}")


def check_candles_fresh() -> CheckResult:
    try:
        newest = _newest_mtime(STORAGE_ROOT)
        if newest is None:
            return CheckResult("storage/candles freshness", False, "No cached candle files found at all")
        age_days = _trading_days_since(newest, now_ist().date())
        if age_days > MAX_STALE_TRADING_DAYS:
            return CheckResult("storage/candles freshness", False,
                                f"Newest file is {age_days} trading days old ({newest.date()})")
        return CheckResult("storage/candles freshness", True, f"Newest file from {newest.date()} ({age_days} trading day(s) old)")
    except Exception as e:
        return CheckResult("storage/candles freshness", False, f"Check itself failed: {type(e).__name__}: {e}")


def check_archive_fresh() -> CheckResult:
    try:
        newest = _newest_mtime(ARCHIVE_ROOT)
        if newest is None:
            return CheckResult("storage/archive freshness", False, "No archive files found at all")
        age_days = _trading_days_since(newest, now_ist().date())
        if age_days > MAX_STALE_TRADING_DAYS:
            return CheckResult("storage/archive freshness", False,
                                f"Newest archive file is {age_days} trading days old ({newest.date()})")
        return CheckResult("storage/archive freshness", True, f"Newest file from {newest.date()} ({age_days} trading day(s) old)")
    except Exception as e:
        return CheckResult("storage/archive freshness", False, f"Check itself failed: {type(e).__name__}: {e}")


def check_scheduled_tasks() -> CheckResult:
    """
    Checks all SCHEDULED_TASK_NAMES via a SINGLE powershell.exe subprocess
    (one PowerShell loop over all names, one Write-Output per name), not
    one subprocess per task -- fixed 2026-08-24 after real evidence across
    four separate health-check runs (2026-08-19, 08-20, 08-22, 08-24) of
    THIS check specifically failing under memory pressure: a "paging file
    too small" OSError spawning powershell.exe, and separately a 30s
    subprocess timeout on another day. Spawning a new PowerShell process
    (itself a real, non-trivial memory allocation) four times instead of
    once quadrupled the exposure to exactly the kind of resource
    exhaustion this machine has repeatedly hit. One retry on failure adds
    cheap resilience against a single transient hiccup without masking a
    persistent problem (the retry uses the same single-subprocess call,
    so it's not re-introducing the old N-subprocess cost).
    """
    ps_cmd = (
        "foreach ($n in @(" + ",".join(f"'{name}'" for name in SCHEDULED_TASK_NAMES) + ")) { "
        "$t = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue; "
        "if ($null -eq $t) { Write-Output \"$n|NOT_FOUND\" } else { Write-Output \"$n|$($t.State)\" } }"
    )
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=30,
            )
            states: dict[str, str] = {}
            for line in result.stdout.strip().splitlines():
                if "|" in line:
                    name, state = line.split("|", 1)
                    states[name] = state
            problems: list[str] = []
            for name in SCHEDULED_TASK_NAMES:
                state = states.get(name, "")
                if state == "NOT_FOUND":
                    problems.append(f"'{name}' not found")
                elif state == "Disabled":
                    problems.append(f"'{name}' is disabled")
                elif state not in ("Ready", "Running"):
                    problems.append(f"'{name}' unexpected state: {state or '(missing from PowerShell output)'}")
            if problems:
                return CheckResult("Scheduled tasks", False, "; ".join(problems))
            return CheckResult("Scheduled tasks", True, f"All {len(SCHEDULED_TASK_NAMES)} tasks enabled")
        except Exception as e:
            last_exc = e
            continue
    return CheckResult("Scheduled tasks", False, f"Check itself failed after 2 attempts: {type(last_exc).__name__}: {last_exc}")


def check_telegram_reachable() -> CheckResult:
    import os
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return CheckResult("Telegram bot", False, "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not configured")
    try:
        resp = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe", timeout=10)
        if resp.status_code == 200 and resp.json().get("ok"):
            return CheckResult("Telegram bot", True, "Reachable")
        return CheckResult("Telegram bot", False, f"getMe returned HTTP {resp.status_code}")
    except requests.RequestException as e:
        return CheckResult("Telegram bot", False, f"Unreachable: {type(e).__name__}: {e}")


def run_all_checks() -> list[CheckResult]:
    checks = [
        check_python_venv,
        check_candles_fresh,
        check_archive_fresh,
        check_scheduled_tasks,
        check_telegram_reachable,
    ]
    results = []
    for check_fn in checks:
        try:
            result = check_fn()
        except Exception as e:
            # Belt-and-braces: even a check function raising something its
            # own try/except didn't anticipate must not stop the OTHER
            # checks (or the final summary send) from running.
            result = CheckResult(check_fn.__name__, False, f"Unexpected error: {type(e).__name__}: {e}")
        _log(f"{'OK' if result.ok else 'FAIL'}: {result.name} -- {result.detail}")
        results.append(result)
    return results


def main() -> None:
    _log("=" * 60)
    _log("HEALTH CHECK START")
    results = run_all_checks()
    failures = [r for r in results if not r.ok]

    if not failures:
        title = "NAMEK System OK"
        message = f"All {len(results)} checks passed. Have a good trading day."
        urgency = "normal"
    else:
        title = "NAMEK System WARNING"
        message = f"{len(failures)}/{len(results)} check(s) failed:\n" + "\n".join(
            f"- {r.name}: {r.detail}" for r in failures
        )
        urgency = "high"

    _log(f"SUMMARY: {title} -- {message}")

    telegram_result = next((r for r in results if r.name == "Telegram bot"), None)
    if telegram_result is not None and not telegram_result.ok:
        # The one check whose own failure means the alert carrying THIS
        # news can't be delivered either -- logged loudly so it's at
        # least visible in health_check.log even though the user won't
        # get a phone notification for it.
        _log("NOTE: Telegram itself is unreachable -- the summary above could NOT be sent as an alert. "
             "This is the one failure mode this script cannot phone home about.")

    sent = send_ops_alert(title=title, message=message, urgency=urgency)
    _log(f"Alert send attempt: {'delivered' if sent else 'NOT delivered'}")
    _log("HEALTH CHECK END")
    _log("=" * 60)

    # Exit code reflects check outcome (useful if Task Scheduler's own
    # LastTaskResult is ever consulted for this task too), but the alert
    # above has ALREADY been sent regardless -- exiting nonzero here must
    # never be interpreted as "the alert wasn't sent."
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
