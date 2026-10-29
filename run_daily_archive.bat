@echo off
REM Wrapper for Windows Task Scheduler -- same pattern as run_auto_start.bat /
REM run_daily_universe_refresh.bat (schtasks' /tr has no direct "working
REM directory" flag, so this sets it via `cd /d` before invoking the venv
REM Python interpreter as a module).
REM Scheduled to run daily at 16:00 IST, after market close (with margin),
REM once the day's data is complete and Upstox's own intraday endpoint has
REM settled -- see examples/daily_archive.py's own docstring for what this
REM does and why (permanently archives history beyond Upstox's rolling
REM ~30-day intraday window, distinct from storage/candles/).
REM Output appended to logs\daily_archive_launcher.log -- separate from the
REM script's own logs\daily_archive.log, which it creates itself once it
REM actually starts (same two-log split as the other two auto-start tasks).
REM Autonomous-system addition (2026-08-19): on a nonzero exit code, sends
REM a Telegram alert via notifications/send_ops_alert.py -- see that
REM module's docstring and run_daily_universe_refresh.bat's identical
REM pattern.
cd /d "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence"
"C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m examples.daily_archive >> logs\daily_archive_launcher.log 2>&1
if %errorlevel% neq 0 (
    "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m notifications.send_ops_alert --title "Daily Archive FAILED" --message "Exit code %errorlevel% -- check logs\daily_archive_launcher.log" --urgency high >> logs\daily_archive_launcher.log 2>&1
)
