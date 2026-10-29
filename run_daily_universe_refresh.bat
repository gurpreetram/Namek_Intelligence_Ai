@echo off
REM Wrapper for Windows Task Scheduler -- same pattern as run_auto_start.bat
REM (schtasks' /tr has no direct "working directory" flag, so this sets it
REM via `cd /d` before invoking the venv Python interpreter as a module).
REM Scheduled to run daily at 08:45 IST, well before the 09:15 market
REM open -- see examples/daily_universe_refresh.py's own docstring for
REM what this does and why (closes the structural staleness gap where
REM paper_trading/live_poller.py only ever refreshes its own ~75-symbol
REM watchlist, never the full ~1,600+ symbol price-filtered universe).
REM Output appended to logs\daily_universe_refresh_launcher.log --
REM separate from the script's own logs\daily_universe_refresh_<date>.log,
REM which it creates itself once it actually starts (same two-log split
REM as run_auto_start.bat / logs\auto_start_launcher.log).
REM Autonomous-system addition (2026-08-19): on a nonzero exit code, sends
REM a Telegram alert via notifications/send_ops_alert.py before this .bat
REM itself exits -- see that module's docstring. Always exits 0 itself
REM (see send_ops_alert.py) so this doesn't mask/replace the real exit
REM code Task Scheduler records for THIS task.
cd /d "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence"
"C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m examples.daily_universe_refresh >> logs\daily_universe_refresh_launcher.log 2>&1
if %errorlevel% neq 0 (
    "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m notifications.send_ops_alert --title "Daily Universe Refresh FAILED" --message "Exit code %errorlevel% -- check logs\daily_universe_refresh_launcher.log" --urgency high >> logs\daily_universe_refresh_launcher.log 2>&1
)
