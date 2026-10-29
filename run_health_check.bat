@echo off
REM Wrapper for Windows Task Scheduler -- same pattern as run_auto_start.bat/
REM run_daily_universe_refresh.bat/run_daily_archive.bat (schtasks' /tr has
REM no direct "working directory" flag, so this sets it via `cd /d` before
REM invoking the venv Python interpreter as a module).
REM Scheduled to run daily at 08:30 IST, BEFORE all three other autonomous-
REM system tasks -- see examples/health_check.py's own docstring for what
REM this checks and why. That script sends its OWN Telegram alert on every
REM run (both "System OK" and "System WARNING" cases) -- unlike the other
REM three .bat wrappers, this one does NOT additionally call send_ops_alert
REM on a nonzero exit code, since health_check.py's exit code failing is
REM EXPECTED whenever it already sent a WARNING alert itself; a second,
REM separate "this task failed" alert on top would be redundant noise, not
REM new information.
REM Output appended to logs\health_check_launcher.log -- separate from the
REM script's own logs\health_check.log, which it creates itself.
cd /d "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence"
"C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m examples.health_check >> logs\health_check_launcher.log 2>&1
