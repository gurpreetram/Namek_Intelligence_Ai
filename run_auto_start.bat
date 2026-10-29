@echo off
REM Wrapper for Windows Task Scheduler -- schtasks' /tr has no direct
REM "working directory" flag, so this .bat sets it via `cd /d` before
REM invoking the venv Python interpreter as a module (matching how every
REM other script in this project is run: `python -m examples.X`, which
REM depends on the project root being the current directory for imports
REM to resolve). Output (both this wrapper's own and whatever the
REM launcher/live session print to stdout/stderr) is appended to
REM logs\auto_start_launcher.log -- separate from run_live_session.py's
REM own logs\live_session_<date>.log, which it creates itself once it
REM actually starts.
REM Autonomous-system addition (2026-08-19): on a nonzero exit code, sends
REM a Telegram alert -- see notifications/send_ops_alert.py's docstring.
REM Normal exits (not a trading day, ran past market close, market-close
REM stop) are all exit 0 -- a nonzero code here means something actually
REM went wrong before/during the watchdog-managed live session, including
REM the watchdog giving up after MAX_CONSECUTIVE_FAILED_RESTARTS (which
REM ALSO sends its own alert directly -- this is deliberate, harmless
REM defense-in-depth, not a bug, if that specific case double-alerts).
cd /d "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence"
"C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m examples.auto_start_live_session >> logs\auto_start_launcher.log 2>&1
if %errorlevel% neq 0 (
    "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m notifications.send_ops_alert --title "Live Session Auto-Start FAILED" --message "Exit code %errorlevel% -- check logs\auto_start_launcher.log and logs\watchdog.log" --urgency high >> logs\auto_start_launcher.log 2>&1
)
