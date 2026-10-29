@echo off
REM Wrapper for Windows Task Scheduler / run_all.bat -- same pattern as
REM the other wrappers (schtasks' /tr has no direct "working directory"
REM flag, so this sets it via `cd /d` before invoking the venv Python
REM interpreter as a module). See examples/run_backend_watchdog.py's own
REM docstring for what this does.
cd /d "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence"
"C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m examples.run_backend_watchdog
