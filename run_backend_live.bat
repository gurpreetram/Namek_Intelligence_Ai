@echo off
REM LIVE TRADING DAY backend: --reload is OFF entirely.
REM
REM --reload is a development convenience (auto-restart on code edits)
REM -- it is not needed to actually run a live trading day, and its file
REM watcher (StatReload, uvicorn's default) is the direct cause of at
REM least one of three distinct backend freezes traced this project (the
REM other two were a sleep-interrupted reload state and a CPU-starved
REM reload event loop -- see README.md's "Running the backend" section
REM for the full history). Running without --reload removes the
REM StatReload mechanism entirely, regardless of root cause -- there is
REM nothing left to watch, nothing left to sweep.
REM
REM Use run_backend_dev.bat instead for active coding sessions where
REM auto-restart-on-edit is actually wanted.
cd /d "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence"
"C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m uvicorn api.main:app --port 8000
