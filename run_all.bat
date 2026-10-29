@echo off
REM Starts EVERYTHING needed for a live trading day, self-healing --
REM the backend watchdog (examples/run_backend_watchdog.py, added
REM 2026-08-19) which itself manages uvicorn (auto-restarts on a crash
REM or 3 consecutive /health failures) + the live-session watchdog
REM (examples/run_with_watchdog.py) which manages run_live_session.py
REM (auto-restarts on a crash, stale heartbeat, or sustained CPU/RAM
REM pressure). Nothing else needed ever -- deliberately does NOT start
REM a separate `npm run dev` process (frontend/dist is served BY the
REM backend, see api/main.py's static mount) or anything else -- this
REM machine has 7.8GB RAM, and extra simultaneous processes were a
REM real, confirmed contributor to the system hanging under load.
REM
REM Each watchdog launches in its OWN console window (via `start`), not
REM this script's window. Ctrl+C in either window stops that watchdog
REM AND whatever it's currently managing (backend/live session) --
REM see each watchdog script's own SIGINT/SIGBREAK handling. You do NOT
REM need to separately find and stop the managed subprocess yourself.
REM
REM Before running this: build the frontend once (`npm run build` in
REM frontend/) so api/main.py's static mount has something to serve --
REM this script does NOT rebuild the frontend itself (that's a dev-time
REM step, not a daily startup step).
REM
REM Order: backend watchdog first, then live-session watchdog a few
REM seconds later, so the dashboard (http://localhost:8000) is reachable
REM while the live session is still building today's watchlist (which
REM takes real time -- see examples/run_morning_watchlist.py).
REM
REM Note: on a normal trading day you likely don't need to run this at
REM all -- "NAMEK Health Check" (08:30), "NAMEK Daily Universe Refresh"
REM (08:45), and "NAMEK Live Session Auto-Start" (09:15, which now also
REM runs the live-session watchdog -- see examples/auto_start_live_
REM session.py) are all scheduled automatically via Windows Task
REM Scheduler. This script is for the backend specifically (which has
REM no Task Scheduler entry of its own -- see run_backend_watchdog.py's
REM docstring) and for manual/ad-hoc use (testing, or after the machine
REM was off during the scheduled windows).
cd /d "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence"

echo Starting backend watchdog (manages uvicorn automatically) on http://localhost:8000 ...
start "NAMEK Backend Watchdog" cmd /k ""C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m examples.run_backend_watchdog"

timeout /t 5 /nobreak >nul

echo Starting live-session watchdog (manages run_live_session.py automatically) ...
start "NAMEK Live Session Watchdog" cmd /k ""C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m examples.run_with_watchdog"

echo.
echo Started in two separate windows: "NAMEK Backend Watchdog" and "NAMEK Live Session Watchdog".
echo Dashboard: http://localhost:8000
echo Backend watchdog log: logs\backend_watchdog.log
echo Live session watchdog log: logs\watchdog.log
echo To stop: Ctrl+C in EACH window separately -- closing this launcher window does not stop them.
