@echo off
REM Development backend: WITH --reload, but SCOPED via --reload-dir to
REM only the real Python source directories -- NOT the whole project
REM root, which is uvicorn's default --reload behavior if --reload-dir
REM is never specified.
REM
REM Why this exists: three distinct backend freezes were traced back to
REM --reload in one way or another (a sleep-interrupted reload state, a
REM CPU-starved reload event loop, and an I/O-bound StatReload sweep of
REM thousands of files under storage/). This script addresses the third
REM one specifically, as defense-in-depth for active development
REM sessions that still want --reload's convenience (auto-restart on
REM code edits). It does NOT fully eliminate the other two failure modes
REM (sleep, CPU starvation) -- those are about system state, not watch
REM scope. For live TRADING days, use run_backend_live.bat instead
REM (--reload OFF entirely), which removes the StatReload mechanism
REM altogether regardless of root cause -- see README.md's "Running the
REM backend" section for the full reasoning.
REM
REM Directories explicitly watched (real Python source only):
REM   api, backtest, cloud_ai, config, data, detectors, examples,
REM   notifications, paper_trading, portfolio, reporting, risk
REM Directories explicitly NOT watched (by omission -- --reload-dir is
REM inclusion-based, so anything not listed is simply never scanned):
REM   storage/  (~7,100+ cached data files -- the main StatReload cost)
REM   logs/     (constantly rewritten during any live/dev session)
REM   frontend/ (636 files, ~605 of them frontend/node_modules -- not
REM             Python source at all, was being needlessly watched under
REM             uvicorn's old unscoped default)
REM   venv/     (the Python virtual environment itself -- tens of
REM             thousands of files, easily the single largest watch
REM             target if left unscoped)
REM   scratch_pdfs/ (static reference PDFs, not code)
cd /d "C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence"
"C:\Users\Gurpreet Ram\Downloads\namek_intelligence-live\namek_intelligence\venv\Scripts\python.exe" -m uvicorn api.main:app --reload --port 8000 --reload-dir api --reload-dir backtest --reload-dir cloud_ai --reload-dir config --reload-dir data --reload-dir detectors --reload-dir examples --reload-dir notifications --reload-dir paper_trading --reload-dir portfolio --reload-dir reporting --reload-dir risk
