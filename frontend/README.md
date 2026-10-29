# NAMEK Intelligence — Dashboard

React frontend for the Scanner, Backtest, and Position Sizing API. Runs
locally alongside the FastAPI backend -- this is not deployed anywhere,
it's a local tool that talks to `localhost:8000`.

## Setup (one-time)

```powershell
cd frontend
npm install
```

## Running (every time)

You need **two terminals** open at once:

**Terminal 1 -- the backend API** (from the project root, `namek_intelligence/`, not the `frontend/` folder):
```powershell
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 -- the frontend** (from the `frontend/` folder):
```powershell
npm run dev
```

Then open the URL Vite prints (usually `http://localhost:5173`) in your browser.

## What you'll see

- **Status strip** at the top: green "API LIVE" if the backend is reachable,
  red "API OFFLINE" with the exact command to fix it if not.
- **Scanner tab**: pick symbols from your local cache, run a scan, see
  ranked results with full reasoning.
- **Backtest tab**: pick a strategy, symbols, and a date range, see real
  performance metrics and every trade.
- **Position Sizing tab**: enter capital/entry/stop, get a suggested
  share quantity with the reasoning behind it.

## If the Scanner/Backtest tabs show "no symbols in cache"

That means no data has been fetched yet. Run one of the data-fetching
scripts first (`examples/run_all_detectors_yfinance.py` for free yfinance
data, or the Upstox fetcher once your account is KYC-verified), then
reload the dashboard page.
