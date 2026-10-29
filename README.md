# NAMEK Intelligence — V1: Data Layer + Backtest Engine

## What this is
Phase 1 of the build sequence: **backtesting engine + historical/live data pipeline**,
built on the Upstox free API. No signal logic yet — this is the foundation
everything else (breakout validator, fakeout detector, scanner) plugs into.

## Before you run anything

### 1. Register an Upstox developer app (free)
Go to https://developer.upstox.com/, log in with your Upstox account, and create
an app. You'll get a `client_id` and `client_secret`.

### 2. Generate an Analytics Token (recommended -- much simpler than daily login)
Upstox now offers a long-lived, read-only **Analytics Token**: generate it
once from the Developer Apps page (Analytics tab -> Generate Token), and
it's valid for **1 year** -- no daily re-login needed, unlike the old
OAuth flow. This is all you need for market/historical data (exactly
what this project uses).

Copy the token into `.env`:
```
UPSTOX_ANALYTICS_TOKEN=your_token_here
```
That's it -- `config/auth.py` will use this automatically, and you can
skip the daily login flow described below entirely.

### 2b. (Fallback) Daily OAuth login -- only if you're not using the Analytics Token
Upstox's older access tokens **expire every day** (they invalidate around
3:30am IST). If you haven't set `UPSTOX_ANALYTICS_TOKEN` above, you'd need
to redo this login every trading day. `config/auth.py` gives you a helper
that prints the login URL, and a function to exchange the authorization
code for a token. Set a redirect URI when creating your app --
`http://localhost:3000/callback` works fine for local use.

### 3. Environment variables
Create a `.env` file (not committed anywhere) -- see `.env.example` for
the full template covering both the Analytics Token path and the OAuth
fallback.

## Data availability constraints (from Upstox docs — plan around these)
- 1-minute and 3-minute candles: available from **January 2022 onward**
- Intraday endpoint gives you the **current day only**, live during market hours
- Historical endpoint gives you any date range within the availability window above
- Rate limits apply — the fetcher chunks requests and backs off automatically

## Running the backend: development vs. live trading days

The dashboard's FastAPI backend (`api/main.py`) can run two ways, and
**which one you use matters** — three separate backend freezes were
traced back to uvicorn's `--reload` flag in one way or another (a
sleep-interrupted reload state, a CPU-starved reload event loop, and an
I/O-bound `StatReload` sweep of thousands of files under `storage/`).

**Development (actively editing code)** — `run_backend_dev.bat`, or:
```
uvicorn api.main:app --reload --port 8000 --reload-dir api --reload-dir backtest --reload-dir cloud_ai --reload-dir config --reload-dir data --reload-dir detectors --reload-dir examples --reload-dir notifications --reload-dir paper_trading --reload-dir portfolio --reload-dir reporting --reload-dir risk
```
`--reload` is on (auto-restarts the server when you edit code), but
**scoped** via `--reload-dir` to only the real Python source
directories. Without explicit `--reload-dir` flags, uvicorn's default is
to watch the *entire* project root recursively — which on this project
means also sweeping `storage/` (~7,100+ cached data files), `venv/`
(the virtual environment itself — tens of thousands of files),
`frontend/` (636 files, ~605 of them `node_modules`, not Python at all),
and the constantly-rewritten `logs/` directory. Scoping to source
directories only removes that cost regardless of what's actually
triggering it, without giving up auto-restart-on-edit.

**Live trading days** — `run_backend_live.bat`, or:
```
uvicorn api.main:app --port 8000
```
No `--reload` at all. You don't need auto-restart-on-edit on a day
you're not editing code, and running without `--reload` removes the
`StatReload` file-watcher mechanism entirely — there's nothing left to
watch, nothing left to sweep, regardless of root cause. This is what the
live trading day should run, and what `examples/auto_start_live_session.py`
reminds you of every time it runs (see that script's own printed output).

## Folder structure
```
namek_intelligence/
├── config/
│   ├── settings.py       # env loading, constants
│   └── auth.py           # OAuth login flow + token management
├── data/
│   ├── instruments.py    # resolve NSE symbol -> Upstox instrument_key
│   ├── fetcher.py        # pulls candles from Upstox V3 API
│   ├── storage.py        # local Parquet cache (per symbol, per timeframe)
│   └── resample.py       # build 3m/5m/15m from 1m base data
├── backtest/
│   ├── strategy_base.py  # abstract interface every future detector implements
│   ├── engine.py         # event-driven backtest loop, position/trade tracking
│   └── metrics.py        # win rate, profit factor, max DD, Sharpe, expectancy, avg R
├── examples/
│   └── run_backtest_example.py   # opening-range-breakout strategy, end-to-end test
└── storage/candles/      # local Parquet cache lives here (gitignored)
```

## Why 1-minute is the base timeframe
Every other timeframe (3m/5m/15m) is built by **resampling the 1-minute data**,
not fetched separately. This guarantees your multi-timeframe view is always
internally consistent (a 5m candle is *exactly* five real 1m candles, not a
separately-fetched series that could have gaps or vendor discrepancies between
timeframes). This matters a lot for backtest correctness.

## What "done" looks like for Phase 1
Run `examples/run_backtest_example.py` on a handful of NSE symbols and get a
metrics report (win rate, profit factor, max drawdown, Sharpe, expectancy,
avg R-multiple) for a real, simple, well-known intraday setup (opening range
breakout). If the engine produces correct, sane numbers for a *simple* known
strategy, it's trustworthy enough to build your actual detectors (breakout
validator, fakeout detector) on top of.
