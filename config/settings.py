"""
Central configuration for NAMEK Intelligence.
Loads credentials from environment / .env and defines constants used
across the data and backtest layers.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    """
    The current time, correctly converted to IST regardless of what
    timezone the underlying host/server's system clock is set to. Market-
    hours logic (open/close window, weekday, holiday checks) must use
    this instead of naive datetime.now() -- a naive call silently assumes
    the host's local time already IS IST, which only happens to hold on
    a machine explicitly configured that way and breaks silently (no
    exception, just wrong answers) on any server/container defaulting to
    UTC, which is the common case for cloud VMs and CI runners.
    """
    return datetime.now(IST)

# --- Upstox API ---
UPSTOX_CLIENT_ID = os.getenv("UPSTOX_CLIENT_ID", "")
UPSTOX_CLIENT_SECRET = os.getenv("UPSTOX_CLIENT_SECRET", "")
UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:3000/callback")

# Analytics Token: a long-lived (1-year validity), read-only token Upstox
# introduced -- generated once from the Developer Apps page (Analytics
# tab), no daily OAuth login required. This is the preferred path for
# market/historical data (exactly what this project needs) and makes the
# old daily-login flow in config/auth.py unnecessary for most use cases.
# Set this in .env once you've generated it and the old OAuth flow below
# becomes a fallback you likely won't need.
UPSTOX_ANALYTICS_TOKEN = os.getenv("UPSTOX_ANALYTICS_TOKEN", "")

UPSTOX_BASE_URL = "https://api.upstox.com"
UPSTOX_AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
UPSTOX_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = PROJECT_ROOT / "storage" / "candles"
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

# Permanent archive, never pruned -- distinct from STORAGE_ROOT above, which
# is Upstox's own ~30-day rolling intraday window. See data/storage.py's
# load_archive()/save_archive()/load_combined() and examples/daily_archive.py.
ARCHIVE_ROOT = PROJECT_ROOT / "storage" / "archive"
ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

TOKEN_CACHE_PATH = PROJECT_ROOT / "config" / ".token_cache.json"

# --- Data constraints (per Upstox docs, confirmed July 2026) ---
# 1-minute and 3-minute intraday data is only available from this date forward.
MINUTE_DATA_AVAILABLE_FROM = "2022-01-01"

# Base timeframe everything else is resampled from. Do not change without
# also updating data/resample.py's assumptions.
BASE_TIMEFRAME_MINUTES = 1

# NSE regular market session (IST). Used to bound resampling and to drop
# any pre/post market ticks that occasionally leak into vendor data.
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

NSE_HOLIDAYS_PATH = PROJECT_ROOT / "config" / "nse_holidays.json"


def load_nse_holidays() -> set[date]:
    """
    Reads config/nse_holidays.json (the official NSE trading-holiday
    calendar -- see that file's _comment for sourcing/verification
    notes). A year not present in the file simply contributes no
    holidays rather than raising, so a missing future year degrades to
    "weekday + time window only" instead of crashing market-hours checks.
    """
    if not NSE_HOLIDAYS_PATH.exists():
        return set()
    data = json.loads(NSE_HOLIDAYS_PATH.read_text())
    holidays: set[date] = set()
    for year, entries in data.items():
        if year == "_comment":
            continue
        for entry in entries:
            holidays.add(date.fromisoformat(entry["date"]))
    return holidays


NSE_HOLIDAYS = load_nse_holidays()


def is_trading_day(d: date) -> bool:
    """
    True if `d` is a real NSE trading day: a weekday (Mon-Fri) that is
    not a listed NSE holiday. Does NOT check the intraday time window --
    see api/main.py's _is_market_hours() for the combined weekday +
    holiday + time-window check actually used to decide "is the market
    open right now."
    """
    return d.weekday() < 5 and d not in NSE_HOLIDAYS

# Rate limiting: Upstox enforces per-second/minute/day caps on historical
# endpoints. Conservative client-side throttle to avoid 429s during bulk
# historical downloads across many symbols.
REQUEST_MIN_INTERVAL_SECONDS = 0.35
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
