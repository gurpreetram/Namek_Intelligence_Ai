"""
Pulls 1-minute OHLCV candles from Upstox's V3 historical/intraday endpoints.

Two endpoints, both required:
  - Historical: https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/1/{to_date}/{from_date}
    -> any completed past date range (subject to the Jan-2022 availability floor)
  - Intraday:   https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/minutes/1
    -> current trading day only, live during market hours

Everything downstream (resample.py, storage.py) works off 1-minute candles,
so this module never fetches other intervals directly -- see README for why.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from config.auth import get_valid_token
from config.settings import (
    MAX_RETRIES,
    PROJECT_ROOT,
    REQUEST_MIN_INTERVAL_SECONDS,
    RETRY_BACKOFF_SECONDS,
    UPSTOX_BASE_URL,
)

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]

# Dedicated error log for per-symbol failures swallowed by the _bulk()
# functions below (a single bad symbol shouldn't kill the whole batch, but
# silently discarding *why* it failed made the 2026-08-03/04/05 daily_archive
# outage undiagnosable after the fact -- see fetch_historical_1min_bulk()).
# Separate from any one caller's own log (e.g. logs/daily_archive.log) since
# this module has multiple callers.
_error_logger = logging.getLogger("namek.fetcher.errors")
if not _error_logger.handlers:
    _error_log_path = PROJECT_ROOT / "logs" / "fetch_errors.log"
    _error_log_path.parent.mkdir(parents=True, exist_ok=True)
    _handler = logging.FileHandler(_error_log_path)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _error_logger.addHandler(_handler)
    _error_logger.setLevel(logging.INFO)
    _error_logger.propagate = False


class FetchError(RuntimeError):
    """is_connection_error is True specifically when every retry failed due
    to a connection-level problem (timeout, DNS failure, connection reset --
    requests.exceptions.RequestException at the socket/transport level, not
    an HTTP error response), as opposed to an API-level failure (auth, a
    real HTTP error status). Added 2026-08-20 so a caller like paper_
    trading/live_poller.py's poll_once() can distinguish "the internet
    itself is down" from "Upstox rejected this specific request" reliably
    (checking this flag), instead of string-matching the exception message."""

    def __init__(self, message: str, is_connection_error: bool = False):
        super().__init__(message)
        self.is_connection_error = is_connection_error


class _RateLimiter:
    """
    Thread-safe request-issuance gate: at most one request is let through
    every `min_interval_seconds`, enforced GLOBALLY across however many
    threads share one instance (a plain module-level "last request time"
    float, mutated without a lock, is not safe under concurrent callers --
    two threads can both read a stale timestamp before either writes the
    new one, so the intended spacing silently stops being enforced).
    """

    def __init__(self, min_interval_seconds: float):
        self._min_interval = min_interval_seconds
        self._lock = threading.Lock()
        self._last_issue_time = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_issue_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_issue_time = time.time()


# The module's default, single-threaded rate limiter -- same
# REQUEST_MIN_INTERVAL_SECONDS interval as always, just now correctly
# enforced under concurrency too (harmless for the existing single-threaded
# callers below; a lock with no contention is a no-op).
_default_limiter = _RateLimiter(REQUEST_MIN_INTERVAL_SECONDS)

# Dedicated limiter for fetch_intraday_1min_bulk() below -- deliberately
# SEPARATE from _default_limiter so this doesn't change the throttle rate
# any existing single-threaded caller (bulk_fetch.py, live_poller.py, etc.)
# experiences. Calibrated against Upstox's actual documented limit, not
# guessed: historical/intraday candle endpoints fall under "Other Standard
# APIs" (confirmed via upstox.com/developer/api-documentation/rate-limiting/
# and Upstox staff confirming the category on their community forum,
# 2026-07-13) -- 50 requests/sec, 500/min, 2000/30min. 0.05s/request caps
# aggregate issuance at ~20 req/sec, 40% of the verified per-second ceiling;
# per-minute/30-min caps are irrelevant here since a full 181-symbol batch
# is far below 500 requests regardless. REQUEST_MIN_INTERVAL_SECONDS
# (0.35s, ~2.86 req/sec) was a conservative default picked without this
# verified number -- left untouched for everything else, since auditing
# every other call site's safety margin wasn't asked for here.
BULK_FETCH_MIN_INTERVAL_SECONDS = 0.05


def _throttle() -> None:
    _default_limiter.wait()


def _get_with_limiter(url: str, limiter: _RateLimiter) -> dict:
    token = get_valid_token()
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}

    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            # Timeout lowered from 20s to 10s on 2026-08-20 -- a single
            # hanging request no longer eats up to 20s before even
            # starting its retry backoff; halves the worst-case time a
            # slow/dead connection can block one symbol's fetch inside
            # poll_once()'s per-symbol loop.
            resp = requests.get(url, headers=headers, timeout=10)
        except requests.exceptions.RequestException as exc:
            # Connection-level failure (timeout, connection reset, DNS
            # hiccup, etc. -- Timeout and ConnectionError are both
            # RequestException subclasses) never reached the status-code
            # checks below, so it used to skip retries entirely and
            # propagate immediately. Treat it the same as a 429: back off
            # and retry, since a transient network blip deserves the same
            # cushion as a transient rate limit. is_connection_error=True
            # here specifically (not on the 429/API-error branches below)
            # -- see FetchError's docstring.
            last_exc = FetchError(f"Connection error: {type(exc).__name__}: {exc}", is_connection_error=True)
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            # Rate limited -- back off and retry. Record the cause in
            # last_exc same as the connection-exception branch above: if
            # every retry keeps hitting 429, the final raise below used to
            # fall through to the generic "no response captured" message,
            # hiding that it was sustained rate-limiting.
            last_exc = FetchError(f"Rate limited (HTTP 429) after {MAX_RETRIES} retries for {url}")
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue
        if resp.status_code in (401, 403):
            raise FetchError(
                f"Auth error ({resp.status_code}). Token likely expired -- "
                f"rerun the daily login flow (see config/auth.py)."
            )
        last_exc = FetchError(f"Upstox API error {resp.status_code}: {resp.text}")
        time.sleep(RETRY_BACKOFF_SECONDS)

    raise last_exc or FetchError("Request failed after retries with no response captured.")


def _get(url: str) -> dict:
    return _get_with_limiter(url, _default_limiter)


def _parse_candles(payload: dict) -> pd.DataFrame:
    candles = payload.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame(columns=CANDLE_COLUMNS).set_index("timestamp")
    df = pd.DataFrame(candles, columns=CANDLE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_historical_1min_with_limiter(
    instrument_key: str, from_date: date, to_date: date, limiter: _RateLimiter,
) -> pd.DataFrame:
    """
    Shared implementation behind fetch_historical_1min() (default,
    sequential limiter) and fetch_historical_1min_bulk() (dedicated
    concurrent limiter) -- the month-by-month chunking Upstox requires for
    older ranges only needs to exist once.
    """
    frames = []
    chunk_start = from_date
    while chunk_start <= to_date:
        chunk_end = min(chunk_start + timedelta(days=29), to_date)
        url = (
            f"{UPSTOX_BASE_URL}/v3/historical-candle/{instrument_key}/minutes/1/"
            f"{chunk_end.isoformat()}/{chunk_start.isoformat()}"
        )
        payload = _get_with_limiter(url, limiter)
        frames.append(_parse_candles(payload))
        chunk_start = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(frames).sort_index()


def fetch_historical_1min(instrument_key: str, from_date: date, to_date: date) -> pd.DataFrame:
    """
    Fetches 1-minute candles for a completed date range. Upstox limits how
    much 1-minute data can be pulled per request for older ranges, so this
    chunks the request month-by-month and concatenates.
    """
    return _fetch_historical_1min_with_limiter(instrument_key, from_date, to_date, _default_limiter)


def fetch_historical_1min_bulk(
    instrument_keys: dict[str, str], from_date: date, to_date: date, max_workers: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Concurrent counterpart to fetch_historical_1min() for fetching MANY
    symbols' historical range at once -- same threading rationale as
    fetch_intraday_1min_bulk() (I/O-bound network waiting, not CPU-bound),
    and the same dedicated BULK_FETCH_MIN_INTERVAL_SECONDS limiter (see
    that constant's docstring for the verified Upstox rate limit it's
    calibrated against). Built for examples/daily_archive.py, which needs
    to pull one completed day across the full cached universe (thousands
    of symbols) without a slow one-at-a-time loop.

    instrument_keys: {symbol: instrument_key}. A symbol whose fetch fails
    for any reason is still omitted from the result (one bad symbol
    shouldn't kill the whole batch), but the exception type/message is now
    recorded to logs/fetch_errors.log rather than discarded -- a run where
    every symbol fails should be diagnosable from the log in under a
    minute, not require re-running the fetch by hand with instrumentation
    bolted on (see the 2026-08-03/04/05 daily_archive outage).
    """
    limiter = _RateLimiter(BULK_FETCH_MIN_INTERVAL_SECONDS)

    def _fetch_one(symbol: str, instrument_key: str) -> tuple[str, Optional[pd.DataFrame]]:
        try:
            df = _fetch_historical_1min_with_limiter(instrument_key, from_date, to_date, limiter)
            return symbol, df
        except Exception as exc:
            _error_logger.info(f"{symbol}: {type(exc).__name__}: {exc}")
            return symbol, None

    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_one, symbol, key) for symbol, key in instrument_keys.items()]
        for future in as_completed(futures):
            symbol, df = future.result()
            if df is not None and not df.empty:
                results[symbol] = df
    return results


def fetch_intraday_1min(instrument_key: str) -> pd.DataFrame:
    """Current trading day's 1-minute candles, live during market hours."""
    url = f"{UPSTOX_BASE_URL}/v3/historical-candle/intraday/{instrument_key}/minutes/1"
    payload = _get(url)
    return _parse_candles(payload)


def fetch_intraday_1min_bulk(
    instrument_keys: dict[str, str], max_workers: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Concurrent counterpart to fetch_intraday_1min() for fetching MANY
    symbols at once -- e.g. api/main.py's /sector-momentum-picks, which
    used to fetch all ~180 sector-mapped symbols one at a time (~87-150s).
    Threads, not processes: this is I/O-bound (waiting on Upstox's network
    response), not CPU-bound like the backtest/scan parallelization
    elsewhere in this project, so a ThreadPoolExecutor is the right tool --
    a process pool would add serialization overhead for no benefit here.

    instrument_keys: {symbol: instrument_key}. A symbol whose fetch fails
    for any reason is silently omitted from the result, same "one bad
    symbol shouldn't kill the whole batch" behavior the sequential caller
    already had -- this function only changes HOW the fetching happens,
    not what a caller does with failures.

    Uses BULK_FETCH_MIN_INTERVAL_SECONDS (its own dedicated rate limiter,
    see that constant's docstring for the verified Upstox limit it's
    calibrated against) -- completely separate from the module's default
    sequential throttle, so this function existing has zero effect on
    fetch_intraday_1min()/fetch_historical_1min()'s behavior.

    MEMORY: checked 2026-08-19 during a live-session memory-pressure
    investigation (real RAM measured at 91% on the 4-core/7.8GB machine
    this runs on) for whether this function itself retains DataFrames
    after returning. It doesn't: `futures` and the thread pool are both
    function-local and released via normal refcounting the instant this
    function returns (the `with` block's __exit__ already joins/shuts down
    every worker thread first) -- there's no closure, cache, or global
    capturing a `df` beyond its own loop iteration. `results` is the only
    thing that outlives this call, by design (it's the return value). The
    real fix for this function's memory footprint isn't inside it -- it's
    in the CALLER holding potentially ~1,600+ of these DataFrames at once
    (see paper_trading/live_poller.py's scan_full_universe_once(), which
    now explicitly dels its `fetched`/`symbol_data` dicts and calls
    gc.collect() right after use, for exactly this reason).
    """
    limiter = _RateLimiter(BULK_FETCH_MIN_INTERVAL_SECONDS)

    def _fetch_one(symbol: str, instrument_key: str) -> tuple[str, Optional[pd.DataFrame]]:
        url = f"{UPSTOX_BASE_URL}/v3/historical-candle/intraday/{instrument_key}/minutes/1"
        try:
            payload = _get_with_limiter(url, limiter)
            return symbol, _parse_candles(payload)
        except Exception:
            return symbol, None

    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_one, symbol, key) for symbol, key in instrument_keys.items()]
        for future in as_completed(futures):
            symbol, df = future.result()
            if df is not None and not df.empty:
                results[symbol] = df
    return results
