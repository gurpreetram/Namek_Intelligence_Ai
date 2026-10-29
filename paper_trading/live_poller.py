"""
Phase B of the live-monitoring system: polls each watchlist symbol (see
paper_trading/watchlist.py's Phase A) for new 1-minute bars and feeds them
into one PaperTradingEngine per (symbol, strategy) pair via on_new_bar().

Deliberately reuses PaperTradingEngine rather than building new state-
tracking: entry/exit detection, the journal, and now (see engine.py) the
notifier hook are all the SAME code already validated to match
BacktestEngine's fill/exit rules byte-for-byte. This module's only job is
sourcing new bars (live fetch, or historical replay -- see feed_bar()) and
routing them to the right engines; it does not duplicate any trading logic.

feed_bar() is the one method both poll_once() (live) and a replay test
harness (examples/run_replay_validation.py) call -- using the exact same
code path for both is what makes replay validation actually prove
something about live behavior, instead of testing a parallel implementation.

FULL-UNIVERSE SIGNAL DISCOVERY -- REMOVED 2026-08-24. From 2026-08-17 to
2026-08-24 this module also ran a periodic full-universe scan
(scan_full_universe_once() / maybe_scan_full_universe()) to catch movers
outside the fixed morning watchlist (confirmed real cases it caught:
Manorama Inds +9.87%, PhysicsWallah +6.22%). It was removed as the single
biggest lever in a live-session memory-pressure fix: each pass bulk-
fetched ~1,600+ symbols' worth of DataFrames at once -- by far the
largest allocation anywhere in the live session -- on a machine that
repeatedly hit real "paging file too small" failures and 90%+ system RAM
under normal multi-process load (backend + watchdog + live session).
Discovering movers outside the morning watchlist is still possible, just
no longer automatic: use the Scanner tab (F1) / POST /scan for an
on-demand full-universe pass whenever wanted, against detectors/
scanner.py's scan_universe() directly -- same underlying function this
module used to call automatically, just triggered manually instead of
every DEFAULT_FULL_SCAN_INTERVAL_SECONDS. The live poller now does
exactly what it did before 2026-08-17: track the fixed morning watchlist
(examples/run_morning_watchlist.py) for the whole session, nothing more.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime
from typing import Callable

import pandas as pd
import psutil

from config.settings import PROJECT_ROOT, now_ist
from data import storage
from data.fetcher import FetchError, fetch_intraday_1min
from data.instruments import resolve_instrument_key
from detectors.fibonacci_strategy import FibonacciStrategy
from detectors.momentum_entry_strategy import MomentumEntryStrategy
from detectors.reversal_strategy import ReversalDetection
from detectors.smc_strategy import SmartMoneyStrategy
from detectors.trend_continuation_strategy import TrendContinuation
from detectors.validated_breakout_strategy import ValidatedBreakout
from detectors.vwap_strategy import VWAPStrategy
from examples.opening_range_breakout import OpeningRangeBreakout
from examples.rolling_range_breakout import RollingRangeBreakout
from notifications.notification_engine import NotificationEngine
from notifications.send_ops_alert import send_ops_alert
from paper_trading.engine import PaperTradingEngine
from paper_trading.journal import TradeJournal
from paper_trading.watchlist import WatchlistEntry

# Written after every poll cycle: {symbol: last close price} plus a
# generated_at timestamp. Exists so the API layer (api/main.py's
# /live-activity) can show a "closer to target or stop" indicator for
# open positions WITHOUT fetching its own fresh quote just to render a
# dashboard -- it reads this file instead, which is never more than one
# poll_interval_seconds stale since it's written every cycle. This is the
# cheapest correct source: the live poller already fetched these prices
# for its own trading logic, so persisting them is free, whereas a fresh
# fetch from the API layer would be redundant network calls purely for
# display.
LATEST_PRICES_PATH = PROJECT_ROOT / "paper_trading" / "latest_prices.json"

# Written after every poll cycle: {timestamp, open_positions, symbols_polled}.
# The dashboard (api/main.py's /live-session-status) uses this to tell
# "session alive" from "session died silently" -- confirmed necessary the
# hard way: a prior live session was killed (process torn down, no
# graceful shutdown -- see run_forever()'s docstring and examples/
# run_live_session.py's signal handling) and the dashboard had no way to
# know the data it was showing was hours stale rather than current.
HEARTBEAT_PATH = PROJECT_ROOT / "paper_trading" / "heartbeat.json"

# Module-level logger (not a print()) specifically so poll_once()'s
# memory-freed line lands in logs/live_session_<date>.log, not just the
# console -- added 2026-08-24. Python's logging is global/hierarchical:
# this propagates to whatever handlers examples/run_live_session.py's
# main() already configured via logging.basicConfig() in the SAME
# process, with zero extra wiring needed here. Deliberately NOT applied
# to this module's existing print() calls (per-symbol fetch failures,
# etc.) -- those match an established convention elsewhere in this file
# and changing them is out of scope for this specific addition.
_log = logging.getLogger(__name__)

# Wall-clock EOD square-off time, shared between each PaperTradingEngine's
# own per-bar check (fires when a fed bar's timestamp reaches this) and
# LivePoller's own safeguard below (fires from the ACTUAL current time,
# independent of any bar). One constant so the two can never drift apart.
EOD_SQUARE_OFF_TIME = "15:15"

# Same construction as examples/run_universe_backtest.py and api/main.py --
# the actually-backtested parameters, not loosened demo values.
STRATEGY_BUILDERS = {
    "opening_range_breakout": (lambda: OpeningRangeBreakout(), [1, 5, 15]),
    "rolling_range_breakout": (lambda: RollingRangeBreakout(), [1, 5, 15]),
    "validated_breakout": (lambda: ValidatedBreakout(), [1, 5, 15]),
    "trend_continuation": (lambda: TrendContinuation(), [1, 15]),
    "reversal_detection": (lambda: ReversalDetection(), [1]),
    "vwap_reclaim": (lambda: VWAPStrategy(), [1]),
    "smart_money": (lambda: SmartMoneyStrategy(), [1]),
    "fibonacci_retracement": (lambda: FibonacciStrategy(), [1]),
    "momentum_entry": (lambda: MomentumEntryStrategy(), [1]),
}

MARKET_CLOSE_TIME = "15:30"

# Added 2026-08-20: poll_once() distinguishes "the internet itself is
# down" (most/all symbols fail with a connection-level error -- see
# FetchError.is_connection_error) from ordinary per-symbol failures (a
# handful of bad instrument keys, one-off API errors), which the loop
# already tolerated fine via its existing per-symbol try/except. A
# WIDESPREAD connection failure is different: without this, every such
# cycle would write a heartbeat with mostly-missing data, and if it
# persists, examples/run_with_watchdog.py's staleness guard eventually
# restarts the whole session over what's actually a transient ISP/Wi-Fi
# blip -- wasted resources and journal churn for a problem restarting
# does nothing to fix. INTERNET_ISSUE_FAILURE_RATIO_THRESHOLD (>50% of
# ATTEMPTED fetches -- symbols skipped for a missing instrument_key don't
# count, see poll_once()) triggers a wait-and-retry of the SAME cycle
# instead of returning normally. Retries are capped at INTERNET_ISSUE_
# MAX_RETRIES so a genuinely sustained outage still surfaces as staleness
# eventually (5 retries x 60s = 5 minutes of retrying before giving up
# and letting the normal heartbeat/staleness path take over) rather than
# retrying forever.
INTERNET_ISSUE_FAILURE_RATIO_THRESHOLD = 0.5
INTERNET_ISSUE_RETRY_WAIT_SECONDS = 60
INTERNET_ISSUE_MAX_RETRIES = 5


class LivePoller:
    def __init__(
        self,
        watchlist: list[WatchlistEntry],
        notifier: NotificationEngine | None = None,
        journal: TradeJournal | None = None,
        poll_interval_seconds: int = 60,
        should_stop: Callable[[], bool] | None = None,
    ):
        """
        should_stop: optional zero-arg callable checked between EACH
            symbol inside poll_once()'s loop (see that method), added
            2026-08-19 -- lets a caller with its own cooperative shutdown
            flag (examples/run_live_session.py's _shutdown_requested) get
            that flag noticed within roughly one symbol's fetch time
            instead of only between whole poll cycles. Confirmed as a real
            gap, not theoretical: examples/run_with_watchdog.py validation
            found a graceful-stop request arriving mid-poll_once() wasn't
            noticed until the WHOLE cycle finished (measured 60-500s+),
            blowing well past the watchdog's 30s graceful-stop timeout and
            forcing a hard kill instead. None (default, every existing
            caller unaffected) means "never stop early" -- identical to
            behavior before this parameter existed.
        """
        self.symbols = [w.symbol for w in watchlist]
        self.notifier = notifier
        self.journal = journal = journal or TradeJournal()
        self.poll_interval_seconds = poll_interval_seconds
        self._should_stop: Callable[[], bool] = should_stop or (lambda: False)

        # One PaperTradingEngine per (symbol, strategy) pair. Engines hold
        # only trade/resampling STATE, not their own data -- cheap to
        # instantiate per pair rather than sharing across strategies the
        # way backtest/engine.py's run_multi() does for a one-shot
        # historical run (that sharing exists to avoid redundant disk
        # loads/resampling across strategies; here every engine gets fed
        # the same already-fetched `latest` frame directly in feed_bar(),
        # so there's no redundant I/O to avoid). All engines share one
        # journal and one notifier so records/alerts land in one place
        # regardless of which strategy produced them.
        # live_mode=True: this IS the actual live trading path -- applies
        # CHANGE 2 (pre-entry momentum/volume/no-pullback verification)
        # inside every engine (target stays each strategy's real 2x-risk-
        # reward value -- CHANGE 1's fixed-point target was also applied
        # here before being reverted 2026-07-22, see detectors/scanner.py's
        # FIXED_TARGET_POINTS docstring). See paper_trading/engine.py's
        # PaperTradingEngine docstring/constructor for exactly what that
        # does; backtest/replay-validation callers (examples/
        # validate_early_exit_replay.py, examples/paper_vs_backtest_crosscheck.py)
        # deliberately do NOT pass this, so they keep comparing against the
        # strategy's real, original 2x-risk-reward backtested behavior,
        # unchanged.
        self.engines: dict[str, dict[str, PaperTradingEngine]] = {
            symbol: {
                name: PaperTradingEngine(build(), tfs, journal=journal,
                                          eod_square_off_time=EOD_SQUARE_OFF_TIME, notifier=notifier,
                                          live_mode=True)
                for name, (build, tfs) in STRATEGY_BUILDERS.items()
            }
            for symbol in self.symbols
        }
        self._last_bar_seen: dict[str, pd.Timestamp] = {}
        self._last_known_prices: dict[str, float] = {}  # updated every poll_once() -- see force_close_all()

        # Prior-day context, loaded ONCE at startup from the local cache
        # (up to Upstox's ~30-day window) -- see poll_once()'s docstring
        # for why this is required, not optional: fetch_intraday_1min()
        # only ever returns TODAY's bars, which alone is never enough
        # history for TrendContinuation's regime gate (detect_regime()
        # requires a minimum of adx_period*3 = 42 bars of 15-min data --
        # a single trading day only ever produces ~25, even by close).
        # Without this, TrendContinuation could never fire through live
        # polling at all, silently -- confirmed by tracing through
        # on_new_bar()'s day-rollover bootstrap, not observed as a live
        # failure (there's no live session to observe yet).
        self._cached_history: dict[str, pd.DataFrame] = {
            symbol: storage.load(symbol) for symbol in self.symbols
        }

    def feed_bar(self, symbol: str, full_1min_history: pd.DataFrame, current_index: pd.Timestamp) -> None:
        """Feeds one new completed bar into every strategy engine tracking
        this symbol. Called by poll_once() (live) or directly, bar-by-bar,
        by a replay harness (historical) -- same code path either way."""
        for engine in self.engines[symbol].values():
            engine.on_new_bar(symbol, full_1min_history, current_index)

    def _sleep_interruptible(self, seconds: float) -> None:
        """Chunked sleep respecting self._should_stop(), same pattern
        examples/run_live_session.py's own loop uses -- added 2026-08-20
        for the internet-issue retry wait below, so a shutdown request
        during that wait is noticed within ~1s instead of blocking it."""
        slept = 0.0
        while slept < seconds and not self._should_stop():
            chunk = min(1.0, seconds - slept)
            time.sleep(chunk)
            slept += chunk

    def _poll_once_attempt(self) -> tuple[dict[str, float], int, int]:
        """One pass over self.symbols -- the actual fetch-and-feed work
        poll_once() used to do inline. Split out 2026-08-20 so poll_once()
        can call this in a retry loop (see INTERNET_ISSUE_* constants)
        without duplicating the per-symbol logic. Returns (latest_prices,
        symbols_attempted, connection_failures) -- the latter two feed the
        internet-issue detection; symbols_attempted excludes symbols
        skipped for a missing instrument_key (never even reached the
        network, so they can't signal anything about connectivity)."""
        latest_prices: dict[str, float] = {}
        symbols_attempted = 0
        connection_failures = 0
        for symbol in self.symbols:
            if self._should_stop():
                break
            instrument_key = resolve_instrument_key(symbol)
            if instrument_key is None:
                print(f"  {symbol}: no instrument_key found, skipping")
                continue
            symbols_attempted += 1
            try:
                today_live = fetch_intraday_1min(instrument_key)
            except FetchError as e:
                if e.is_connection_error:
                    connection_failures += 1
                print(f"  {symbol}: fetch failed -- {e}")
                continue
            except Exception as e:
                print(f"  {symbol}: fetch failed -- {e}")
                continue
            if today_live.empty:
                continue

            # Persist today's live-fetched bars to the local parquet cache
            # every cycle -- storage.save() already merges/dedupes safely
            # against overlapping ranges (see its docstring), so re-saving
            # the same growing day's data each poll is safe, just not the
            # most efficient possible design (rewrites each symbol's whole
            # file every cycle rather than appending incrementally). A
            # deliberately simple first step toward daily archiving, not
            # the full dedicated system -- see this module's discussion
            # around the backfill gap this closes (live-fetched data was
            # previously never written to storage.py's cache at all).
            # CLEARED as a suspect 2026-07-13: a live session attempt with
            # this line disabled died with the exact same ~8-11min, no-
            # exception signature as two prior attempts that had it active
            # -- ruling this out as the cause.
            storage.save(symbol, today_live)

            latest_prices[symbol] = float(today_live["close"].iloc[-1])

            cached = self._cached_history.get(symbol, pd.DataFrame())
            today_date = today_live.index[0].date()
            prior_days = cached[cached.index.date < today_date] if not cached.empty else cached
            combined = pd.concat([prior_days, today_live]) if not prior_days.empty else today_live

            last_seen = self._last_bar_seen.get(symbol)
            new_bars = today_live.index if last_seen is None else today_live.index[today_live.index > last_seen]
            for current_index in new_bars:
                self.feed_bar(symbol, combined, current_index)
            if len(today_live.index) > 0:
                self._last_bar_seen[symbol] = today_live.index[-1]

            # Explicit del of this iteration's DataFrames, added 2026-08-20
            # -- belt-and-braces alongside poll_once()'s end-of-cycle
            # gc.collect(): these names get REBOUND (not left dangling)
            # next iteration regardless, so refcounting already frees the
            # previous iteration's objects the moment that happens; this
            # just makes it explicit rather than implicit, per the same
            # memory-pressure effort that added the gc.collect() calls.
            del today_live, cached, prior_days, combined

        return latest_prices, symbols_attempted, connection_failures

    def poll_once(self) -> None:
        """One live polling pass: fetch each watchlist symbol's intraday
        data so far today, combine it with that symbol's cached prior-day
        history (loaded at startup -- see __init__), and feed any bars
        newer than the last one seen into every engine for that symbol.
        The combine step matters: fetch_intraday_1min() alone only ever
        covers today, which isn't enough history on its own for every
        strategy (see __init__'s note on TrendContinuation's regime gate).

        Checks self._should_stop() at the top of EACH symbol's iteration
        (see __init__'s should_stop param) and breaks out early if it
        returns True, leaving any remaining symbols unpolled this cycle --
        added 2026-08-19 so a caller's shutdown request is noticed within
        roughly one symbol's fetch time instead of only between whole
        cycles (a real gap: this loop previously ran to completion no
        matter what, and real cycles have measured 60-500s+, which blew
        past examples/run_with_watchdog.py's 30s graceful-stop timeout and
        forced a hard kill in testing). Everything after the loop (writing
        latest prices/heartbeat, the EOD safeguard) still runs on whatever
        was collected before the break -- an interrupted cycle is safe to
        end early, not something that needs to fully complete or fully
        roll back.

        INTERNET-ISSUE RETRY, added 2026-08-20: if more than
        INTERNET_ISSUE_FAILURE_RATIO_THRESHOLD of ATTEMPTED symbol fetches
        fail with a connection-level error (see FetchError.is_connection_
        error) in one pass, this is treated as "the internet is down," not
        "these specific symbols are having a bad day" -- logs, sends a
        Telegram alert, waits INTERNET_ISSUE_RETRY_WAIT_SECONDS, and
        retries the SAME cycle (up to INTERNET_ISSUE_MAX_RETRIES times)
        instead of returning a mostly-empty cycle to the caller. See those
        constants' docstring for why this exists -- a transient network
        blip shouldn't cost a full watchdog restart."""
        latest_prices: dict[str, float] = {}
        for retry_num in range(INTERNET_ISSUE_MAX_RETRIES + 1):
            latest_prices, symbols_attempted, connection_failures = self._poll_once_attempt()
            failure_ratio = (connection_failures / symbols_attempted) if symbols_attempted else 0.0
            internet_issue = failure_ratio > INTERNET_ISSUE_FAILURE_RATIO_THRESHOLD

            if not internet_issue or self._should_stop():
                break
            if retry_num >= INTERNET_ISSUE_MAX_RETRIES:
                print(f"  Internet issue persisted through {INTERNET_ISSUE_MAX_RETRIES} retries "
                      f"({connection_failures}/{symbols_attempted} symbols still failing) -- giving up "
                      f"on this cycle, returning what was collected.")
                break

            msg = (f"{connection_failures}/{symbols_attempted} symbol fetches failed with a connection "
                   f"error this cycle ({failure_ratio:.0%}) -- looks like an internet issue, not a data "
                   f"problem. Waiting {INTERNET_ISSUE_RETRY_WAIT_SECONDS}s and retrying "
                   f"(attempt {retry_num + 1}/{INTERNET_ISSUE_MAX_RETRIES}).")
            print(f"  Internet issue detected -- {msg}")
            send_ops_alert(title="NAMEK internet issue detected", message=f"\U0001F4E1 {msg}")
            self._sleep_interruptible(INTERNET_ISSUE_RETRY_WAIT_SECONDS)

        self._last_known_prices.update(latest_prices)
        self._enforce_eod_safeguard(latest_prices)
        self._write_latest_prices(latest_prices)
        self._write_heartbeat()

        # Explicit collect at the end of EVERY poll cycle, added 2026-08-19
        # -- pandas DataFrames are well-documented to sometimes form
        # internal reference cycles (block manager/indexing back-
        # references) that only the cyclic GC reclaims, not plain
        # refcounting alone. Runs every poll_interval_seconds (60s), all
        # day, every cycle, unconditionally -- not sampled/skipped.
        #
        # Logs the actual RSS delta (this process's own resident memory,
        # via psutil.Process(), measured immediately before/after the
        # collect() call) added 2026-08-24 so this is independently
        # verifiable from logs/live_session_<date>.log rather than trusted
        # on faith -- "Memory freed: Xmb" reports whatever the real number
        # is, INCLUDING zero or negative (the OS/allocator doesn't always
        # hand freed pages back immediately, and a cycle that allocated
        # more than it freed can legitimately show negative here) -- this
        # is a real measurement, not a curated one.
        rss_before = psutil.Process(os.getpid()).memory_info().rss
        gc.collect()
        rss_after = psutil.Process(os.getpid()).memory_info().rss
        freed_mb = (rss_before - rss_after) / (1024 * 1024)
        _log.info(f"Memory freed: {freed_mb:.1f}mb (RSS {rss_before/1024/1024:.1f}mb -> {rss_after/1024/1024:.1f}mb)")

    def _enforce_eod_safeguard(self, latest_prices: dict[str, float]) -> None:
        """
        Wall-clock EOD square-off, IN ADDITION to (not instead of) each
        PaperTradingEngine's own per-bar check inside on_new_bar(). That
        per-bar check only fires when a FED BAR's own timestamp reaches
        eod_square_off_time -- if a poll cycle runs slow/delayed, or a
        fetch's last bar stops just short of 15:15, or the process falls
        behind for any reason, that trigger can be missed entirely,
        leaving a position open indefinitely. Confirmed this isn't
        hypothetical: exactly this left 30 positions frozen open when a
        prior live session was killed before ever reaching a >=15:15 bar.
        This check instead uses the ACTUAL current wall-clock time, every
        single poll cycle, regardless of what any bar's timestamp says --
        so a slow cycle can delay it, but can't cause it to be skipped
        entirely the way the per-bar version could.
        """
        now = now_ist()
        if now.strftime("%H:%M") < EOD_SQUARE_OFF_TIME:
            return
        for symbol, engines in self.engines.items():
            price = latest_prices.get(symbol)
            exit_time = self._last_bar_seen.get(symbol)
            if price is None or exit_time is None:
                # This symbol didn't fetch successfully this cycle -- no
                # price/timestamp to close it at. Retried next cycle;
                # not closing with a fabricated price is safer than
                # guessing one.
                continue
            for engine in engines.values():
                if symbol in engine.open_positions():
                    engine.force_close(symbol, price, "eod_square_off", exit_time)

    def force_close_all(self, reason: str = "shutdown_close") -> int:
        """
        Force-closes every currently-open position across every symbol/
        strategy engine, using each symbol's LAST KNOWN price/bar
        timestamp from this poller's own history (not a fresh fetch --
        called from graceful-shutdown handling, where the whole point is
        to close out fast without depending on a network call that might
        itself be what's failing). A symbol with no known price/timestamp
        yet (e.g. shut down before its first successful poll) is skipped,
        not closed with a fabricated price.

        Returns the number of positions actually closed, for logging.
        """
        closed = 0
        for symbol, engines in self.engines.items():
            price = self._last_known_prices.get(symbol)
            exit_time = self._last_bar_seen.get(symbol)
            if price is None or exit_time is None:
                continue
            for engine in engines.values():
                if symbol in engine.open_positions():
                    engine.force_close(symbol, price, reason, exit_time)
                    closed += 1
        return closed

    def count_open_positions(self) -> int:
        return sum(len(engine.open_positions()) for engines in self.engines.values() for engine in engines.values())

    def _write_heartbeat(self) -> None:
        """Atomic write, same pattern as _write_latest_prices() -- see
        HEARTBEAT_PATH's module-level docstring for why this exists."""
        open_count = sum(
            len(engine.open_positions()) for engines in self.engines.values() for engine in engines.values()
        )
        payload = {
            "timestamp": datetime.now().isoformat(),
            "open_positions": open_count,
            "symbols_polled": len(self.symbols),
        }
        tmp_path = HEARTBEAT_PATH.with_suffix(".json.tmp")
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload))
        os.replace(tmp_path, HEARTBEAT_PATH)

    @staticmethod
    def _write_latest_prices(prices: dict[str, float]) -> None:
        """Atomic-ish write (temp file + rename) -- cheap insurance against
        a reader (the API process) seeing a half-written file, for a
        payload small enough that this is basically free."""
        if not prices:
            return
        payload = {"generated_at": datetime.now().isoformat(), "prices": prices}
        tmp_path = LATEST_PRICES_PATH.with_suffix(".json.tmp")
        LATEST_PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload))
        os.replace(tmp_path, LATEST_PRICES_PATH)

    def run_forever(self) -> None:
        """Polls every poll_interval_seconds until market close or
        interrupted. This is the actual live entrypoint -- only meaningful
        during real market hours (09:15-15:30 IST), since real time can't
        be fast-forwarded; see examples/run_replay_validation.py for
        testing this logic against historical data instead.

        Also runs maybe_scan_full_universe() every cycle (a no-op unless
        full_scan_interval_seconds have actually elapsed and full-universe
        scanning is enabled) -- see that method's and scan_full_universe_
        once()'s docstrings."""
        print(f"Live polling {len(self.symbols)} watchlist symbols every "
              f"{self.poll_interval_seconds}s (until {MARKET_CLOSE_TIME})...")
        while True:
            now = now_ist()
            if now.strftime("%H:%M") >= MARKET_CLOSE_TIME:
                print("Market closed -- stopping poll loop.")
                break
            self.poll_once()
            self.maybe_scan_full_universe()
            time.sleep(self.poll_interval_seconds)
