"""
Phase A of the live-monitoring system: builds one morning watchlist by
combining two sources, each cheap enough to run once across the full
cached universe:

  1. Confirmed picks from detectors.scanner.daily_picks() -- fast, O(symbols)
     single-call scan (see scanner.py). Only OpeningRangeBreakout and
     TrendContinuation can ever appear here (ValidatedBreakout and
     ReversalDetection both need a confirmation bar a single cold call
     can't supply -- see scan_symbol()'s docstring).
  2. "Near-level" candidates: symbols where the latest price sits within
     `near_level_atr_mult` x ATR of a known 15-min support/resistance
     level (>=2 touches), and passes the same liquidity filter the
     scanner uses. These are NOT confirmed signals -- they're symbols
     worth watching live, because a ValidatedBreakout/ReversalDetection
     setup (both fundamentally about price interacting with a level) is
     more likely to develop here than at a random symbol. Phase B's
     PaperTradingEngine replays bar-by-bar and so CAN catch the full
     confirmation sequence once/if it actually happens -- Phase A only
     needs to narrow the universe down to a manageable subset worth that
     closer, per-bar attention.

This two-source design is the whole reason Phase A and Phase B are split:
Phase A is a cheap, approximate, universe-wide pre-filter; Phase B is the
full-fidelity engine, applied only to the watchlist it produces instead of
all ~2000+ symbols (which bar-by-bar replay, all day, cannot afford).

Per-symbol helpers (_check_near_level_for_symbol, select_watchlist) are
exposed separately from build_morning_watchlist() so a parallelized driver
(examples/run_morning_watchlist.py, for full-universe runs) can reuse the
EXACT same per-symbol logic and selection/ranking rules via a
multiprocessing.Pool, instead of a second, potentially-diverging
implementation -- the same "don't reimplement, orchestrate the real thing"
principle detectors/scanner.py's own module docstring describes.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.resample import multi_timeframe_view
from detectors.liquidity_analysis import analyze_liquidity
from detectors.scanner import MIN_STOCK_PRICE, ScanResult, daily_picks, display_target_price
from detectors.support_resistance import average_true_range, find_levels

# Lowered from 75 to 50 on 2026-08-24 as part of the same live-session
# memory-pressure fix that removed automatic full-universe scanning
# (paper_trading/live_poller.py's module docstring) -- the watchlist is
# now the ONLY thing the live poller tracks all day (no dynamic growth on
# top of it anymore), so its size alone determines every poll cycle's
# fetch/DataFrame footprint. Fewer tracked symbols directly means less
# memory held per cycle.
DEFAULT_MAX_WATCHLIST_SIZE = 50
DEFAULT_PICKS_TOP_N = 15  # per side (buy/sell) -- see daily_picks()
DEFAULT_NEAR_LEVEL_ATR_MULT = 1.0
NEAR_LEVEL_RESAMPLE_DAYS = 5  # same bound used in scanner.py's fast scan, same rationale


@dataclass
class WatchlistEntry:
    symbol: str
    reason: str  # "confirmed_pick" | "near_level"
    direction: str  # "long" | "short" -- best guess for near_level entries (see below)
    detail: str  # human-readable note: setup_type+score, or level price/distance/touches
    # The firing strategy's own confidence for confirmed_pick entries;
    # None for near_level entries -- those don't have a real "score", only
    # a distance_atr (see select_watchlist()), and inventing a fake score
    # for them (e.g. 1 - distance_atr) would misrepresent an unconfirmed
    # proximity heuristic as if it were a calibrated confidence like the
    # strategies' own scores are.
    score: float | None = None
    # Structured distance-to-level for near_level entries (None for
    # confirmed_pick) -- exposed separately from `detail`'s free text so
    # callers (e.g. the API/dashboard) can sort/format it without parsing
    # a human-readable string.
    distance_atr: float | None = None
    # confirmed_pick only: the firing strategy's real entry/stop-loss, plus
    # a display-only target (see detectors.scanner.display_target_price).
    # None for near_level -- there's no confirmed setup yet to price, and
    # inventing one would misrepresent an unconfirmed proximity heuristic
    # as a real trade plan (same principle as the `score` field above).
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    # near_level only: the real numbers behind `detail`'s free text, exposed
    # structurally so the dashboard can show them without parsing a string.
    # This is NOT an entry/stop/target -- it's "here's the level worth
    # watching and where price is relative to it right now."
    current_price: float | None = None
    level_price: float | None = None
    # confirmed_pick only: multi-factor re-ranking score from detectors/
    # composite_score.py, and the human-readable breakdown behind it.
    # None means composite scoring wasn't computed (e.g. this entry
    # predates that feature) -- callers should fall back to `score`.
    composite_score: float | None = None
    composite_reasoning: list[str] | None = None


def build_morning_watchlist(
    symbol_data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp | None = None,
    max_size: int = DEFAULT_MAX_WATCHLIST_SIZE,
    picks_top_n: int = DEFAULT_PICKS_TOP_N,
    near_level_atr_mult: float = DEFAULT_NEAR_LEVEL_ATR_MULT,
) -> list[WatchlistEntry]:
    """
    Single-threaded reference implementation -- fine for small symbol sets
    (e.g. tests, a bounded subset). For the full ~2373-symbol universe, use
    examples/run_morning_watchlist.py instead, which parallelizes the same
    per-symbol logic via multiprocessing with real progress visibility;
    this function has no progress reporting and runs entirely on one core.

    symbol_data: {symbol: full_1min_dataframe}, typically every cached
        symbol -- same shape daily_picks()/scan_universe() expect.
    as_of: None (default) scans each symbol as of ITS OWN latest cached
        bar -- see scanner.py's scan_universe() docstring for why a single
        shared as_of is fragile against uneven cache freshness.
    Returns confirmed picks FIRST (highest priority -- these already
    fired), then near-level candidates ranked by proximity (closest to
    its level first), capped at max_size total.
    """
    picks = daily_picks(symbol_data, as_of=as_of, top_n=picks_top_n)
    confirmed = [
        ScanResult(symbol=p.symbol, score=p.score, setup_type=p.setup_type,
                   direction=p.direction, reasoning=p.reasoning, entry_price=p.entry_price,
                   stop_loss=p.stop_loss, target=p.target,
                   composite_score=p.composite_score, composite_reasoning=p.composite_reasoning)
        for p in picks["buy"] + picks["sell"]
    ]
    seen = {c.symbol for c in confirmed}

    near_level: list[tuple[float, WatchlistEntry]] = []
    for symbol, df in symbol_data.items():
        if symbol in seen or df.empty:
            continue
        symbol_as_of = as_of if as_of is not None else df.index.max()
        result = _check_near_level_for_symbol(symbol, df, symbol_as_of, near_level_atr_mult)
        if result is not None:
            near_level.append(result)

    return select_watchlist(confirmed, near_level, max_size, picks_top_n)


def select_watchlist(
    confirmed: list[ScanResult],
    near_level: list[tuple[float, WatchlistEntry]],
    max_size: int,
    picks_top_n: int,
) -> list[WatchlistEntry]:
    """
    Shared selection/ranking/cap logic -- the single source of truth both
    build_morning_watchlist() and the parallelized universe driver use, so
    a full-universe run and a small-scale run always make the same
    decision given the same per-symbol inputs.

    confirmed: every confirmed ScanResult found (NOT yet top-N sliced --
        this function does that, same as daily_picks() does internally,
        so both entry points rank identically).
    near_level: every near-level candidate found, as (distance_atr, entry)
        pairs -- sorted here, not by the caller.

    Tie-breaking: many candidates share the same score (e.g. every
    OpeningRangeBreakout fire ties at its fixed placeholder confidence,
    0.50 -- see examples/opening_range_breakout.py) or the same distance_atr.
    Ranking by score/distance alone leaves ties resolved by whatever order
    they happened to arrive in `confirmed`/`near_level` -- which is NOT
    deterministic when those lists are built by a multiprocessing.Pool's
    imap_unordered() (see examples/run_morning_watchlist.py), since worker
    completion order varies run to run even with identical input data.
    Symbol name is used as an explicit secondary sort key so identical
    input always produces an identical watchlist, regardless of which
    worker happened to finish first.

    Ranks confirmed picks by composite_score when the caller has already
    computed one (see detectors/composite_score.py) -- falls back to raw
    strategy `score` for any ScanResult where composite_score is still
    None (composite scoring not computed, or every one of its components
    was unavailable for that symbol).
    """
    def _rank_key(c: ScanResult):
        primary = c.composite_score if c.composite_score is not None else c.score
        return (-primary, c.symbol)

    longs = sorted(
        [c for c in confirmed if c.direction == "long" and c.entry_price > 0],
        key=_rank_key,
    )[:picks_top_n]
    shorts = sorted(
        [c for c in confirmed if c.direction == "short" and c.entry_price > 0],
        key=_rank_key,
    )[:picks_top_n]

    watchlist = [
        WatchlistEntry(symbol=c.symbol, reason="confirmed_pick", direction=c.direction,
                        detail=f"{c.setup_type} score={c.score:.2f}", score=c.score,
                        entry_price=c.entry_price, stop_loss=c.stop_loss,
                        target_price=display_target_price(c.entry_price, c.direction),
                        composite_score=c.composite_score, composite_reasoning=c.composite_reasoning or None)
        for c in longs + shorts
    ]
    seen = {w.symbol for w in watchlist}

    remaining_slots = max_size - len(watchlist)
    if remaining_slots > 0:
        filtered = [(dist, entry) for dist, entry in near_level if entry.symbol not in seen]
        filtered.sort(key=lambda t: (t[0], t[1].symbol))
        for dist, entry in filtered[:remaining_slots]:
            entry.distance_atr = dist
            watchlist.append(entry)

    return watchlist[:max_size]


def _check_near_level_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    near_level_atr_mult: float,
) -> tuple[float, WatchlistEntry] | None:
    """Per-symbol near-level check -- returns (distance_atr, WatchlistEntry)
    or None. Factored out so both the single-threaded path and the
    parallelized universe driver call the exact same logic."""
    truncated = df.loc[:as_of]
    if truncated.empty:
        return None

    # Same price floor the scanner uses (detectors.scanner.MIN_STOCK_PRICE)
    # -- reject before the liquidity gate or any level-detection work.
    if float(truncated["close"].iloc[-1]) < MIN_STOCK_PRICE:
        return None

    # Same liquidity gate the scanner uses -- a level sitting right next
    # to price in an illiquid stock isn't a real breakout candidate, it's
    # microstructure noise (see BANG/AFFLE cases documented in
    # backtest/engine.py).
    liquidity = analyze_liquidity(truncated)
    if liquidity is not None and not liquidity.is_liquid:
        return None

    # Same bounded resample window used in scanner.py's fast scan, and for
    # the same reason: a single-point-in-time check doesn't need the full
    # ~30-day history, and pandas resample()'s per-call overhead dominates
    # cost at universe scale otherwise.
    recent_dates = sorted({d for d in truncated.index.date})[-NEAR_LEVEL_RESAMPLE_DAYS:]
    bounded = truncated[truncated.index.date >= recent_dates[0]]
    df_15 = multi_timeframe_view(bounded, [15])[15]
    if df_15.empty:
        return None

    levels = find_levels(df_15, min_touches=2)
    if not levels:
        return None
    atr = average_true_range(df_15.iloc[-30:])
    if atr <= 0:
        return None

    current_price = float(truncated["close"].iloc[-1])
    closest = min(levels, key=lambda lvl: abs(lvl.price - current_price))
    distance_atr = abs(closest.price - current_price) / atr
    if distance_atr > near_level_atr_mult:
        return None

    # Best-guess direction: approaching a level from below suggests a
    # potential upside breakout if it clears; from above, a potential
    # downside breakdown. This is a WATCH direction, not a confirmed
    # signal -- ValidatedBreakout's actual confirmation logic (level cross
    # + follow-through bar + MTF alignment) is what determines the real
    # direction once/if it fires in Phase B.
    direction = "long" if current_price < closest.price else "short"

    return distance_atr, WatchlistEntry(
        symbol=symbol, reason="near_level", direction=direction,
        detail=f"price {current_price:.2f} within {distance_atr:.2f}x ATR of level "
               f"{closest.price:.2f} (touched {closest.touch_count}x)",
        current_price=current_price, level_price=closest.price,
    )
