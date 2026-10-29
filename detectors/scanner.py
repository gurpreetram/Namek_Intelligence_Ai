"""
Scanner: runs across a universe of symbols and surfaces the ones with the
strongest CURRENT setups, ranked by the confidence of whichever strategy
fired.

Historically this module reimplemented lighter, parallel versions of the
breakout/pullback/regime logic that actually lives in
detectors/validated_breakout_strategy.py, detectors/trend_continuation_strategy.py,
and detectors/reversal_strategy.py -- with its own hardcoded thresholds, no
confirmation-bar requirement for breakouts, a soft MTF-alignment score bump
instead of a hard gate, and no reversal coverage at all. That meant a signal
the scanner flagged was NOT guaranteed to be something the actual backtested
strategy would have traded, and the scanner never even computed a stop-loss
or target, so there was no way to know what a real entry would even look like.

This version fixes that by running the ACTUAL Strategy classes
(OpeningRangeBreakout, ValidatedBreakout, TrendContinuation, ReversalDetection,
VWAPStrategy, SmartMoneyStrategy, FibonacciStrategy) through their real evaluate() method -- the exact same code path
backtest/engine.py and paper_trading/engine.py run, fed via the same
data.resample.multi_timeframe_view construction and the same no-lookahead
truncation pattern. A signal reported here is guaranteed to be exactly
what the corresponding strategy would trade in a backtest or paper-trading
run on the same data: same confidence thresholds, same MTF/regime gates,
same stop-loss and target computation. There is no more divergence between
"what the live tool shows" and "what was actually validated" -- if the
scanner flags it, backtesting that exact bar produces the exact same
Signal, because it IS the same Signal.

scan_symbol() calls each strategy's evaluate() exactly ONCE per call, on
the single latest bar (`as_of`), not a bar-by-bar replay of the day --
this is what makes scanning a 2000+ symbol universe fast (O(symbols), not
O(symbols x bars_since_open)), which is what daily_picks() needs. The one
real cost of that speed: ValidatedBreakout and ReversalDetection both
require a confirmation bar AFTER their candidate bar, tracked via internal
state that only gets set by an earlier evaluate() call -- a single cold
call can never supply that, so under this scan mode those two strategies
never fire (see scan_symbol()'s docstring for the full explanation).
OpeningRangeBreakout, TrendContinuation, VWAPStrategy, SmartMoneyStrategy,
and FibonacciStrategy CAN all produce a Signal through this path (the
latter three use a single-bar retest+rejection design, not a 2-step
state machine -- see each one's own module docstring). That is a known,
deliberate limitation of fast-scan mode for the two excluded strategies,
not an oversight -- a caller that needs ValidatedBreakout/ReversalDetection
coverage from a scan needs the (much slower) bar-by-bar replay backtest/
engine.py itself uses for actual backtesting.

Design principle carried over from your own brief's "no feature addiction"
rule still holds: the scanner does NOT invent its own signal logic, it only
orchestrates strategies (now the real ones) and ranks their output. If you
want the scanner to catch a new pattern, that pattern needs its own detector
+ Strategy + backtest first, not a shortcut bolted onto the scanner directly.

Two things the scanner explicitly reports, matching the "explainable AI"
requirement:
  - WHY each symbol ranked where it did (the real Signal's reasoning list)
  - What did NOT fire, so a flat scan isn't mistaken for "nothing happening"
    when it's actually "conditions aren't set up yet"
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from backtest.strategy_base import Signal, SignalDirection, Strategy
from data.resample import multi_timeframe_view
from detectors.liquidity_analysis import analyze_liquidity
from detectors.momentum_entry_strategy import MomentumEntryStrategy
from detectors.pattern_recognition import PatternType, find_double_top_bottom, find_head_and_shoulders
from detectors.relative_strength import compute_relative_strength
from detectors.fibonacci_strategy import FibonacciStrategy
from detectors.reversal_strategy import ReversalDetection
from detectors.smart_money_concepts import OrderBlockType, find_liquidity_sweeps, find_order_blocks
from detectors.smc_strategy import SmartMoneyStrategy
from detectors.support_resistance import average_true_range
from detectors.trend_continuation_strategy import TrendContinuation
from detectors.validated_breakout_strategy import ValidatedBreakout
from detectors.vwap_strategy import VWAPStrategy
from examples.opening_range_breakout import OpeningRangeBreakout
from examples.rolling_range_breakout import RollingRangeBreakout

# Below this price, a symbol is excluded from scanning entirely -- penny/
# low-priced stocks tend to have wider relative spreads, noisier ticks,
# and less institutional participation for a given rupee move than the
# same % move in a higher-priced name. A configurable constant (not a
# magic number inline) so it's easy to find and adjust later. Applied as
# early as possible in scan_symbol() -- before liquidity analysis, before
# any strategy evaluation -- so it also reduces the effective universe
# size being processed, not just hides low-price results after scoring.
MIN_STOCK_PRICE = 100.0

# CHANGE 1, REVERTED 2026-07-22 (was: live Daily Picks / live_poller only,
# a fixed point-distance target replacing the strategy's own 2x-risk-reward
# target on that path). A 400-symbol backtest comparing verified-with-
# fixed-target against the original 2x-risk-reward target showed it hurt
# 3 of 4 verified strategies (vwap_reclaim and fibonacci_retracement both
# flipped from marginally profitable to unprofitable) -- most likely
# because a flat point value isn't scale-invariant across this project's
# ~Rs 100-30,000+ price range the way a percentage/ATR-based target is
# (every OTHER threshold in this codebase -- stop-loss floors, early-exit
# triggers, magnitude classification -- is deliberately ATR-relative for
# exactly this reason). Live Daily Picks and paper_trading/engine.py's
# live trading path both now use the strategy's real Signal.target again,
# unchanged, same as backtest/engine.py always has.
#
# Left defined but UNUSED (not deleted): smart_money's verified variant
# showed a genuine improvement under the fixed-point system (40 trades,
# every metric up: win rate, profit factor, avg R, Sharpe) -- worth a
# follow-up test isolating whether verification alone (with the ORIGINAL
# target) drives that improvement, or the fixed target specifically, before
# deciding whether to selectively re-enable this for smart_money only.
# See detectors/pick_verification.py's apply_fixed_point_target_to_signal()
# for the (still fully functional, just currently uncalled from any
# live_mode path) implementation.
FIXED_TARGET_POINTS = 12.0
MIN_FIXED_TARGET_RR = 0.75


@dataclass
class ScanResult:
    symbol: str
    score: float  # 0-1 -- the firing strategy's own Signal.confidence, not an invented score
    setup_type: str  # Strategy.name of whichever fired, or "none" | "illiquid" | "error"
    direction: str  # "long" | "short" | "none"
    reasoning: list[str] = field(default_factory=list)
    # entry_price: the close of the bar the winning strategy actually fired
    # on (not necessarily `as_of` -- see scan_symbol's docstring on why a
    # setup completed a few minutes ago is still reported). Every strategy
    # in this project computes stop_loss/target off that same bar's close,
    # so this is the correct, consistent "entry" reference, not a guess --
    # see daily_picks() below for what it's used for. 0.0 when direction
    # is "none"/"illiquid"/"error" -- there's no signal to price.
    entry_price: float = 0.0
    stop_loss: float = 0.0  # the winning Signal's own stop-loss, unchanged -- not invented here
    # The winning Signal's own REAL, backtested target (2x/3x risk,
    # whatever the firing strategy used) -- distinct from daily_picks()'s
    # display_target_price(), which is a flat 2.5% width for every pick
    # regardless of strategy (display-only, see that function's docstring).
    # This field exists so "profit potential" in composite_score.py can be
    # measured against the REAL target width, which varies pick to pick,
    # not the flat display convention, which is identical for every pick
    # and therefore useless as a ranking signal.
    target: float = 0.0
    # Set by a caller that re-ranks confirmed picks with detectors/
    # composite_score.py (e.g. daily_picks(), examples/run_morning_
    # watchlist.py) -- None means composite scoring wasn't computed for
    # this result, in which case callers should fall back to `score`.
    composite_score: float | None = None
    composite_reasoning: list[str] = field(default_factory=list)


def _fresh_strategies() -> list[Strategy]:
    """
    New instances every call, at their default (i.e. actually-backtested)
    parameters -- _traded_today, _pending_confirmation, and cached levels
    are per-instance state that must never leak between symbols or between
    repeated scans of the same symbol at different times of day.

    VWAPStrategy, SmartMoneyStrategy, and FibonacciStrategy are included
    here (unlike ValidatedBreakout/ReversalDetection): all three use a
    single-bar retest+rejection design (see each strategy's own module
    docstring), not a 2-step candidate-then-confirmation-bar state
    machine, so a single cold evaluate() call CAN produce a signal --
    they aren't subject to this function's single-call limitation.

    RollingRangeBreakout (examples/rolling_range_breakout.py) is the same
    single-decisive-bar design as OpeningRangeBreakout, just with a
    trailing rolling window instead of a range fixed at market open -- see
    that module's docstring for why it exists as a separate strategy
    rather than a change to OpeningRangeBreakout itself.

    MomentumEntryStrategy (detectors/momentum_entry_strategy.py) is also a
    single-cold-call-safe design (its own move/volume/persistence gates
    are all computable from data already visible in one call, no 2-step
    state machine) -- built specifically to catch a stock that fires NO
    other strategy because it was never in the morning watchlist.
    """
    return [
        OpeningRangeBreakout(),
        RollingRangeBreakout(),
        ValidatedBreakout(),
        TrendContinuation(),
        ReversalDetection(),
        VWAPStrategy(),
        SmartMoneyStrategy(),
        FibonacciStrategy(),
        MomentumEntryStrategy(),
    ]


def _smc_pattern_context(direction: str, df_1min: pd.DataFrame, current_price: float) -> list[str]:
    """
    Purely additive reasoning context for an ALREADY-FIRED signal --
    checks whether a recent Smart Money Concepts order block/liquidity
    sweep, or a recently CONFIRMED classic chart pattern, lines up with
    the signal's own direction and sits near the current price. This
    does not create a signal, does not gate one, and is never called
    unless one of the 4 real Strategy classes already fired (see
    scan_symbol() below) -- same "additive, non-gating context" principle
    already used for compute_relative_strength() a few lines down from
    where this is called. See smart_money_concepts.py and
    pattern_recognition.py's own module docstrings for why neither of
    these is treated as validated, standalone evidence.
    """
    if direction not in ("long", "short"):
        return []

    atr = average_true_range(df_1min)
    if atr <= 0:
        return []
    near_tolerance = atr * 1.0
    reasons: list[str] = []

    order_blocks = find_order_blocks(df_1min)
    wanted_block_type = OrderBlockType.BULLISH if direction == "long" else OrderBlockType.BEARISH
    matching_blocks = [
        b for b in order_blocks if b.block_type == wanted_block_type
        and (b.low - near_tolerance) <= current_price <= (b.high + near_tolerance)
    ]
    if matching_blocks:
        block = max(matching_blocks, key=lambda b: b.candle_time)
        reasons.append(
            f"SMC context: price sits near a {block.block_type.value} order block from "
            f"{block.candle_time.strftime('%H:%M')} ({block.low:.2f}-{block.high:.2f}, preceded a "
            f"{block.move_size_atr:.1f} ATR move) -- unvalidated price-action pattern, not standalone proof."
        )

    sweeps = find_liquidity_sweeps(df_1min)
    wanted_sweep_dir = "up" if direction == "long" else "down"
    matching_sweeps = [s for s in sweeps if s.reversal_direction == wanted_sweep_dir]
    if matching_sweeps:
        sweep = matching_sweeps[-1]
        reasons.append(
            f"SMC context: liquidity sweep of {sweep.swept_level:.2f} at {sweep.sweep_time.strftime('%H:%M')} "
            f"reversed {sweep.reversal_direction} ({sweep.penetration_atr:.2f} ATR penetration) -- "
            "unvalidated price-action pattern, not standalone proof."
        )

    bullish_patterns = {PatternType.DOUBLE_BOTTOM, PatternType.INVERSE_HEAD_AND_SHOULDERS}
    bearish_patterns = {PatternType.DOUBLE_TOP, PatternType.HEAD_AND_SHOULDERS}
    wanted_pattern_types = bullish_patterns if direction == "long" else bearish_patterns

    all_patterns = find_double_top_bottom(df_1min) + find_head_and_shoulders(df_1min)
    matching_patterns = [
        p for p in all_patterns if p.confirmed and p.pattern_type in wanted_pattern_types
        and abs(p.neckline_price - current_price) <= near_tolerance
    ]
    if matching_patterns:
        pattern = matching_patterns[-1]
        reasons.append(
            f"Pattern context: confirmed {pattern.pattern_type.value.replace('_', ' ')} near current price "
            f"(neckline {pattern.neckline_price:.2f}) -- mixed academic evidence, treat as context only."
        )

    return reasons


def _apply_fixed_point_target(candidate: ScanResult) -> tuple[ScanResult | None, str]:
    """
    CURRENTLY UNCALLED (see FIXED_TARGET_POINTS' docstring -- reverted
    2026-07-22, kept for a possible smart_money-only follow-up test, not
    wired into scan_symbol()'s live_mode path anymore).

    CHANGE 1: overrides `candidate.target` with a fixed FIXED_TARGET_POINTS
    distance from entry, instead of the strategy's own backtested 2x-risk
    target -- see FIXED_TARGET_POINTS' own docstring for why this is safe
    (additive, live-path-only, backtest untouched).

    stop_loss is left completely unchanged (still the strategy's real,
    market-structure-based stop). Rejects (returns None, reason) any
    candidate whose R:R -- FIXED_TARGET_POINTS / stop_distance -- falls
    below MIN_FIXED_TARGET_RR, since a fixed target paired with a wide
    stop is a bad trade regardless of how the setup itself looks.
    """
    risk = abs(candidate.entry_price - candidate.stop_loss)
    if risk <= 0:
        return None, f"{candidate.symbol} ({candidate.setup_type}): rejected -- zero/invalid stop distance, cannot evaluate fixed-target R:R"

    rr = FIXED_TARGET_POINTS / risk
    if rr < MIN_FIXED_TARGET_RR:
        return None, (
            f"{candidate.symbol} ({candidate.setup_type}): rejected -- fixed {FIXED_TARGET_POINTS:.0f}pt target "
            f"vs {risk:.2f}pt stop distance gives R:R {rr:.2f}, below the {MIN_FIXED_TARGET_RR:.2f} minimum "
            f"(stop too wide relative to a fixed-point target to be worth the risk)"
        )

    sign = 1.0 if candidate.direction == "long" else -1.0
    fixed_target = candidate.entry_price + sign * FIXED_TARGET_POINTS
    updated = replace(
        candidate, target=fixed_target,
        reasoning=candidate.reasoning + [
            f"Fixed target: {FIXED_TARGET_POINTS:.0f} points from entry ({candidate.entry_price:.2f} -> "
            f"{fixed_target:.2f}), R:R {rr:.2f} against the {risk:.2f}pt stop (min required {MIN_FIXED_TARGET_RR:.2f})"
        ],
    )
    return updated, ""


def _apply_verification(
    candidate: ScanResult, signal: Signal, today_df: pd.DataFrame, current_index: pd.Timestamp,
) -> tuple[ScanResult | None, str]:
    """
    CHANGE 2: pre-entry momentum/volume/no-pullback verification -- see
    detectors/pick_verification.py's module docstring for the full
    rationale and the honest caveat on how trigger_time is inferred for
    this (single-call, no cross-call memory) fast-scan path specifically.
    No-op (returns candidate unchanged) for any strategy not in
    VERIFIED_STRATEGIES.
    """
    from detectors.pick_verification import VERIFIED_STRATEGIES, _infer_trigger_time, verify_pick

    if candidate.setup_type not in VERIFIED_STRATEGIES:
        return candidate, ""

    direction_enum = SignalDirection.LONG if candidate.direction == "long" else SignalDirection.SHORT
    trigger_level = signal.trigger_level
    trigger_time = (
        current_index if trigger_level is None
        else _infer_trigger_time(today_df, current_index, trigger_level, direction_enum)
    )
    result = verify_pick(today_df, candidate.entry_price, direction_enum, trigger_level, trigger_time)
    if not result.passed:
        return None, (
            f"{candidate.symbol} ({candidate.setup_type}): verification {result.status} -- "
            + "; ".join(result.reasoning)
        )

    updated = replace(
        candidate,
        reasoning=candidate.reasoning + [f"Pre-entry verification: {r}" for r in result.reasoning],
    )
    return updated, ""


def scan_symbol(symbol: str, df_1min: pd.DataFrame, as_of: pd.Timestamp,
                 benchmark_df: pd.DataFrame | None = None,
                 min_liquidity_value: float = 500_000.0,
                 live_mode: bool = False) -> ScanResult:
    """
    live_mode=False (default): unchanged exploratory scan behavior --
    Scanner tab / general /scan calls -- target is the strategy's real
    2x-risk-reward Signal.target, no pre-entry verification applied.
    live_mode=True: used specifically by daily_picks() (Daily Picks tab
    and paper_trading/watchlist.py, which live_poller.py's actual live
    trading watchlist is built from) -- applies CHANGE 2 (pre-entry
    momentum/volume/no-pullback verification, for the 4 strategies in
    detectors.pick_verification.VERIFIED_STRATEGIES) to every fired
    candidate before picking the best one; target stays the strategy's
    real 2x-risk-reward Signal.target (CHANGE 1's fixed-point target was
    tried here and reverted -- see FIXED_TARGET_POINTS' own docstring for
    why). A symbol whose only fired candidate(s) get rejected by
    verification returns setup_type "none" with the rejection reason(s)
    in `reasoning`, same as "no qualifying setup" -- these are real,
    loggable rejections, not silent.

    Evaluates ONE symbol's CURRENT, live-moment setup: each of the 7
    Strategy classes' evaluate() is called EXACTLY ONCE, using data
    truncated up to and including `as_of` (the latest cached bar), not a
    bar-by-bar replay of the day. This is O(1) strategy calls per symbol
    (7, fixed) instead of O(bars_since_open) -- deliberately, for Daily
    Picks' "scan the whole universe fast" use case (see daily_picks()),
    which does not need "what fired at any point earlier today," only
    "what's true right now."

    IMPORTANT, non-obvious consequence of single-call evaluation:
    ValidatedBreakout and ReversalDetection both gate their signal behind
    a two-step confirmation state (_pending_confirmation) -- a candidate
    is detected on one bar, then confirmed (or invalidated) on a LATER
    bar, and that state only exists if evaluate() was actually called on
    the earlier candidate bar. A single cold call from a fresh strategy
    instance can only ever reach the "look for a fresh candidate" branch
    (which sets pending state and returns None) -- it can never reach the
    "confirm a pending candidate" branch, because nothing was ever
    pending. Practically: under this function, ValidatedBreakout and
    ReversalDetection ALWAYS return None. OpeningRangeBreakout,
    TrendContinuation, VWAPStrategy, SmartMoneyStrategy, and
    FibonacciStrategy can all fire here: the first two compute their
    signal entirely from data visible in a single call (no cross-call
    state besides the daily _traded_today gate), and the latter three
    use a single-bar retest+rejection design where the "confirmation" IS
    the entry condition on that same bar, not a separate follow-through
    bar (see each strategy's own module docstring). This is a real,
    permanent limitation of "scan fast" mode for ValidatedBreakout/
    ReversalDetection specifically, not a bug to silently work around --
    if a scan needs their coverage, it needs the slower bar-by-bar replay
    this function used to do (still exactly what backtest/engine.py does
    for actual backtesting -- this function only changed how it powers a
    live/near-live universe scan).

    benchmark_df: optional index/benchmark data (e.g. NIFTY 50) for
        relative strength context on any setup found. Purely additive
        context, not a gate -- a good setup in an underperforming stock
        is still worth knowing about, just with that context attached.
    """
    today = as_of.date()
    truncated = df_1min.loc[:as_of]
    if truncated.empty:
        return ScanResult(symbol, 0.0, "none", "none", ["No data as of this timestamp"])

    today_bar_count = int((truncated.index.date == today).sum())
    if today_bar_count < 25:
        return ScanResult(symbol, 0.0, "none", "none",
                           ["Not enough bars yet today to evaluate"])

    # Price floor: reject before the liquidity gate or any strategy call --
    # see MIN_STOCK_PRICE's own docstring for why.
    current_price = float(truncated["close"].iloc[-1])
    if current_price < MIN_STOCK_PRICE:
        return ScanResult(symbol, 0.0, "below_min_price", "none",
                           [f"Price {current_price:.2f} below MIN_STOCK_PRICE ({MIN_STOCK_PRICE:.2f}), skipped"])

    # Liquidity pre-filter: reject before spending effort on strategy calls
    # -- a technically valid setup in an illiquid stock is still a bad trade
    # (see detectors/liquidity_analysis.py for why this is a proxy, not a
    # true depth/spread measurement). This isn't strategy entry/exit logic,
    # so it stays here rather than moving into the Strategy classes.
    liquidity = analyze_liquidity(truncated, min_avg_traded_value=min_liquidity_value)
    if liquidity is not None and not liquidity.is_liquid:
        return ScanResult(symbol, 0.0, "illiquid", "none", liquidity.reasoning)

    strategies = _fresh_strategies()
    fired: dict[str, Signal] = {}
    fired_entry_price: dict[str, float] = {}

    # Bound the resample input to the last few trading days, NOT the full
    # cached history (up to ~30 days). Profiled on real data: resampling
    # the full history was ~147ms/symbol -- 79% of total scan_symbol()
    # cost -- for a single-call scan that, per the docstring above, can
    # only ever be answered by OpeningRangeBreakout (today-only) and
    # TrendContinuation's regime gate (detect_regime() needs a minimum of
    # adx_period*3 = 42 bars of 15-min data -- by its OWN definition of
    # "enough history", not an arbitrary number picked here). Neither
    # needs 30 days. ValidatedBreakout/ReversalDetection would want
    # multi-day history for level detection, but they can never fire
    # through this single-call path regardless (see docstring), so
    # bounding their input doesn't change this function's output.
    # RECENT_DAYS_FOR_MTF=5 gives ~5*25=125 bars of 15-min data at typical
    # density -- ~3x detect_regime()'s own minimum, the same "3x period"
    # margin the codebase already uses elsewhere for warm-up. Honesty
    # note: unlike the ATR rolling-window bounding fixed earlier today,
    # this is NOT guaranteed byte-identical to feeding the full history --
    # detect_regime()'s ADX is Wilder-smoothed (a recursive EMA-like
    # process), so it technically depends on all prior history, just with
    # exponentially decaying weight. A 5-day window is a bounded
    # approximation (old data's influence is negligible by then), not a
    # mathematically exact truncation -- scoped to THIS fast-scan path
    # only; backtest/engine.py's TrendContinuation still sees full history
    # via its normal per-day incremental cache, unchanged.
    RECENT_DAYS_FOR_MTF = 5
    recent_dates = sorted({d for d in truncated.index.date})[-RECENT_DAYS_FOR_MTF:]
    bounded_for_mtf = truncated[truncated.index.date >= recent_dates[0]]

    mtf_view = multi_timeframe_view(bounded_for_mtf, [1, 5, 15])
    current_time = truncated.index[-1]
    current_bar = truncated.loc[current_time]

    for strategy in strategies:
        signal = strategy.evaluate(symbol, mtf_view, current_time)
        if signal is not None and signal.direction != SignalDirection.NONE:
            fired[strategy.name] = signal
            # Every strategy computes stop_loss/target off THIS bar's
            # close (verified across all 4) -- capturing it here is the
            # only reliable way to recover "entry price" since Signal
            # itself doesn't carry it.
            fired_entry_price[strategy.name] = float(current_bar["close"])

    if not fired:
        return ScanResult(symbol, 0.0, "none", "none", ["No qualifying setup right now"])

    candidates = [
        ScanResult(
            symbol, signal.confidence, name, signal.direction.value, list(signal.reasoning),
            entry_price=fired_entry_price[name], stop_loss=signal.stop_loss, target=signal.target,
        )
        for name, signal in fired.items()
    ]

    if live_mode:
        # CHANGE 1 (fixed 12-point target + min R:R filter) was REVERTED
        # 2026-07-22: a 400-symbol backtest showed it hurt 3 of 4 verified
        # strategies (vwap_reclaim and fibonacci_retracement notably --
        # both flipped from marginally profitable to unprofitable), most
        # likely because a flat point value doesn't scale the way
        # percentage/ATR-based targets do across this project's ~Rs
        # 100-30,000+ price range -- the same reason every OTHER threshold
        # in this codebase (stop-loss floors, early-exit triggers,
        # magnitude classification) is ATR-relative, not a fixed point
        # value. _apply_fixed_point_target() and its constants
        # (FIXED_TARGET_POINTS, MIN_FIXED_TARGET_RR) are left in place,
        # unused, rather than deleted -- smart_money's verified variant
        # showed a genuine improvement under the fixed-point system (40
        # trades, every metric up) worth isolating in a future test (does
        # verification alone drive that, or the fixed target specifically)
        # before deciding whether to bring this back for that one strategy.
        # CHANGE 2 (pre-entry verification) is UNCHANGED and still active --
        # see _apply_verification() below.
        rejection_notes: list[str] = []
        today_df = truncated[truncated.index.date == today]

        verified: list[ScanResult] = []
        for candidate in candidates:
            updated, note = _apply_verification(candidate, fired[candidate.setup_type], today_df, current_time)
            if updated is None:
                rejection_notes.append(note)
                continue
            verified.append(updated)
        candidates = verified

        if not candidates:
            reasoning = rejection_notes if rejection_notes else ["No qualifying setup right now"]
            return ScanResult(symbol, 0.0, "none", "none", reasoning)

    best = max(candidates, key=lambda c: c.score)

    if benchmark_df is not None and not benchmark_df.empty:
        rs_reading = compute_relative_strength(truncated, benchmark_df)
        if rs_reading is not None:
            best.reasoning = best.reasoning + rs_reading.reasoning

    smc_pattern_reasons = _smc_pattern_context(best.direction, bounded_for_mtf, best.entry_price)
    if smc_pattern_reasons:
        best.reasoning = best.reasoning + smc_pattern_reasons

    return best


def scan_universe(symbol_data: dict[str, pd.DataFrame], as_of: pd.Timestamp | None,
                   min_score: float = 0.5, benchmark_df: pd.DataFrame | None = None,
                   live_mode: bool = False) -> list[ScanResult]:
    """
    live_mode: passed straight through to scan_symbol() -- see that
        function's docstring. False (default) preserves this function's
        original exploratory-scan behavior; daily_picks() passes True.
    symbol_data: {symbol: full_1min_dataframe} for every symbol in the
        universe -- typically loaded from data/storage.py per symbol.
    as_of: a single shared timestamp applied to every symbol, OR None to
        scan each symbol as of ITS OWN most recent cached bar. Pass a fixed
        timestamp when you need every symbol compared at the exact same
        point in time (e.g. a backtest-style snapshot). Pass None when the
        goal is "the current state of the universe" and per-symbol data
        freshness legitimately varies -- which is the normal case for a
        bulk-fetched cache, since symbols get fetched at different times
        and don't all land on the same last trading day. A single shared
        as_of computed as max(every symbol's last bar) is a trap here: if
        even a handful of symbols are a day ahead of the rest (confirmed
        on real data -- 5 of 2373 cached symbols reached 2026-07-08 while
        2365 stopped at 2026-07-07), every symbol whose data doesn't reach
        that exact day gets silently treated as having "not enough bars
        today" and excluded entirely, even though it has a perfectly
        current, fully-formed trading day of its own to scan. This isn't
        hypothetical -- it's what actually happened the first time
        daily_picks() was run against the full cache (5 total setups
        found instead of hundreds).
    benchmark_df: optional NIFTY 50 (or other benchmark) data for
        relative strength context attached to results.
    Returns results sorted by score descending, filtered to min_score --
    this is the actual "give me the refined, high quality stocks" output.
    """
    results = []
    for symbol, df in symbol_data.items():
        symbol_as_of = as_of if as_of is not None else (df.index.max() if not df.empty else None)
        if symbol_as_of is None:
            continue
        try:
            result = scan_symbol(symbol, df, symbol_as_of, benchmark_df=benchmark_df, live_mode=live_mode)
        except Exception as e:
            # A single symbol's bad data shouldn't crash the whole scan --
            # but we surface it rather than silently skipping, since a
            # scanner that silently drops symbols on error is worse than
            # one that's loud about it.
            results.append(ScanResult(symbol, 0.0, "error", "none", [f"Scan error: {e}"]))
            continue
        if result.score >= min_score:
            results.append(result)

    return sorted(results, key=lambda r: r.score, reverse=True)


# Display-only target width for daily_picks(), NOT the strategy's own
# backtested R-multiple target. This is a deliberate, separate concept:
# the strategy's real `target` (2x/3x risk, whatever the firing strategy
# used) is what backtest/engine.py actually manages a trade to -- it can
# be a fraction of a percent or several percent away depending on the
# day's volatility, which is correct for backtesting but not what a
# "top picks" reader wants to see as a simple, comparable number across
# 10 very different stocks. 2.5% (midpoint of the requested 2-3% range)
# is used flat for every pick rather than varying it, to keep this
# honest: there is no backtested evidence that a 2.5% target is a good
# exit for these strategies specifically (that would require backtesting
# a percent-based target, which hasn't been done) -- treat this number as
# a display convenience only, not a validated trade plan. The stop-loss
# shown alongside it IS the real, backtested one (see ScanResult.stop_loss).
DAILY_PICKS_TARGET_PCT = 0.025


def display_target_price(entry_price: float, direction: str, pct: float = DAILY_PICKS_TARGET_PCT) -> float:
    """Shared formula behind DailyPick.target_price -- factored out so
    paper_trading/watchlist.py's confirmed_pick watchlist entries (which
    never go through daily_picks()) can show the same display-only target
    instead of a second, potentially-diverging computation."""
    sign = 1.0 if direction == "long" else -1.0
    return entry_price * (1.0 + sign * pct)


@dataclass
class DailyPick:
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    stop_loss: float  # the firing strategy's own stop-loss (real, backtested)
    target_price: float  # entry +/- DAILY_PICKS_TARGET_PCT (display-only, see note above)
    score: float
    setup_type: str
    reasoning: list[str]
    # The firing strategy's real, backtested 2x-risk-reward target --
    # same value run_universe_backtest.py/backtest research use. (CHANGE 1
    # briefly overrode this with a fixed FIXED_TARGET_POINTS target on
    # this live path; reverted 2026-07-22, see that constant's docstring.)
    target: float = 0.0
    # Multi-factor re-ranking score from detectors/composite_score.py, and
    # the human-readable breakdown behind it -- None if use_composite_score
    # was False or composite scoring had nothing to work with.
    composite_score: float | None = None
    composite_reasoning: list[str] | None = None


def daily_picks(
    symbol_data: dict[str, pd.DataFrame], as_of: pd.Timestamp | None = None, top_n: int = 5,
    benchmark_df: pd.DataFrame | None = None, use_composite_score: bool = True,
) -> dict[str, list[DailyPick]]:
    """
    Scans the full universe (same scan_universe() used everywhere else --
    no separate logic, per this module's "no feature addiction" rule) and
    splits qualifying setups into the top `top_n` longs ("buy") and top
    `top_n` shorts ("sell") by score.

    as_of defaults to None -- each symbol is scanned as of ITS OWN most
    recent cached bar, not one shared timestamp forced across the whole
    universe. See scan_universe()'s docstring for why: a single shared
    as_of computed as the max across all symbols silently excludes any
    symbol whose data doesn't happen to reach that exact day, which is
    the normal case (not the exception) for a bulk-fetched cache where
    symbols were fetched at different times. Pass an explicit timestamp
    only if you deliberately want a single frozen snapshot (e.g. testing).

    min_score=0.0 is passed to scan_universe() deliberately: this function
    wants to rank and slice ALL long/short candidates itself, not work
    from a pre-filtered (default min_score=0.5) subset that could hide a
    legitimate top-5 pick with a lower raw confidence score. The min_score
    default elsewhere exists to keep a general-purpose scan list free of
    noise; daily_picks() doesn't need that since take-the-top-N already
    does the filtering.

    KNOWN LIMITATION, inherited from scan_symbol()'s single-call design
    (see its docstring): ValidatedBreakout and ReversalDetection cannot
    fire through this fast-scan path at all -- both need a confirmation
    bar that a single cold evaluate() call can never supply. Every pick
    returned here will have setup_type "opening_range_breakout",
    "rolling_range_breakout", "trend_continuation", "vwap_reclaim",
    "smart_money", "fibonacci_retracement", or "momentum_entry" -- never
    the other two. This is a real, permanent tradeoff for scanning 2000+
    symbols in seconds rather than minutes, not a bug -- but it means
    "top 5 buy / top 5 sell" is really "top 5 from 7 of the 9 strategies,"
    which matters when judging pick quality or diversity.

    use_composite_score=True (default): after ranking/scoring, re-ranks
    each side by detectors.composite_score.compute_composite_score()
    (strategy confidence + momentum + profit potential + liquidity +
    sector alignment) instead of raw strategy confidence alone, before
    slicing to top_n -- see that module's docstring for the weights and
    the honesty caveats on the momentum/sector-alignment components.
    Requires computing sector momentum once across config/sector_
    mapping.json's ~180 symbols, which needs their data already present
    in `symbol_data` (true for any full-universe caller). Set False to
    skip this and rank by raw score only (e.g. for a bounded test subset
    that doesn't include sector-mapped symbols). Also where CHANGE 3's
    sector-context reasoning line gets attached (see _sector_context_line()) --
    purely informational, never a filter, even when this is False composite
    scoring itself is skipped but the reasoning line still is NOT added in
    that case (both live in _apply_composite_ranking()).

    CHANGE 2 (pre-entry verification): applied via scan_universe(...,
    live_mode=True) below -- see scan_symbol()'s docstring for exactly
    what that does. A pick that fails verification (failed or still-
    pending) simply doesn't appear here at all, same as any other "no
    qualifying setup." (CHANGE 1's fixed-point target was also applied on
    this path; reverted 2026-07-22 after a 400-symbol backtest showed it
    hurt 3 of 4 verified strategies -- target here is the firing
    strategy's real 2x-risk-reward value again, see FIXED_TARGET_POINTS'
    docstring.)
    """
    # live_mode=True: this IS the live Daily Picks path -- applies
    # CHANGE 2 (pre-entry verification) inside scan_symbol(). Target is
    # the strategy's real 2x-risk-reward value (CHANGE 1 reverted). The
    # Scanner tab's own /scan calls go through scan_universe()/scan_symbol()
    # directly (live_mode defaults False there), so they're unaffected.
    results = scan_universe(symbol_data, as_of, min_score=0.0, benchmark_df=benchmark_df, live_mode=True)
    longs = [r for r in results if r.direction == "long" and r.entry_price > 0]
    shorts = [r for r in results if r.direction == "short" and r.entry_price > 0]

    if use_composite_score and (longs or shorts):
        _apply_composite_ranking(longs, shorts, symbol_data, as_of)

    def to_pick(r: ScanResult) -> DailyPick:
        target_price = display_target_price(r.entry_price, r.direction)
        return DailyPick(
            symbol=r.symbol, direction=r.direction, entry_price=r.entry_price,
            stop_loss=r.stop_loss, target_price=target_price, target=r.target,
            score=r.score, setup_type=r.setup_type, reasoning=r.reasoning,
            composite_score=r.composite_score, composite_reasoning=r.composite_reasoning,
        )

    # longs/shorts are sorted by composite_score (if computed above) or
    # else scan_universe()'s original score-descending order.
    return {
        "buy": [to_pick(r) for r in longs[:top_n]],
        "sell": [to_pick(r) for r in shorts[:top_n]],
    }


def _apply_composite_ranking(
    longs: list[ScanResult], shorts: list[ScanResult],
    symbol_data: dict[str, pd.DataFrame], as_of: pd.Timestamp | None,
) -> None:
    """Computes composite_score in-place on every ScanResult in longs/
    shorts, then re-sorts both lists by it (descending). Lazy imports --
    detectors.composite_score imports ScanResult FROM this module, so a
    top-level import here would be circular."""
    import json

    from config.settings import PROJECT_ROOT
    from detectors.composite_score import compute_composite_score
    from detectors.sector_momentum import compute_sector_momentum, pick_reference_as_of

    mapping_path = PROJECT_ROOT / "config" / "sector_mapping.json"
    sector_mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    sector_symbol_data = {
        s: symbol_data[s] for s in sector_mapping if s in symbol_data and not symbol_data[s].empty
    }
    sector_as_of = pick_reference_as_of(sector_symbol_data) if sector_symbol_data else None
    sector_results = (
        compute_sector_momentum(sector_mapping, sector_symbol_data, as_of=sector_as_of)
        if sector_as_of is not None else []
    )
    sector_lookup = {r.sector: r for r in sector_results}

    for r in longs + shorts:
        df = symbol_data.get(r.symbol)
        if df is None or df.empty:
            continue
        symbol_as_of = as_of if as_of is not None else df.index.max()
        cs = compute_composite_score(r, df, symbol_as_of, sector_mapping, sector_lookup)
        r.composite_score = cs.total
        r.composite_reasoning = cs.reasoning
        # Sector context: purely INFORMATIONAL, appended to `reasoning`
        # (not composite_reasoning, and not a component of composite_score
        # itself -- composite_score.py's own sector_alignment component
        # already covers sector as a WEIGHTED factor; this is a separate,
        # human-readable line for the pick's own reasoning list, same
        # honest weak-signal disclaimer pattern the Sectors tab already
        # uses). Reuses sector_mapping/sector_lookup already computed just
        # above for composite scoring -- no second sector_momentum call.
        r.reasoning = r.reasoning + [_sector_context_line(r.symbol, r.direction, sector_mapping, sector_lookup)]

    def _rank_key(r: ScanResult) -> float:
        return -(r.composite_score if r.composite_score is not None else r.score)

    longs.sort(key=_rank_key)
    shorts.sort(key=_rank_key)


def _sector_context_line(
    symbol: str, direction: str, sector_mapping: dict[str, str], sector_lookup: dict,
) -> str:
    """CHANGE 3: additional informational reasoning line -- NOT a filter,
    NOT a scoring component. Matches the Sectors tab's own honest
    disclaimer that sector-level readings are a weak, unvalidated signal
    (see detectors/sector_momentum.py's module docstring) -- this is
    context for a human reading the pick, nothing more."""
    sector = sector_mapping.get(symbol)
    if sector is None:
        return "Sector context: no alignment data available (symbol not in the sector mapping) -- informational only, not a filter"
    reading = sector_lookup.get(sector)
    if reading is None:
        return f"Sector context: no {sector} sector reading available today -- informational only, not a filter"
    aligned = (reading.return_pct > 0 and direction == "long") or (reading.return_pct < 0 and direction == "short")
    return (
        f"Sector context: {sector} sector {reading.return_pct:+.2f}% today, "
        f"{'aligned with' if aligned else 'NOT aligned with'} this {direction} -- "
        f"informational only, not a filter (sector-level edge is unvalidated, see Sectors tab)"
    )
