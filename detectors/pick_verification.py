"""
Pre-entry verification layer for live Daily Picks / live trading: a
strategy matching its rule-based entry condition is NECESSARY but not
SUFFICIENT to treat a setup as genuinely tradeable right now. This module
adds three additional checks, directly targeting "why do stocks reverse
right after being shown as a pick":

  1. Minimum move (check_minimum_move): the stock must have actually
     moved a meaningful amount from today's open, not just technically
     crossed a level by a hair.
  2. Volume confirmation (check_volume_confirmation): the move must be
     backed by genuinely elevated volume -- reuses detectors/
     breakout_validator.py's volume_zscore, the same z-score convention
     already used elsewhere in this project, not a second, potentially-
     diverging threshold invented here.
  3. No immediate pullback (check_continuation): price must hold beyond
     the strategy's own trigger_level (Signal.trigger_level -- the exact
     level broken/reclaimed/retested) for a few bars AFTER the trigger,
     not reverse straight back through it.

Check 3 is the only one that needs bars AFTER the trigger to exist, and
the two real callers get there differently:
  - detectors/scanner.py's fast-scan path re-evaluates with a FRESH
    strategy instance every call (no cross-call memory), so there's no
    strategy-reported "this is the bar that fired" from a previous call
    to anchor on. _infer_trigger_time() recovers a reasonable proxy: the
    start of the unbroken run of bars where price has stayed beyond
    trigger_level, ending at the current bar. This is an approximation,
    not the strategy's own true firing moment -- documented honestly
    here because it's a real design compromise, not hidden. A related,
    harder limitation: if the strategy's own evaluate() ALSO gates on
    THIS bar's volume being elevated (true for all four strategies this
    module applies to), a later scan can stop reporting a Signal at all
    once volume normalizes a few bars after the real breakout -- meaning
    fast-scan mode's chance to even reach this verification layer can
    close faster than the continuation check needs to resolve. This is
    NOT a problem for the second caller below.
  - paper_trading/engine.py's true bar-by-bar live streaming holds the
    candidate the INSTANT it first fires (in a _verifying state) and
    re-runs check_continuation() on each subsequent real bar as it
    arrives -- no re-derivation of the trigger bar needed, and no risk of
    the underlying Signal stopping to fire, since the signal was already
    captured. This is the more rigorous of the two, and it's the one
    that actually risks/tracks paper capital.

Applied to: opening_range_breakout, rolling_range_breakout, vwap_reclaim,
smart_money, fibonacci_retracement (VERIFIED_STRATEGIES below) -- the five
strategies whose "confirmation" is a single decisive bar (a level cross /
reclaim / retest+rejection) with no existing multi-bar confirmation
mechanism of their own (rolling_range_breakout is the same single-bar
design as opening_range_breakout, just with a trailing window -- see
examples/rolling_range_breakout.py). Deliberately NOT applied to validated_breakout /
reversal_detection (both already gate behind their own 2-step candidate-
then-follow-through-bar state machine -- layering this on top would be
redundant, not additive) or trend_continuation (its signal is a
persistently-rechecked regime condition, not a point-in-time trigger --
"did it immediately reverse after the trigger bar" isn't the right
question for a condition that's continuously true or false, not a
one-off event; its existing check_invalidation() already serves the
analogous post-entry role).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import pandas as pd

from backtest.strategy_base import Signal, SignalDirection
from detectors.breakout_validator import volume_zscore

# Lowered from 1.0 to 0.5 (2026-08-17): the 1.0% floor was rejecting real
# movers early -- e.g. a stock later confirmed up +9.87% could still be
# sitting at only 0.8% at the moment its strategy fired, and got voided
# here before ever getting a chance to run. Volume confirmation
# (MIN_VOLUME_ZSCORE) and the 3-bar no-pullback check
# (CONTINUATION_CONFIRMATION_BARS) are UNCHANGED -- this only relaxes how
# far price must already have moved, not how real/sustained that move
# must look once it's happened.
MIN_MOVE_PCT = 0.5
MIN_VOLUME_ZSCORE = 1.0
CONTINUATION_CONFIRMATION_BARS = 3
VOLUME_LOOKBACK_BARS = 20
TRIGGER_INFERENCE_LOOKBACK_BARS = 15  # see _infer_trigger_time()

VERIFIED_STRATEGIES = frozenset({
    "opening_range_breakout", "rolling_range_breakout", "vwap_reclaim", "smart_money", "fibonacci_retracement",
})


@dataclass
class VerificationResult:
    status: Literal["passed", "failed", "pending"]
    reasoning: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def check_minimum_move(
    day_df: pd.DataFrame, current_price: float, direction: SignalDirection, min_pct: float = MIN_MOVE_PCT,
) -> tuple[bool, str]:
    """% move from today's first bar's open, in the trade's own direction
    -- same "since market open" reference point detectors/
    sector_momentum.py's compute_stock_return_today() already uses, not a
    second, potentially-diverging definition."""
    if day_df.empty:
        return False, "No data for today -- cannot compute move from open"
    open_price = float(day_df["open"].iloc[0])
    if open_price <= 0:
        return False, "Invalid opening price"
    move_pct = (current_price - open_price) / open_price * 100.0
    if direction == SignalDirection.SHORT:
        move_pct = -move_pct
    if move_pct >= min_pct:
        return True, f"Moved {move_pct:+.2f}% from today's open ({open_price:.2f}) -- passes {min_pct:.1f}% minimum"
    return False, (
        f"Only moved {move_pct:+.2f}% from today's open ({open_price:.2f}) -- "
        f"below the {min_pct:.1f}% minimum, too weak to trust"
    )


def check_volume_confirmation(
    day_df: pd.DataFrame, lookback_bars: int = VOLUME_LOOKBACK_BARS, min_zscore: float = MIN_VOLUME_ZSCORE,
) -> tuple[bool, str]:
    """Reuses detectors/breakout_validator.py's volume_zscore -- same
    z-score convention already used for breakout confirmation elsewhere,
    not a second threshold invented here."""
    if len(day_df) < lookback_bars + 2:
        return False, f"Not enough bars yet today ({len(day_df)}) to judge volume confirmation"
    current_volume = float(day_df["volume"].iloc[-1])
    recent_volumes = day_df["volume"].iloc[-(lookback_bars + 1):-1]
    z = volume_zscore(current_volume, recent_volumes)
    if z >= min_zscore:
        return True, f"Volume z-score {z:.2f} (vs {lookback_bars}-bar recent average) -- genuinely elevated"
    return False, (
        f"Volume z-score {z:.2f} -- not meaningfully elevated (need >= {min_zscore:.1f}), "
        f"move may lack real participation"
    )


def _infer_trigger_time(
    day_df: pd.DataFrame, current_index: pd.Timestamp, trigger_level: float, direction: SignalDirection,
    max_lookback_bars: int = TRIGGER_INFERENCE_LOOKBACK_BARS,
) -> pd.Timestamp:
    """Best-effort recovery of WHEN price first crossed trigger_level, for
    callers (detectors/scanner.py) that don't have the strategy's own
    cross-call memory of its firing bar. Walks backward from
    current_index while price has stayed continuously beyond
    trigger_level, bounded to max_lookback_bars -- see module docstring's
    honesty note on this being an approximation."""
    window = day_df.loc[:current_index].iloc[-max_lookback_bars:]
    if window.empty:
        return current_index
    beyond = window["close"] > trigger_level if direction == SignalDirection.LONG else window["close"] < trigger_level
    if not bool(beyond.iloc[-1]):
        return current_index
    idx = len(beyond) - 1
    while idx > 0 and bool(beyond.iloc[idx - 1]):
        idx -= 1
    return window.index[idx]


def check_continuation(
    day_df: pd.DataFrame, trigger_time: pd.Timestamp, trigger_level: Optional[float], direction: SignalDirection,
    confirmation_bars: int = CONTINUATION_CONFIRMATION_BARS,
) -> tuple[str, str]:
    """
    Returns (status, reason), status one of "passed" | "failed" | "pending".

    "pending": fewer than `confirmation_bars` bars exist yet after
      trigger_time -- not a rejection, just not decidable yet.
    "failed": price closed back through trigger_level at some point in
      the confirmation window, OR the low/high sequence net-reversed
      (lower lows for a long, higher highs for a short) across it -- the
      breakout/reclaim/rejection didn't hold.
    "passed": price held beyond trigger_level for the whole window, with
      a net higher-low (long) / lower-high (short) trend across it.
    """
    if trigger_level is None:
        return "passed", "Strategy did not provide a trigger_level -- continuation check skipped"

    after_trigger = day_df[day_df.index > trigger_time]
    if len(after_trigger) < confirmation_bars:
        return "pending", (
            f"Only {len(after_trigger)}/{confirmation_bars} confirmation bars available since the "
            f"trigger bar -- not yet decidable"
        )

    window = after_trigger.iloc[:confirmation_bars]

    if direction == SignalDirection.LONG:
        if bool((window["close"] < trigger_level).any()):
            return "failed", (
                f"Price closed back below the trigger level ({trigger_level:.2f}) within "
                f"{confirmation_bars} bars of the breakout -- setup failed, not surfaced"
            )
        if float(window["low"].iloc[-1]) < float(window["low"].iloc[0]):
            return "failed", (
                f"Lows trended DOWN over the {confirmation_bars} bars after the trigger "
                f"(expected higher lows for a long) -- momentum did not confirm"
            )
        return "passed", (
            f"Held above the trigger level ({trigger_level:.2f}) with higher lows across "
            f"{confirmation_bars} bars since the breakout"
        )
    else:
        if bool((window["close"] > trigger_level).any()):
            return "failed", (
                f"Price closed back above the trigger level ({trigger_level:.2f}) within "
                f"{confirmation_bars} bars of the breakdown -- setup failed, not surfaced"
            )
        if float(window["high"].iloc[-1]) > float(window["high"].iloc[0]):
            return "failed", (
                f"Highs trended UP over the {confirmation_bars} bars after the trigger "
                f"(expected lower highs for a short) -- momentum did not confirm"
            )
        return "passed", (
            f"Held below the trigger level ({trigger_level:.2f}) with lower highs across "
            f"{confirmation_bars} bars since the breakdown"
        )


def verify_pick(
    day_df: pd.DataFrame, current_price: float, direction: SignalDirection,
    trigger_level: Optional[float], trigger_time: pd.Timestamp,
    min_move_pct: float = MIN_MOVE_PCT, min_volume_zscore: float = MIN_VOLUME_ZSCORE,
    confirmation_bars: int = CONTINUATION_CONFIRMATION_BARS,
) -> VerificationResult:
    """
    Runs all three checks in order. Overall status:
      - "failed" if EITHER the move/volume check fails (both checked
        instantly, at the trigger bar) OR the continuation check
        explicitly fails.
      - "pending" if move/volume both pass but continuation hasn't had
        enough bars yet to resolve -- the caller should wait, not reject.
      - "passed" only if all three pass.
    day_df: today's 1-min bars for this symbol, truncated to no later
        than the caller's current point of evaluation (no lookahead).
    """
    reasoning: list[str] = []

    move_ok, move_reason = check_minimum_move(day_df, current_price, direction, min_move_pct)
    reasoning.append(move_reason)
    if not move_ok:
        return VerificationResult(status="failed", reasoning=reasoning)

    vol_ok, vol_reason = check_volume_confirmation(day_df, min_zscore=min_volume_zscore)
    reasoning.append(vol_reason)
    if not vol_ok:
        return VerificationResult(status="failed", reasoning=reasoning)

    cont_status, cont_reason = check_continuation(day_df, trigger_time, trigger_level, direction, confirmation_bars)
    reasoning.append(cont_reason)
    return VerificationResult(status=cont_status, reasoning=reasoning)


def apply_fixed_point_target_to_signal(signal: Signal, reference_price: float) -> Optional[Signal]:
    """
    CHANGE 1 (fixed point-distance target + minimum R:R rejection),
    operating directly on a raw Signal -- the single shared implementation
    both paper_trading/engine.py's PaperTradingEngine and backtest/
    engine.py's run_multi() (verification_config) use, so "what does a
    12-point fixed target mean" is answered in exactly one place. A
    Signal has no entry_price of its own (the real fill only happens at
    the NEXT bar's open) -- `reference_price` is the signal-time bar's
    close, the same convention detectors/scanner.py's scan_symbol() uses
    for fired_entry_price, so a dashboard pick, a live paper trade, and a
    backtested trade all agree on what "12 points from entry" means.

    Stop-loss is left completely unchanged (still the strategy's real,
    market-structure-based stop). Returns None if the resulting R:R --
    FIXED_TARGET_POINTS / stop_distance -- falls below MIN_FIXED_TARGET_RR;
    caller is responsible for logging/voiding the rejected signal.

    Lazy import of detectors.scanner's constants (not at module level):
    detectors/scanner.py imports strategies that import backtest.engine.Trade
    for type hints, so a module-level import here would risk a circular
    import if this module is ever imported while backtest/engine.py itself
    is still mid-load -- see that call site's own comment for the concrete
    case (run_multi()'s verification_config, a lazy import for the same
    reason). Safe either way since paper_trading/engine.py already imports
    detectors.scanner directly at module level with no issue.
    """
    from detectors.scanner import FIXED_TARGET_POINTS, MIN_FIXED_TARGET_RR

    risk = abs(reference_price - signal.stop_loss)
    if risk <= 0:
        return None
    rr = FIXED_TARGET_POINTS / risk
    if rr < MIN_FIXED_TARGET_RR:
        return None
    sign = 1.0 if signal.direction == SignalDirection.LONG else -1.0
    fixed_target = reference_price + sign * FIXED_TARGET_POINTS
    return replace(
        signal, target=fixed_target,
        reasoning=signal.reasoning + [
            f"Fixed target: {FIXED_TARGET_POINTS:.0f} points from {reference_price:.2f} -> {fixed_target:.2f}, "
            f"R:R {rr:.2f} against the {risk:.2f}pt stop (min required {MIN_FIXED_TARGET_RR:.2f})"
        ],
    )
