"""
"Today's Best 2 Picks" for the dedicated F8 Best Picks dashboard tab -- a
simple, purely informational (see BEST_PICKS_DISCLAIMER in api/main.py,
kept visible at all times by the frontend) top-2 ranked list. Deliberately
lighter-weight than detectors/momentum_entry_strategy.py's full Strategy
(no persistence-window or freshness gates) -- this is a display feature,
not a trade-entry signal wired into the scanner/live poller.

Six criteria, ALL must hold (direction inferred from move sign):
  1. Liquid enough to actually trade -- detectors/liquidity_analysis.py's
     analyze_liquidity(), SAME thresholds every other strategy/scanner
     path in this project uses (min_avg_traded_value=Rs.500,000/min
     default, max_dead_bar_pct=10% default) -- NOT a looser or separate
     threshold invented here. Added 2026-08-17 after DISAQ (a genuinely
     illiquid stock: Rs.98K/min avg traded value, 70% dead bars) passed
     every OTHER gate below and ranked #1, because a thin, near-zero
     volume baseline produces an inflated, meaningless z-score in gate 3
     -- the exact BANG (+180R) / AFFLE (+46R) phantom-signal pattern that
     is why every other backtest/live path in this codebase already gates
     on liquidity (see backtest/engine.py's run()/run_multi() and
     paper_trading/engine.py's on_new_bar()). This is intentionally
     checked FIRST, before any move/volume/persistence logic -- same
     order detectors/scanner.py's scan_symbol() uses (price floor ->
     liquidity -> strategy logic) -- so a thin stock is rejected before
     any of its numbers are trusted, not filtered out after the fact.
  2. Move >= min_move_pct (default 2.0%) from today's open -- reuses
     detectors/sector_momentum.py's compute_stock_return_today() rather
     than a second, potentially-diverging "% from open" definition.
  3. Volume z-score >= min_volume_zscore (default 1.0) -- reuses
     detectors/breakout_validator.py's volume_zscore(), over the same
     VOLUME_LOOKBACK_BARS window detectors/pick_verification.py uses.
  4. Last 3 FIVE-MINUTE bars not reversing against the move direction --
     resamples via data/resample.py (reused, not reimplemented) and
     checks the last 3 five-min closes. A simple, literal check, NOT
     detectors/strong_trend.py's heavier persistence gate (that one
     requires >=3 CONSECUTIVE higher-high+higher-low bars within a
     rolling 60-min window; this only rejects a clean, monotonic 2-step
     move against the direction in the last 3 five-min bars -- ordinary
     chop/pause passes through).
  5. Not broken back through the opening-range level -- reuses detectors/
     strong_trend.py's check_opening_range_hold() as-is (same condition
     examples/opening_range_breakout.py's check_invalidation() uses for a
     real ORB trade, checked across the whole day so far).
  6. Price >= detectors.scanner.MIN_STOCK_PRICE (Rs.100) -- applied by
     the caller (api/main.py's get_best_picks()) before symbol_data is
     even passed in here, same floor every other full-universe endpoint
     in this project uses -- not re-checked inside this module.

Ranked by |move_pct| * volume_zscore (strongest combined signal first),
top-N across BOTH directions combined -- not top-N-per-side, since the
brief is "the 2 best stocks right now," not "the best long and the best
short."

Stop/target_1/target_2 reuse detectors/momentum_entry_strategy.py's
compute_swing_stop() (swing low/high over the last 10 bars, ATR-floored)
-- the SAME calculation that strategy uses, not a second implementation
that could silently drift from it. target_1 = entry +/- 2x risk (the
partial-exit level), target_2 = entry +/- 3x risk (the hold-remainder-to-
EOD level) -- both display-only, like every other risk/reward number in
this project's informational sections. volume_label reuses detectors/
top_movers.py's classify_volume_zscore() (same high/normal/low bucketing,
not a third definition).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.strategy_base import SignalDirection
from data.resample import resample
from detectors.breakout_validator import volume_zscore
from detectors.liquidity_analysis import analyze_liquidity
from detectors.momentum_entry_strategy import compute_swing_stop
from detectors.pick_verification import VOLUME_LOOKBACK_BARS
from detectors.sector_momentum import compute_stock_return_today, pick_reference_as_of
from detectors.strong_trend import STRONG_TREND_OPENING_RANGE_MINUTES, check_opening_range_hold
from detectors.top_movers import classify_volume_zscore

BEST_PICKS_MIN_MOVE_PCT = 2.0
BEST_PICKS_MIN_VOLUME_ZSCORE = 1.0
BEST_PICKS_COUNT = 2
BEST_PICKS_SWING_LOOKBACK_BARS = 10
BEST_PICKS_MIN_STOP_DISTANCE_ATR_MULT = 0.5
BEST_PICKS_TARGET1_RR = 2.0
BEST_PICKS_TARGET2_RR = 3.0
BEST_PICKS_RESAMPLE_MINUTES = 5


@dataclass
class BestPick:
    symbol: str
    direction: SignalDirection
    current_price: float
    entry_price: float  # same value as current_price -- named separately since callers ask for both
    move_pct_from_open: float
    volume_zscore: float
    volume_label: str  # "high" | "normal" | "low" -- see detectors/top_movers.py's classify_volume_zscore()
    stop_loss: float
    risk_per_share: float  # |entry_price - stop_loss|
    target_1: float  # 1:2 risk-reward -- partial-exit level
    target_2: float  # 1:3 risk-reward -- hold-remainder-to-EOD level
    reasoning: list[str]
    rank_score: float  # |move_pct| x volume_zscore -- the ranking value, exposed for transparency


def _last_3_five_min_bars_not_reversing(day_df: pd.DataFrame, direction: SignalDirection) -> tuple[bool, str]:
    five_min = resample(day_df, BEST_PICKS_RESAMPLE_MINUTES)
    if len(five_min) < 3:
        return False, "Not enough 5-min bars yet to judge the last 3"
    closes = five_min["close"].iloc[-3:].to_numpy()
    if direction == SignalDirection.LONG:
        reversing = bool(closes[2] < closes[1] < closes[0])
    else:
        reversing = bool(closes[2] > closes[1] > closes[0])
    trail = f"{closes[0]:.2f} -> {closes[1]:.2f} -> {closes[2]:.2f}"
    if reversing:
        return False, f"Last 3 five-min bars reversing against the {direction.value} move ({trail})"
    return True, f"Last 3 five-min bars still moving with the {direction.value} ({trail})"


def find_best_picks(
    symbol_data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp | None = None,
    min_move_pct: float = BEST_PICKS_MIN_MOVE_PCT,
    min_volume_zscore: float = BEST_PICKS_MIN_VOLUME_ZSCORE,
    top_n: int = BEST_PICKS_COUNT,
) -> list[BestPick]:
    """
    symbol_data should already be price-filtered by the caller (see
    module docstring's gate 5). as_of=None picks the same shared
    reference timestamp compute_sector_momentum() would (see
    pick_reference_as_of) -- pass it explicitly if the caller already
    computed it, so the reported as_of always matches what was used.
    """
    if as_of is None:
        as_of = pick_reference_as_of(symbol_data)
        if as_of is None:
            return []

    candidates: list[BestPick] = []
    for symbol, df in symbol_data.items():
        truncated = df.loc[:as_of]
        today_df = truncated[truncated.index.date == as_of.date()]
        if today_df.empty or len(today_df) < VOLUME_LOOKBACK_BARS + 2:
            continue

        # Gate 1: liquidity -- checked FIRST, before any move/volume/
        # persistence logic trusts this symbol's numbers at all. Same
        # analyze_liquidity() defaults every other strategy/scanner path
        # uses (min_avg_traded_value=Rs.500,000/min, max_dead_bar_pct=10%)
        # -- see module docstring for the real DISAQ case this fixes.
        liquidity = analyze_liquidity(today_df)
        if liquidity is not None and not liquidity.is_liquid:
            continue  # gate 1 failed

        current_price = float(today_df["close"].iloc[-1])
        move_pct = compute_stock_return_today(df, as_of)
        if move_pct is None:
            continue

        if move_pct >= min_move_pct:
            direction = SignalDirection.LONG
        elif move_pct <= -min_move_pct:
            direction = SignalDirection.SHORT
        else:
            continue  # gate 2 failed

        current_volume = float(today_df["volume"].iloc[-1])
        recent_volumes = today_df["volume"].iloc[-(VOLUME_LOOKBACK_BARS + 1):-1]
        vz = volume_zscore(current_volume, recent_volumes)
        if vz < min_volume_zscore:
            continue  # gate 3 failed

        not_reversing, direction_reason = _last_3_five_min_bars_not_reversing(today_df, direction)
        if not not_reversing:
            continue  # gate 4 failed

        range_ok, range_reason = check_opening_range_hold(today_df, direction, STRONG_TREND_OPENING_RANGE_MINUTES)
        if not range_ok:
            continue  # gate 5 failed

        stop, risk = compute_swing_stop(
            today_df, direction, current_price, BEST_PICKS_SWING_LOOKBACK_BARS, BEST_PICKS_MIN_STOP_DISTANCE_ATR_MULT,
        )
        if risk <= 0:
            continue
        sign = 1.0 if direction == SignalDirection.LONG else -1.0
        target_1 = current_price + sign * BEST_PICKS_TARGET1_RR * risk
        target_2 = current_price + sign * BEST_PICKS_TARGET2_RR * risk
        rank_score = abs(move_pct) * vz

        candidates.append(BestPick(
            symbol=symbol, direction=direction, current_price=current_price, entry_price=current_price,
            move_pct_from_open=move_pct, volume_zscore=vz, volume_label=classify_volume_zscore(vz),
            stop_loss=stop, risk_per_share=risk, target_1=target_1, target_2=target_2, rank_score=rank_score,
            reasoning=[
                f"Liquid: avg traded value Rs.{liquidity.avg_traded_value:,.0f}/min, "
                f"{liquidity.dead_bar_pct:.0%} dead bars" if liquidity is not None
                else "Liquidity: not enough bars yet to judge, treated as liquid",
                f"Moved {move_pct:+.2f}% from today's open -- passes the {min_move_pct:.1f}% minimum",
                f"Volume z-score {vz:.2f} (>= {min_volume_zscore:.1f} minimum) -- volume confirmed, not thin",
                direction_reason,
                range_reason,
                f"Rank score {rank_score:.2f} (= |move%| x volume z-score)",
            ],
        ))

    candidates.sort(key=lambda c: c.rank_score, reverse=True)
    return candidates[:top_n]
