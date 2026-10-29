"""
"Early Momentum" for the F8 Best Picks dashboard tab -- the deliberate
OPPOSITE of detectors/best_picks.py: Best Picks surfaces stocks already up
2%+ (often 6-9%+ by the time volume/persistence gates confirm them); this
surfaces a stock in the 0.8-2.5% window, catching the move BEFORE most of
it has happened. Entry is early and riskier by construction -- there is no
backtested edge behind this yet (see EARLY_MOMENTUM_DISCLAIMER in
api/main.py, kept visible at all times by the frontend, same convention as
every other informational section in this project).

Six gates, ALL must hold (direction inferred from move sign):
  1. Liquid enough to actually trade -- detectors/liquidity_analysis.py's
     analyze_liquidity(), SAME thresholds every other strategy/scanner
     path in this project uses (min_avg_traded_value=Rs.500,000/min
     default, max_dead_bar_pct=10% default). Checked FIRST, before any
     move/volume/breakout logic trusts this symbol's numbers -- same
     order and same reasoning as detectors/best_picks.py's gate 1: a
     thin stock produces an inflated, meaningless volume z-score (the
     BANG +180R / AFFLE +46R / DISAQ phantom-signal pattern), and this
     module's whole premise leans on volume z-score even more heavily
     than Best Picks does (>=2.0 here vs >=1.0 there), so trusting a
     thin stock's numbers here would be worse, not better.
  2. Move BETWEEN min_move_pct and max_move_pct (default 0.8%-2.5%) from
     today's open -- reuses detectors/sector_momentum.py's compute_stock_
     return_today() rather than a second "% from open" definition. The
     upper bound is the entire point of this module: a stock already
     past max_move_pct belongs in Best Picks, not here -- catching it
     here too would just be a redundant, later, worse-priced copy of
     that list.
  3. Volume z-score >= min_volume_zscore (default 2.0, noticeably
     stricter than Best Picks' 1.0) -- reuses detectors/breakout_
     validator.py's volume_zscore() over the same VOLUME_LOOKBACK_BARS
     window detectors/pick_verification.py uses. Set higher here
     deliberately: at a 0.8-2.5% move, price alone barely distinguishes
     a real institutional entry from routine noise -- an early,
     small move is only worth acting on if it's ALSO carrying unusually
     heavy volume for this specific stock.
  4. FRESH opening-range breakout -- broke through the opening-range
     high/low within the last breakout_freshness_minutes (default 15)
     and hasn't broken back since. Deliberately NOT detectors/strong_
     trend.py's check_opening_range_hold() (that one only asks "is it
     STILL holding, whenever it broke out" -- a breakout from 2 hours
     ago would pass): see check_fresh_opening_range_breakout() below --
     this module needs the breakout to be CURRENT, since "catch it
     before the big move" only makes sense if the move is still early.
  5. Sector alignment (only if the symbol has a sector mapping): reuses
     detectors/sector_momentum.py's compute_sector_momentum() to check
     the stock's sector is trending the same direction -- same gate,
     same "unmapped symbols pass vacuously" convention detectors/
     strong_trend.py's gate 5 already established (most of the ~2373-
     symbol universe has no official sector label; excluding all of them
     would gut this list down to the ~180 mapped symbols for no reason
     the codebase has validated). See sector_momentum.py's own module
     docstring for why sector agreement is never wired in as a hard
     STRATEGY-entry gate -- this is a display-only informational list,
     same distinction detectors/strong_trend.py's docstring already
     draws for its own identical gate.
  6. Price >= detectors.scanner.MIN_STOCK_PRICE (Rs.100) -- applied by
     the caller (api/main.py's get_early_momentum()) before symbol_data
     is even passed in here, same floor every other full-universe
     endpoint in this project uses -- not re-checked inside this module.

Ranked by volume_zscore * sector_alignment_score (a simple, NOT backtested
combination -- see _sector_alignment_score()'s docstring), top-N across
BOTH directions combined, same "best N right now" convention detectors/
best_picks.py uses rather than top-N-per-side.

Stop/target_1/target_2 reuse detectors/momentum_entry_strategy.py's
compute_swing_stop() (swing low/high over the last 10 bars, ATR-floored)
-- the SAME calculation Best Picks and MomentumEntryStrategy both already
use, not a third implementation that could silently drift from those.
target_1 = entry +/- 2x risk (partial-exit level), target_2 = entry +/-
3x risk (hold-remainder level) -- both display-only. If detectors/
strong_trend.py's own trend-consistency + volume-consistency checks ALSO
pass (not required to qualify -- purely an extra, informational note),
reasoning includes a suggestion to consider holding beyond target_2 with
a trailing stop, since that combination is what Strong Trend Stocks itself
looks for in a stock already further along the same kind of move.

Deliberately NOT added to detectors/pick_verification.py's
VERIFIED_STRATEGIES -- this has its own entry criteria (volume z-score +
sector alignment + fresh-breakout freshness window), so pick_verification's
generic re-verification would be redundant, same reasoning best_picks.py's
module docstring gives for the same decision. This is also NOT a
detectors/scanner.py Strategy -- like best_picks.py, it is a display
feature for the F8 tab, not wired into the scanner/live poller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backtest.strategy_base import SignalDirection
from detectors.breakout_validator import volume_zscore
from detectors.liquidity_analysis import analyze_liquidity
from detectors.momentum_entry_strategy import compute_swing_stop
from detectors.pick_verification import VOLUME_LOOKBACK_BARS
from detectors.sector_momentum import compute_sector_momentum, pick_reference_as_of
from detectors.strong_trend import (
    STRONG_TREND_MIN_CONSECUTIVE,
    STRONG_TREND_VOLUME_MIN_RATIO,
    STRONG_TREND_WINDOW_MINUTES,
    check_trend_consistency,
    check_volume_consistency,
)

EARLY_MOMENTUM_MIN_MOVE_PCT = 0.8
EARLY_MOMENTUM_MAX_MOVE_PCT = 2.5
EARLY_MOMENTUM_MIN_VOLUME_ZSCORE = 2.0
EARLY_MOMENTUM_OPENING_RANGE_MINUTES = 15
EARLY_MOMENTUM_BREAKOUT_FRESHNESS_MINUTES = 15
EARLY_MOMENTUM_COUNT = 3
EARLY_MOMENTUM_SWING_LOOKBACK_BARS = 10
EARLY_MOMENTUM_MIN_STOP_DISTANCE_ATR_MULT = 0.5
EARLY_MOMENTUM_TARGET1_RR = 2.0
EARLY_MOMENTUM_TARGET2_RR = 3.0


@dataclass
class EarlyMomentumPick:
    symbol: str
    direction: SignalDirection
    current_price: float
    entry_price: float  # same value as current_price -- named separately since callers ask for both
    move_pct_from_open: float
    volume_zscore: float
    sector: Optional[str]
    sector_aligned: Optional[bool]  # None if the symbol has no sector mapping (vacuous pass -- see gate 5)
    stop_loss: float
    risk_per_share: float  # |entry_price - stop_loss|
    target_1: float  # 1:2 risk-reward -- partial-exit level
    target_2: float  # 1:3 risk-reward -- hold-remainder level
    reasoning: list[str]
    rank_score: float  # volume_zscore x sector_alignment_score -- the ranking value, exposed for transparency


def check_fresh_opening_range_breakout(
    day_df: pd.DataFrame, as_of: pd.Timestamp, direction: SignalDirection,
    range_minutes: int, freshness_minutes: int,
) -> tuple[bool, str]:
    """
    Like detectors/strong_trend.py's check_opening_range_hold(), but with
    an added FRESHNESS requirement: the breakout itself (the first bar
    that closed beyond the opening-range level) must have happened within
    the last freshness_minutes of as_of, not merely "at some point today."
    A NOT-reused, separate function from check_opening_range_hold() rather
    than a parameter added to it -- that function has a second real caller
    (detectors/momentum_entry_strategy.py) that deliberately wants "still
    holding, no matter how long ago it broke out," so bolting a freshness
    requirement onto it would change behavior for that caller too.

    Still holding = same "hasn't closed back through the level since"
    condition check_opening_range_hold() checks, so a fresh-but-already-
    failed breakout still correctly fails this gate.
    """
    market_open = day_df.index[0]
    range_end = market_open + pd.Timedelta(minutes=range_minutes)
    range_bars = day_df[day_df.index < range_end]
    if range_bars.empty:
        return False, "Not enough bars yet to establish the opening range"
    range_high = float(range_bars["high"].max())
    range_low = float(range_bars["low"].min())
    post_range = day_df[day_df.index >= range_end]
    if post_range.empty:
        return False, "Not enough bars yet since the opening range closed"

    if direction == SignalDirection.LONG:
        broke_out = post_range[post_range["close"] > range_high]
        level_label = f"opening range high ({range_high:.2f})"
    else:
        broke_out = post_range[post_range["close"] < range_low]
        level_label = f"opening range low ({range_low:.2f})"

    if broke_out.empty:
        return False, f"Hasn't broken the {level_label} yet today"

    breakout_time = broke_out.index[0]
    age_minutes = (as_of - breakout_time).total_seconds() / 60.0
    if age_minutes > freshness_minutes:
        return False, (
            f"Broke the {level_label} at {breakout_time.strftime('%H:%M')} "
            f"({age_minutes:.0f} min ago) -- stale, not within the last {freshness_minutes} min"
        )

    after = post_range[post_range.index >= breakout_time]
    if direction == SignalDirection.LONG:
        broke_back = bool((after["close"] < range_high).any())
    else:
        broke_back = bool((after["close"] > range_low).any())
    if broke_back:
        return False, f"Broke back through the {level_label} after breaking out at {breakout_time.strftime('%H:%M')}"

    return True, (
        f"Fresh breakout of the {level_label} at {breakout_time.strftime('%H:%M')} "
        f"({age_minutes:.0f} min ago) -- still holding"
    )


def _sector_alignment_score(sector_return_pct: Optional[float]) -> float:
    """
    Numeric multiplier for ranking, NOT a validated weighting -- a simple,
    documented heuristic in the same spirit as detectors/best_picks.py's
    |move%| x volume_zscore rank_score. sector_return_pct=None (symbol has
    no sector mapping, gate 5 passed vacuously) gets a neutral 1.0 -- an
    unmapped symbol shouldn't rank ABOVE an equally-good mapped one just
    because it skipped the check, but it also shouldn't be penalized for
    a piece of data that doesn't exist for it. A mapped, aligned symbol's
    score scales with how strongly its sector itself is trending (a
    sector moving +3% is a stronger institutional-flow signal behind this
    stock's move than a sector barely positive at +0.2%), capped at 2.5
    so one very hot sector can't completely dominate the ranking over
    volume z-score.
    """
    if sector_return_pct is None:
        return 1.0
    return min(1.0 + abs(sector_return_pct) / 5.0, 2.5)


def find_early_momentum_picks(
    symbol_data: dict[str, pd.DataFrame],
    sector_mapping: dict[str, str],
    as_of: pd.Timestamp | None = None,
    min_move_pct: float = EARLY_MOMENTUM_MIN_MOVE_PCT,
    max_move_pct: float = EARLY_MOMENTUM_MAX_MOVE_PCT,
    min_volume_zscore: float = EARLY_MOMENTUM_MIN_VOLUME_ZSCORE,
    top_n: int = EARLY_MOMENTUM_COUNT,
) -> list[EarlyMomentumPick]:
    """
    symbol_data should already be price-filtered by the caller (see module
    docstring's gate 6). as_of=None picks the same shared reference
    timestamp compute_sector_momentum() would (see pick_reference_as_of) --
    pass it explicitly if the caller already computed it, so the reported
    as_of always matches what was actually used.
    """
    if as_of is None:
        as_of = pick_reference_as_of(symbol_data)
        if as_of is None:
            return []

    # Computed once for the whole batch (gate 5), not re-derived per
    # symbol -- reuses detectors/sector_momentum.py as-is, same pattern
    # detectors/strong_trend.py's find_strong_trend_stocks() uses.
    sector_results = compute_sector_momentum(sector_mapping, symbol_data, as_of=as_of)
    sector_return_by_name = {r.sector: r.return_pct for r in sector_results}

    candidates: list[EarlyMomentumPick] = []
    for symbol, df in symbol_data.items():
        truncated = df.loc[:as_of]
        today_df = truncated[truncated.index.date == as_of.date()]
        if today_df.empty or len(today_df) < VOLUME_LOOKBACK_BARS + 2:
            continue

        # Gate 1: liquidity -- checked FIRST, see module docstring.
        liquidity = analyze_liquidity(today_df)
        if liquidity is not None and not liquidity.is_liquid:
            continue  # gate 1 failed

        current_price = float(today_df["close"].iloc[-1])
        open_price = float(today_df["open"].iloc[0])
        if open_price <= 0:
            continue
        move_pct = (current_price - open_price) / open_price * 100.0

        if min_move_pct <= move_pct <= max_move_pct:
            direction = SignalDirection.LONG
        elif -max_move_pct <= move_pct <= -min_move_pct:
            direction = SignalDirection.SHORT
        else:
            continue  # gate 2 failed -- either too small (noise) or already too big (belongs in Best Picks)

        current_volume = float(today_df["volume"].iloc[-1])
        recent_volumes = today_df["volume"].iloc[-(VOLUME_LOOKBACK_BARS + 1):-1]
        vz = volume_zscore(current_volume, recent_volumes)
        if vz < min_volume_zscore:
            continue  # gate 3 failed

        fresh_ok, breakout_reason = check_fresh_opening_range_breakout(
            today_df, as_of, direction, EARLY_MOMENTUM_OPENING_RANGE_MINUTES, EARLY_MOMENTUM_BREAKOUT_FRESHNESS_MINUTES,
        )
        if not fresh_ok:
            continue  # gate 4 failed

        sector = sector_mapping.get(symbol)
        sector_aligned: Optional[bool] = None
        sector_return: Optional[float] = None
        if sector is not None and sector in sector_return_by_name:
            sector_return = sector_return_by_name[sector]
            sector_aligned = (sector_return > 0) if direction == SignalDirection.LONG else (sector_return < 0)
            if not sector_aligned:
                continue  # gate 5 failed -- sector trending against this stock

        stop, risk = compute_swing_stop(
            today_df, direction, current_price,
            EARLY_MOMENTUM_SWING_LOOKBACK_BARS, EARLY_MOMENTUM_MIN_STOP_DISTANCE_ATR_MULT,
        )
        if risk <= 0:
            continue
        sign = 1.0 if direction == SignalDirection.LONG else -1.0
        target_1 = current_price + sign * EARLY_MOMENTUM_TARGET1_RR * risk
        target_2 = current_price + sign * EARLY_MOMENTUM_TARGET2_RR * risk

        alignment_score = _sector_alignment_score(sector_return if sector_aligned else None)
        rank_score = vz * alignment_score

        reasoning = [
            f"Liquid: avg traded value Rs.{liquidity.avg_traded_value:,.0f}/min, "
            f"{liquidity.dead_bar_pct:.0%} dead bars" if liquidity is not None
            else "Liquidity: not enough bars yet to judge, treated as liquid",
            f"Moved {move_pct:+.2f}% from today's open -- still early "
            f"({min_move_pct:.1f}%-{max_move_pct:.1f}% window, not already extended)",
            f"Volume z-score {vz:.2f} (>= {min_volume_zscore:.1f} minimum) -- unusually heavy for this stock",
            breakout_reason,
        ]
        if sector is not None and sector_aligned is not None:
            reasoning.append(f"Sector '{sector}' trending {'up' if sector_return > 0 else 'down'} "
                              f"({sector_return:+.2f}%) -- aligned with this move")
        else:
            reasoning.append("No sector mapping for this symbol -- sector gate passed vacuously")

        trend_ok, _ = check_trend_consistency(
            today_df, as_of, direction, STRONG_TREND_WINDOW_MINUTES, STRONG_TREND_MIN_CONSECUTIVE,
        )
        volume_steady, _ = check_volume_consistency(today_df, STRONG_TREND_VOLUME_MIN_RATIO)
        if trend_ok and volume_steady:
            reasoning.append(
                "Also meets Strong Trend's persistence + volume-steady criteria -- if this continues, "
                "consider holding beyond target_2 with a trailing stop instead of exiting at 1:3"
            )

        reasoning.append(f"Rank score {rank_score:.2f} (= volume z-score x sector alignment score)")

        candidates.append(EarlyMomentumPick(
            symbol=symbol, direction=direction, current_price=current_price, entry_price=current_price,
            move_pct_from_open=move_pct, volume_zscore=vz, sector=sector, sector_aligned=sector_aligned,
            stop_loss=stop, risk_per_share=risk, target_1=target_1, target_2=target_2,
            rank_score=rank_score, reasoning=reasoning,
        ))

    candidates.sort(key=lambda c: c.rank_score, reverse=True)
    return candidates[:top_n]
