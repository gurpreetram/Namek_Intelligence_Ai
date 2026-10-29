"""
Composite, multi-factor ranking score for confirmed picks (see detectors/
scanner.py's ScanResult / daily_picks()) -- built at explicit user request
to rank Daily Picks by more than the firing strategy's own confidence
score alone.

Deliberately reuses EXISTING, already-computed detector output rather
than inventing new indicators: liquidity_analysis.py's analyze_liquidity(),
sector_momentum.py's compute_sector_momentum()/compute_daily_atr_pct()
(the same ATR-relative normalization Correction 4 built for the Sectors
tab), and the firing strategy's own real Signal.target (see
ScanResult.target's docstring for why that's used instead of daily_picks()'s
flat 2.5% display target).

HONESTY NOTE, important before trusting this ranking: no other work in
this project has found sector-momentum or short-horizon momentum to be a
validated, backtested predictive edge on their own -- see detectors/
sector_momentum.py's module docstring and the research scripts it
references (both found no real edge over random selection across multiple
lookback/forward combinations). Both are used here anyway, at explicit
user request, as reasonable RANKING heuristics among an already-confirmed
set of picks (i.e. "given these already-fired setups, which look more
favorable"), not because either has been shown to predict outcomes on its
own. This is a sensible, explainable weighted sum -- not an ML-tuned or
backtested model. Treat the specific weights below as a reasonable
starting point, not a proven optimum.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from detectors.liquidity_analysis import analyze_liquidity
from detectors.scanner import ScanResult
from detectors.sector_momentum import (
    SectorMomentum,
    classify_move_magnitude,
    compute_daily_atr_pct,
    compute_stock_return_today,
)

# Default weights, summing to 1.0.
#   strategy_confidence (0.30): the strategy's own Signal.confidence --
#     still the single most direct "is this a real, confirmed setup"
#     signal (a structured detector output, not a heuristic), but no
#     longer the ONLY input.
#   momentum (0.20): today's move-so-far, direction-aligned and
#     normalized against the stock's OWN typical daily range. Meaningful
#     context, but explicitly not shown elsewhere in this codebase to be
#     a validated predictive signal on its own, so kept below
#     strategy_confidence.
#   profit_potential (0.15): the REAL strategy target's % distance from
#     entry. Informative, but a wide target usually comes with a wide
#     stop too (bigger reward often means bigger risk), so this is
#     deliberately one of the smaller weights, not a dominant one.
#   liquidity (0.15): how comfortably tradeable the stock is. Every
#     candidate reaching this stage already passed the hard is_liquid
#     gate in scan_symbol(), so this only differentiates "just liquid
#     enough" from "very liquid" -- real, but secondary, information.
#   sector_alignment (0.20): whether the stock's sector is trending the
#     same direction. Meaningful additive context, but sector_momentum.py
#     is explicitly documented elsewhere as having shown no real
#     predictive edge in isolated testing, so 0.20 is already a generous
#     weight for an unvalidated signal, not something to raise further
#     without new evidence.
DEFAULT_WEIGHTS: dict[str, float] = {
    "strategy_confidence": 0.30,
    "momentum": 0.20,
    "profit_potential": 0.15,
    "liquidity": 0.15,
    "sector_alignment": 0.20,
}

# Reference bands used to normalize each raw measurement onto a 0-1 scale.
# Simple, explainable choices -- not backtested/fit to data.
PROFIT_POTENTIAL_REFERENCE_PCT = 3.0  # a 3%+ real target distance scores ~max
LIQUIDITY_REFERENCE_VALUE = 5_000_000.0  # 10x analyze_liquidity()'s own min_avg_traded_value default
SECTOR_ALIGNMENT_REFERENCE_PCT = 1.0  # a sector move of +/-1% since open scores ~max/min


@dataclass
class CompositeScore:
    total: float  # 0-1, higher = ranked higher
    components: dict[str, float]  # component_name -> raw 0-1 sub-score, ONLY for components that were available
    weights_used: dict[str, float]  # component_name -> weight actually applied, after redistributing any missing components
    reasoning: list[str]  # human-readable breakdown, one line per component + a total line


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_composite_score(
    pick: ScanResult,
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    sector_mapping: dict[str, str],
    sector_lookup: dict[str, SectorMomentum],
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> CompositeScore:
    """
    pick: an already-confirmed ScanResult (direction is "long"/"short",
        entry_price/stop_loss/target/score all set by scan_symbol()).
    df: the SAME symbol's full cached 1-min history (used for momentum's
        ATR normalization, which needs prior trading days -- see
        detectors.sector_momentum.compute_daily_atr_pct()).
    sector_lookup: {sector_name: SectorMomentum}, from a single shared
        detectors.sector_momentum.compute_sector_momentum() call the
        caller runs ONCE per batch, not once per symbol -- sector
        momentum is a universe-wide computation, too expensive to redo
        per pick.

    Missing-data handling: any component that can't be computed for this
    symbol (no sector mapping, not enough prior-day history for ATR, etc.)
    is dropped entirely -- NOT scored as 0 (that would penalize missing
    data as if it were bad data) -- and the other available components'
    weights are scaled up proportionally so they still sum to 1.0.
    """
    sign = 1.0 if pick.direction == "long" else -1.0
    raw_components: dict[str, float] = {}
    reasoning: list[str] = []

    # 1. Strategy confidence -- already 0-1, used as-is.
    raw_components["strategy_confidence"] = _clamp01(pick.score)
    reasoning.append(f"strategy_confidence: {pick.score:.2f} (the firing strategy's own signal confidence)")

    # 2. Momentum -- today's move-so-far, direction-aligned, normalized
    # against this stock's OWN typical daily range (reuses Correction 4's
    # compute_daily_atr_pct()/classify_move_magnitude() from detectors/
    # sector_momentum.py). A move equal to the full typical daily range
    # scores 1.0; no move (or a move against the pick's direction) scores 0.
    return_pct = compute_stock_return_today(df, as_of)
    atr_pct = compute_daily_atr_pct(df, as_of) if return_pct is not None else None
    if return_pct is not None and atr_pct is not None and atr_pct > 0:
        aligned_pct = return_pct * sign
        momentum_score = _clamp01(aligned_pct / atr_pct)
        raw_components["momentum"] = momentum_score
        label = classify_move_magnitude(return_pct, atr_pct)
        reasoning.append(
            f"momentum: {momentum_score:.2f} (today's move {return_pct:+.2f}% since open, "
            f"{'aligned with' if aligned_pct > 0 else 'against'} the {pick.direction} direction, "
            f"a '{label}' move vs this stock's own typical daily range)"
        )
    else:
        reasoning.append("momentum: unavailable (not enough prior-day history cached for this symbol) -- weight redistributed")

    # 3. Profit potential -- the REAL strategy target's % distance from
    # entry (NOT daily_picks()'s flat 2.5% display target, which is
    # identical for every pick and so carries no ranking information --
    # see ScanResult.target's docstring).
    if pick.entry_price > 0 and pick.target > 0:
        profit_pct = abs(pick.target - pick.entry_price) / pick.entry_price * 100.0
        profit_score = _clamp01(profit_pct / PROFIT_POTENTIAL_REFERENCE_PCT)
        raw_components["profit_potential"] = profit_score
        reasoning.append(
            f"profit_potential: {profit_score:.2f} (real target is {profit_pct:.2f}% from entry, "
            f"vs a {PROFIT_POTENTIAL_REFERENCE_PCT:.1f}% reference for a max score)"
        )
    else:
        reasoning.append("profit_potential: unavailable (no real target on this signal) -- weight redistributed")

    # 4. Liquidity -- analyze_liquidity() already gates out anything
    # illiquid before a pick can exist at all (see scan_symbol()), so
    # every candidate reaching this function already cleared that bar;
    # this only differentiates how comfortably above it a symbol sits.
    liquidity = analyze_liquidity(df.loc[:as_of])
    if liquidity is not None:
        value_component = _clamp01(liquidity.avg_traded_value / LIQUIDITY_REFERENCE_VALUE)
        liquidity_score = _clamp01(value_component * (1.0 - liquidity.dead_bar_pct))
        raw_components["liquidity"] = liquidity_score
        reasoning.append(
            f"liquidity: {liquidity_score:.2f} (avg traded value Rs.{liquidity.avg_traded_value:,.0f}/min, "
            f"{liquidity.dead_bar_pct:.0%} dead bars)"
        )
    else:
        reasoning.append("liquidity: unavailable (not enough recent bars to assess) -- weight redistributed")

    # 5. Sector alignment -- only for the ~180/2373 symbols with an
    # official sector classification (config/sector_mapping.json); see
    # detectors/sector_momentum.py's module docstring for the explicit
    # "no validated predictive edge found" caveat on this signal.
    sector = sector_mapping.get(pick.symbol)
    sector_reading = sector_lookup.get(sector) if sector else None
    if sector_reading is not None:
        aligned_sector_pct = sector_reading.return_pct * sign
        sector_score = _clamp01(0.5 + aligned_sector_pct / (2.0 * SECTOR_ALIGNMENT_REFERENCE_PCT))
        raw_components["sector_alignment"] = sector_score
        reasoning.append(
            f"sector_alignment: {sector_score:.2f} (sector '{sector}' is {sector_reading.return_pct:+.2f}% "
            f"since open, {'aligned with' if aligned_sector_pct > 0 else 'against'} the {pick.direction} direction)"
        )
    else:
        reason = "no sector mapping for this symbol" if not sector else "sector has no current reading"
        reasoning.append(f"sector_alignment: unavailable ({reason}) -- weight redistributed")

    # Redistribute any missing components' weight proportionally among
    # whatever IS available, so the final weights always sum to 1.0 and no
    # symbol is penalized purely for missing data (e.g. the ~92% of the
    # universe with no sector mapping).
    available_weight_total = sum(weights[name] for name in raw_components)
    if available_weight_total <= 0:
        # Every component was unavailable -- fall back to raw strategy
        # confidence alone rather than returning a meaningless score.
        return CompositeScore(total=pick.score, components={}, weights_used={},
                               reasoning=reasoning + ["all components unavailable -- fell back to raw strategy_confidence"])

    weights_used = {name: weights[name] / available_weight_total for name in raw_components}
    total = sum(raw_components[name] * weights_used[name] for name in raw_components)

    reasoning.append("weights used: " + ", ".join(f"{name}={weights_used[name]:.2f}" for name in raw_components))
    reasoning.append(f"composite_score total: {total:.3f}")

    return CompositeScore(total=total, components=raw_components, weights_used=weights_used, reasoning=reasoning)
