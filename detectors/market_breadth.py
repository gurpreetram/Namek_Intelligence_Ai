"""
Market Breadth: how many NSE stocks are advancing vs declining right now,
across the full cached universe -- a broad market-health signal that sits
ABOVE the Sectors tab's per-sector breakdown. Same "informational, not a
trading signal" principle as the rest of that tab: this doesn't gate or
generate trades, it's context for a human reading the dashboard.

Reuses detectors/sector_momentum.py's pick_reference_as_of() and
compute_stock_return_today() rather than reimplementing "% return since
today's open" -- the same reasoning applies here as there: every symbol's
return must be measured over the SAME reference session, or the breadth
count silently mixes different trading days (see pick_reference_as_of()'s
docstring for the confirmed real-world case this guards against -- 5 of
180 real symbols one day ahead of the rest due to bulk-fetch timing drift).

New highs/lows: deliberately NOT included. A meaningful reading needs a
real lookback window (52-week is standard; even a 20/50-day version needs
real historical depth), and the cached universe currently spans only
Upstox's own ~30-day rolling window plus 1 archived day (see examples/
daily_archive.py) -- nowhere near enough for a 52-week reading, and a
shorter window would be an arbitrary substitute dressed up as something
recognizable. Revisit once the archive has accumulated real depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from detectors.sector_momentum import compute_stock_return_today, pick_reference_as_of


@dataclass
class BreadthReading:
    as_of: pd.Timestamp
    advancing: int
    declining: int
    flat: int
    total: int
    pct_advancing: float
    pct_declining: float
    pct_flat: float
    interpretation: str  # "broadly bullish" | "broadly bearish" | "mixed/neutral"


# A simple, honest heuristic -- NOT statistically derived (that would need
# a real historical distribution of breadth readings to calibrate
# against, which this project doesn't have yet). >=65% advancing counts
# as bullish, <=35% as bearish, the 35-65% band in between as neutral.
BULLISH_THRESHOLD_PCT = 65.0
BEARISH_THRESHOLD_PCT = 35.0


def _interpret(pct_advancing: float) -> str:
    if pct_advancing >= BULLISH_THRESHOLD_PCT:
        return "broadly bullish"
    if pct_advancing <= BEARISH_THRESHOLD_PCT:
        return "broadly bearish"
    return "mixed/neutral"


def compute_market_breadth(
    symbol_data: dict[str, pd.DataFrame], as_of: pd.Timestamp | None = None,
) -> BreadthReading | None:
    """
    symbol_data: {symbol: full_1min_dataframe} -- typically the FULL
        cached universe (not just sector-mapped symbols; breadth is a
        whole-market reading and doesn't need sector classification).
        Advancing/declining is measured from each symbol's own today's-
        open (see compute_stock_return_today()), the same "since market
        open" reference sector_momentum.py uses by default.
    as_of: None (default) picks the single shared reference timestamp via
        pick_reference_as_of() -- same reasoning as
        compute_sector_momentum(): every symbol must be measured over the
        SAME session, not each symbol's own latest (possibly
        different-day) bar, or the advance/decline count would silently
        mix different trading days together.
    Returns None if no symbol has usable data as of the reference time.
    """
    if as_of is None:
        as_of = pick_reference_as_of(symbol_data)
        if as_of is None:
            return None

    advancing = declining = flat = 0
    for df in symbol_data.values():
        if df.empty:
            continue
        ret = compute_stock_return_today(df, as_of)
        if ret is None:
            continue
        if ret > 0:
            advancing += 1
        elif ret < 0:
            declining += 1
        else:
            flat += 1

    total = advancing + declining + flat
    if total == 0:
        return None

    pct_advancing = advancing / total * 100.0
    pct_declining = declining / total * 100.0
    pct_flat = flat / total * 100.0

    return BreadthReading(
        as_of=as_of, advancing=advancing, declining=declining, flat=flat, total=total,
        pct_advancing=pct_advancing, pct_declining=pct_declining, pct_flat=pct_flat,
        interpretation=_interpret(pct_advancing),
    )
