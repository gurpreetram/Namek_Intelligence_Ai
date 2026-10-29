"""
"Top Movers Today" for the Daily Picks (F4) dashboard -- stocks with the
biggest % move from today's open, REGARDLESS of any strategy signal.
Informational only (see _TOP_MOVERS_DISCLAIMER in api/main.py, kept
visible at all times by the frontend) -- same convention as detectors/
sector_momentum.py and detectors/strong_trend.py: this module has no
opinion about entries/targets/stops, it just ranks and displays.

Reuses detectors/sector_momentum.py's compute_stock_return_today() for
the % move (same "since market open" definition every other move-from-
open check in this project uses, not a second, potentially-diverging
one) and detectors/breakout_validator.py's volume_zscore() for the
volume context (same z-score convention detectors/pick_verification.py's
check_volume_confirmation() already uses).

classify_volume_zscore() is public (promoted from module-private on
2026-08-17 when detectors/best_picks.py became a second real caller,
needing the exact same high/normal/low bucketing for its own volume
badge) -- keep its signature stable for that caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from detectors.breakout_validator import volume_zscore
from detectors.sector_momentum import compute_stock_return_today, pick_reference_as_of

TOP_MOVERS_COUNT = 10
# Same lookback window detectors/pick_verification.py's VOLUME_LOOKBACK_BARS
# uses for "is this bar's volume meaningfully elevated" -- reused, not a
# second threshold invented here.
VOLUME_LOOKBACK_BARS = 20


@dataclass
class TopMover:
    symbol: str
    move_pct_from_open: float
    current_price: float
    # None when there aren't yet VOLUME_LOOKBACK_BARS+1 bars today to judge
    # a z-score from (early in the session) -- not the same as a real 0.0.
    volume_zscore: Optional[float]
    volume_label: str  # "high" | "normal" | "low" | "unknown"


def classify_volume_zscore(z: Optional[float]) -> str:
    """Coarse, display-only bucketing -- not a gate, not backtested (this
    whole module is informational, see its own docstring). >=1.0 matches
    detectors/pick_verification.py's MIN_VOLUME_ZSCORE ("meaningfully
    elevated"); <=-0.5 is a reasonable, symmetric-ish "notably quiet" cut,
    not a value derived from any test."""
    if z is None:
        return "unknown"
    if z >= 1.0:
        return "high"
    if z <= -0.5:
        return "low"
    return "normal"


def _compute_volume_zscore(day_df: pd.DataFrame, lookback_bars: int = VOLUME_LOOKBACK_BARS) -> Optional[float]:
    if len(day_df) < lookback_bars + 2:
        return None
    current_volume = float(day_df["volume"].iloc[-1])
    recent_volumes = day_df["volume"].iloc[-(lookback_bars + 1):-1]
    return volume_zscore(current_volume, recent_volumes)


def find_top_movers(
    symbol_data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp | None = None,
    top_n: int = TOP_MOVERS_COUNT,
) -> tuple[list[TopMover], list[TopMover]]:
    """
    Returns (gainers, losers), each sorted strongest-move-first and capped
    at top_n. symbol_data should already be price-filtered by the caller
    (see api/main.py's get_top_movers(), which applies detectors.scanner.
    MIN_STOCK_PRICE before calling this, same floor the live scanner/
    Daily Picks/Strong Trend use) -- this function itself has no opinion
    on liquidity, it only ranks whatever it's given.

    as_of=None picks the same shared reference timestamp compute_sector_
    momentum() would (see pick_reference_as_of) -- pass it explicitly if
    the caller already computed it, so the reported as_of always matches
    what was actually used.
    """
    if as_of is None:
        as_of = pick_reference_as_of(symbol_data)
        if as_of is None:
            return [], []

    movers: list[TopMover] = []
    for symbol, df in symbol_data.items():
        truncated = df.loc[:as_of]
        today_bars = truncated[truncated.index.date == as_of.date()]
        if today_bars.empty:
            continue
        ret = compute_stock_return_today(df, as_of)
        if ret is None:
            continue
        current_price = float(today_bars["close"].iloc[-1])
        z = _compute_volume_zscore(today_bars)
        movers.append(TopMover(
            symbol=symbol, move_pct_from_open=ret, current_price=current_price,
            volume_zscore=z, volume_label=classify_volume_zscore(z),
        ))

    gainers = sorted(
        (m for m in movers if m.move_pct_from_open > 0), key=lambda m: m.move_pct_from_open, reverse=True,
    )[:top_n]
    losers = sorted(
        (m for m in movers if m.move_pct_from_open < 0), key=lambda m: m.move_pct_from_open,
    )[:top_n]
    return gainers, losers
