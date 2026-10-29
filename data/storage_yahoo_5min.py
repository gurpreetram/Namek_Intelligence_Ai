"""
RESEARCH-ONLY storage for data/fetcher_yahoo_5min.py's output -- a
DELIBERATELY separate location (storage/yahoo_5min/) from storage.py's
storage/candles/ (the primary 1-minute Upstox cache everything else in
this project depends on). Never confuse the two: this holds 5-MINUTE
bars from a free, unauthenticated, unofficial supplementary source, used
for research that's explicitly fine with coarser granularity (see
examples/research_sector_momentum_rolling_yahoo5min.py) -- never as a
drop-in replacement for storage.load()/load_combined() in any live or
backtest path, since every existing Strategy evaluates 1-minute bars.

Mirrors storage.py's load()/save() interface exactly (same merge/dedupe-
by-timestamp safety) so anyone already familiar with that module needs
no new mental model here -- just a different directory and column
granularity.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import PROJECT_ROOT

YAHOO_5MIN_ROOT = PROJECT_ROOT / "storage" / "yahoo_5min"
YAHOO_5MIN_ROOT.mkdir(parents=True, exist_ok=True)


def _path_for(symbol: str) -> Path:
    return YAHOO_5MIN_ROOT / f"{symbol.upper()}_5min.parquet"


def load(symbol: str) -> pd.DataFrame:
    path = _path_for(symbol)
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.read_parquet(path)


def save(symbol: str, df: pd.DataFrame) -> None:
    existing = load(symbol)
    combined = pd.concat([existing, df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(_path_for(symbol))


def cached_symbols() -> list[str]:
    return sorted(p.stem.replace("_5min", "") for p in YAHOO_5MIN_ROOT.glob("*_5min.parquet"))
