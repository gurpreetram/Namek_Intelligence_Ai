"""
Resolves NSE trading symbols (e.g. "RELIANCE") to Upstox instrument_keys
(e.g. "NSE_EQ|INE002A01018"), which the candle APIs require.

Upstox publishes a daily instrument master file (JSON, gzipped) covering
every tradable instrument across all segments. We download and cache it
locally, refreshing once per day, rather than hardcoding instrument keys
(which is fragile -- ISINs don't change but Upstox's file is the source
of truth and the only sane way to bulk-resolve a scanner universe later).
"""
from __future__ import annotations

import gzip
import io
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config.settings import PROJECT_ROOT

INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
CACHE_PATH = PROJECT_ROOT / "config" / ".instrument_master.parquet"
CACHE_MAX_AGE_SECONDS = 24 * 3600


def _download_master() -> pd.DataFrame:
    resp = requests.get(INSTRUMENT_MASTER_URL, timeout=30)
    resp.raise_for_status()
    raw = gzip.decompress(resp.content)
    records = json.loads(raw)
    df = pd.DataFrame(records)
    # Keep only equity segment -- F&O, currency, etc. are out of scope per
    # the project's own stated constraint (intraday cash equity, not F&O).
    # segment == "NSE_EQ" alone is NOT enough: Upstox files State Development
    # Loans, GOI strips/securities, and SME-board listings under the same
    # NSE_EQ segment (instrument_type "SG", "GS", "SM", etc.), so without
    # this second filter list_all_nse_equity_symbols() returns thousands of
    # bonds/SDLs alongside real stocks. instrument_type == "EQ" is what
    # Upstox actually uses to mean "common equity share".
    df = df[(df["segment"] == "NSE_EQ") & (df["instrument_type"] == "EQ")].copy()
    df.to_parquet(CACHE_PATH, index=False)
    return df


def load_instrument_master(force_refresh: bool = False) -> pd.DataFrame:
    if not force_refresh and CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            return pd.read_parquet(CACHE_PATH)
    return _download_master()


def resolve_instrument_key(trading_symbol: str) -> Optional[str]:
    """
    e.g. resolve_instrument_key("RELIANCE") -> "NSE_EQ|INE002A01018"
    Returns None if the symbol isn't found (typo, delisted, or not an
    equity instrument).
    """
    df = load_instrument_master()
    match = df[df["trading_symbol"] == trading_symbol.upper()]
    if match.empty:
        return None
    return str(match.iloc[0]["instrument_key"])


def resolve_many(trading_symbols: list[str]) -> dict[str, Optional[str]]:
    df = load_instrument_master()
    lookup = dict(zip(df["trading_symbol"], df["instrument_key"]))
    return {sym: lookup.get(sym.upper()) for sym in trading_symbols}


def list_all_nse_equity_symbols() -> list[str]:
    """Returns all NSE_EQ trading symbols from the instrument master file."""
    df = load_instrument_master()
    return sorted(df["trading_symbol"].unique().tolist())
