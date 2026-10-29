"""
Builds any higher timeframe (3m, 5m, 15m, ...) from 1-minute base candles.

Doing this instead of fetching each timeframe separately guarantees a 5m
candle is exactly five real 1m candles aggregated -- no risk of the 5m
series and the 1m series disagreeing about where a candle boundary falls,
which would silently corrupt any multi-timeframe confirmation logic
(e.g. "5m trend + 1m entry trigger", which is exactly what your notes
describe wanting).
"""
from __future__ import annotations

import pandas as pd

from config.settings import MARKET_CLOSE, MARKET_OPEN

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample(df_1min: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    timeframe_minutes: e.g. 3, 5, 15. Candle boundaries are anchored to
    market open (09:15 IST), not wall-clock midnight, so a "15m" candle
    on NSE is 09:15-09:30, 09:30-09:45, etc. -- matching how every trading
    platform actually displays 15m candles, not arbitrary UTC-aligned bins.
    """
    if df_1min.empty:
        return df_1min.copy()

    out_frames = []
    for trade_date, day_df in df_1min.groupby(df_1min.index.date):
        day_df = day_df.between_time(MARKET_OPEN, MARKET_CLOSE)
        if day_df.empty:
            continue
        origin = day_df.index[0]
        resampled = (
            day_df.resample(f"{timeframe_minutes}min", origin=origin, label="left", closed="left")
            .agg(_AGG)
            .dropna(subset=["open"])
        )
        out_frames.append(resampled)

    if not out_frames:
        return pd.DataFrame(columns=df_1min.columns)
    return pd.concat(out_frames).sort_index()


def multi_timeframe_view(df_1min: pd.DataFrame, timeframes: list[int]) -> dict[int, pd.DataFrame]:
    """Convenience: build several timeframes at once, e.g. [1, 5, 15]."""
    result = {}
    for tf in timeframes:
        result[tf] = df_1min.copy() if tf == 1 else resample(df_1min, tf)
    return result


class IncrementalResampler:
    """
    Builds a resampled higher-timeframe view bar-by-bar as 1-minute candles
    arrive one at a time, instead of calling resample() (a full pandas
    groupby + resample) from scratch on every new bar.

    Why this exists: a backtest/scan replay evaluates a strategy on every
    bar of a trading day, and each evaluation needs "the higher-timeframe
    view as of right now". Calling resample() fresh each time re-aggregates
    every bar seen so far even though only the CURRENT (still-forming)
    bucket actually changed -- updating one bucket's running OHLCV, or
    starting a new one, is O(1); resample() redoes O(bars-so-far) of work
    to get the same answer. Over a single trading day this is the dominant
    remaining cost once cross-day recomputation is eliminated (see
    backtest/engine.py's per-day history cache).

    Semantics are matched to resample() bar-for-bar: buckets are anchored
    to `day_origin` (the day's first in-market-hours bar), closed="left",
    label="left" -- i.e. a bucket labeled T covers [T, T+timeframe). Only
    bars within MARKET_OPEN..MARKET_CLOSE are counted, same as resample()'s
    between_time() filter, so a stray pre/post-market tick can't pollute a
    bucket here when it wouldn't have polluted resample()'s output either.

    Bars MUST be fed in increasing timestamp order within a single trading
    day -- this class has no cross-day logic, callers reset one instance
    per trade_date (see backtest/engine.py and detectors/scanner.py).
    """

    def __init__(self, timeframe_minutes: int, day_origin: pd.Timestamp):
        self.timeframe_minutes = timeframe_minutes
        self.day_origin = day_origin
        self._market_open = pd.Timestamp(MARKET_OPEN).time()
        self._market_close = pd.Timestamp(MARKET_CLOSE).time()
        self._bucket_starts: list[pd.Timestamp] = []
        self._opens: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []
        self._volumes: list[float] = []
        self._current_bucket_idx: int | None = None

    def update(self, ts: pd.Timestamp, bar: pd.Series) -> None:
        if not (self._market_open <= ts.time() <= self._market_close):
            return  # matches resample()'s between_time(MARKET_OPEN, MARKET_CLOSE) filter

        bucket_idx = int((ts - self.day_origin) / pd.Timedelta(minutes=self.timeframe_minutes))
        if bucket_idx == self._current_bucket_idx:
            self._highs[-1] = max(self._highs[-1], float(bar["high"]))
            self._lows[-1] = min(self._lows[-1], float(bar["low"]))
            self._closes[-1] = float(bar["close"])
            self._volumes[-1] += float(bar["volume"])
        else:
            bucket_start = self.day_origin + pd.Timedelta(minutes=self.timeframe_minutes * bucket_idx)
            self._bucket_starts.append(bucket_start)
            self._opens.append(float(bar["open"]))
            self._highs.append(float(bar["high"]))
            self._lows.append(float(bar["low"]))
            self._closes.append(float(bar["close"]))
            self._volumes.append(float(bar["volume"]))
            self._current_bucket_idx = bucket_idx

    def as_dataframe(self) -> pd.DataFrame:
        if not self._bucket_starts:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.DataFrame(
            {"open": self._opens, "high": self._highs, "low": self._lows,
             "close": self._closes, "volume": self._volumes},
            index=pd.DatetimeIndex(self._bucket_starts),
        )
