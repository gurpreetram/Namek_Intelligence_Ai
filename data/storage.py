"""
Local Parquet cache for 1-minute candles, one file per symbol.

Why Parquet over SQLite/CSV: columnar storage compresses OHLCV time series
well, pandas reads/writes it natively and fast, and for a single-user local
setup we don't need SQLite's concurrent-write guarantees. If this later
grows into a multi-process live system, swap this for a proper time-series
DB (e.g. TimescaleDB) -- the interface below is deliberately narrow so that
swap only touches this one file.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from config.settings import ARCHIVE_ROOT, STORAGE_ROOT

# os.replace() can raise PermissionError ([WinError 5] Access is denied) on
# Windows if a reader currently has the destination file open -- unlike
# POSIX rename(), Windows won't replace a file with an open handle. A
# reader's open() is normally held only for the duration of one read (a
# few ms for these file sizes), so a short bounded retry absorbs that
# window instead of surfacing a spurious failure to the writer. Confirmed
# necessary empirically, not just theoretical: a concurrent read+write
# stress test (6 reader threads vs. 1 writer, both hammering the same
# file) hit WinError 5 on nearly every write without this retry, on this
# same Windows host the live poller/API pair actually run on.
_REPLACE_MAX_RETRIES = 300
_REPLACE_RETRY_DELAY_SECONDS = 0.05

# The flip side of the same Windows quirk: a reader's open() can also
# transiently raise PermissionError ([Errno 13] Permission denied) during
# the brief instant of a concurrent os.replace() rename, even though the
# rename itself is atomic (the reader never sees a torn/partial file --
# confirmed via stress testing: every failure under concurrent load was a
# clean, catchable PermissionError, never a parse/corruption error). The
# window is much shorter than the write-side one above (the rename
# instant itself, not the duration of a full read), so a smaller retry
# budget is enough.
_READ_MAX_RETRIES = 40
_READ_RETRY_DELAY_SECONDS = 0.05


def _read_parquet_with_retry(path: Path) -> pd.DataFrame:
    for attempt in range(_READ_MAX_RETRIES):
        try:
            return pd.read_parquet(path)
        except PermissionError:
            if attempt == _READ_MAX_RETRIES - 1:
                raise
            time.sleep(_READ_RETRY_DELAY_SECONDS)


def _path_for(symbol: str) -> Path:
    return STORAGE_ROOT / f"{symbol.upper()}_1min.parquet"


def _archive_path_for(symbol: str) -> Path:
    return ARCHIVE_ROOT / f"{symbol.upper()}_1min.parquet"


def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Writes to a `.tmp` sibling then os.replace()s it into place -- same
    atomic pattern already used by paper_trading/live_poller.py's
    latest_prices.json and paper_trading/partial_exits.py's
    partial_exits.json. A direct df.to_parquet(path) writes in place, so
    a reader (e.g. api/main.py's storage.load(), called while the live
    poller is mid-write) can land between open and final flush/close and
    get a torn file -- pyarrow then raises "OSError: Unexpected end of
    stream" trying to parse it. Confirmed as a real, not hypothetical,
    failure while validating api/main.py's live-session staleness-gate
    fix against an actively-running live session. os.replace() is a
    single filesystem rename, so a reader always sees either the
    complete old file or the complete new one, never a partial write --
    and the ".tmp" suffix (not ".parquet") keeps it out of every
    `*_1min.parquet` glob used for universe discovery, so an in-progress
    write is never mistaken for a real cached symbol either.

    The retry loop around os.replace() itself (see _REPLACE_MAX_RETRIES)
    is a separate, Windows-specific concern from the atomicity above --
    see that constant's comment.
    """
    tmp_path = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path)
    for attempt in range(_REPLACE_MAX_RETRIES):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == _REPLACE_MAX_RETRIES - 1:
                raise
            time.sleep(_REPLACE_RETRY_DELAY_SECONDS)


def load(symbol: str) -> pd.DataFrame:
    path = _path_for(symbol)
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return _read_parquet_with_retry(path)


def save(symbol: str, df: pd.DataFrame) -> None:
    """
    Merges new candles with whatever's cached, de-duplicates by timestamp
    index, and writes back atomically (see _atomic_to_parquet()). Safe to
    call repeatedly with overlapping ranges (e.g. re-fetching today's
    data every few minutes intraday) AND safe for a concurrent reader
    (e.g. the API layer) to hit at any moment -- it always sees a
    complete file, old or new, never a torn one.
    """
    existing = load(symbol)
    combined = pd.concat([existing, df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    _atomic_to_parquet(combined, _path_for(symbol))


def load_archive(symbol: str) -> pd.DataFrame:
    """Reads from the permanent archive (ARCHIVE_ROOT) only -- never
    pruned, distinct from load()'s ~30-day rolling STORAGE_ROOT cache.
    Retries on a transient PermissionError same as load() -- see
    _read_parquet_with_retry()'s docstring -- now that save_archive()
    below writes atomically too, the same rename-instant read race
    applies here."""
    path = _archive_path_for(symbol)
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return _read_parquet_with_retry(path)


def save_archive(symbol: str, df: pd.DataFrame) -> None:
    """
    Same merge/dedupe-by-timestamp-index safety as save(), just against
    the permanent archive instead of the rolling cache -- see that
    function's docstring. Used by examples/daily_archive.py; safe to call
    repeatedly (e.g. re-running the archive script, or re-archiving an
    already-archived day) without creating duplicates.

    Writes atomically (see _atomic_to_parquet()), same as save() -- a
    direct to_parquet() here previously left a real corrupted archive
    file (VINDHYATEL_1min.parquet) when a Task Scheduler run crashed
    mid-write (STATUS_CONTROL_C_EXIT); the whole symbol's history was
    lost, not just that day's bars, since to_parquet() rewrites the
    complete file rather than appending. Atomic replace means a crash at
    any point now leaves either the untouched prior file or the complete
    new one -- never a truncated one.
    """
    existing = load_archive(symbol)
    combined = pd.concat([existing, df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    _atomic_to_parquet(combined, _archive_path_for(symbol))


def load_combined(symbol: str) -> pd.DataFrame:
    """
    Archive + current rolling cache merged and de-duplicated by timestamp
    -- the full accumulated history this project has ever seen for a
    symbol, spanning further back than Upstox's own ~30-day intraday
    window once daily_archive.py has been run for a while. load() itself
    is left unchanged (rolling-cache-only) for backward compatibility --
    existing callers that only need recent history don't need to change.
    """
    archive = load_archive(symbol)
    current = load(symbol)
    combined = pd.concat([archive, current])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def coverage(symbol: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Returns (earliest, latest) cached timestamp for a symbol, or None."""
    df = load(symbol)
    if df.empty:
        return None
    return df.index.min(), df.index.max()
