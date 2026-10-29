"""
Local-only partial exit tracker for Daily Picks (F4) -- records a
partial exit the user already placed manually (e.g. in Kotak Neo)
against an open position, so the dashboard can show "how much is left,
at what stop, and what's it worth right now." This module does NOT place,
modify, or cancel any real order -- it is pure bookkeeping over what the
user reports happened, same "local observation only" boundary as
paper_trading/journal.py's paper trades.

Persistence: a single JSON array file (atomic tmp-write + os.replace,
same pattern as paper_trading/live_poller.py's latest_prices.json) rather
than journal.py's append-only JSONL -- partial exits need row-level
pruning (see get_active_partial_exits()'s daily auto-clear), which a
plain JSON array supports directly; an ever-growing event log would not.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from config.settings import PROJECT_ROOT

PARTIAL_EXITS_PATH = PROJECT_ROOT / "paper_trading" / "partial_exits.json"


@dataclass
class PartialExit:
    id: str
    symbol: str
    direction: str  # "long" | "short"
    total_quantity: int
    exit_quantity: int
    exit_price: float
    remaining_quantity: int
    new_stop_loss: float
    created_at: str  # ISO datetime -- basis for the daily auto-clear below


def _load_all() -> list[PartialExit]:
    if not PARTIAL_EXITS_PATH.exists():
        return []
    try:
        raw = json.loads(PARTIAL_EXITS_PATH.read_text())
    except (OSError, ValueError):
        return []
    return [PartialExit(**r) for r in raw]


def _write_all(records: list[PartialExit]) -> None:
    PARTIAL_EXITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PARTIAL_EXITS_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps([asdict(r) for r in records], indent=2))
    os.replace(tmp_path, PARTIAL_EXITS_PATH)  # atomic rename, avoids a torn read mid-write


def add_partial_exit(
    symbol: str, direction: str, total_quantity: int, exit_quantity: int,
    exit_price: float, remaining_quantity: int, new_stop_loss: float,
) -> PartialExit:
    record = PartialExit(
        id=uuid.uuid4().hex, symbol=symbol, direction=direction,
        total_quantity=total_quantity, exit_quantity=exit_quantity, exit_price=exit_price,
        remaining_quantity=remaining_quantity, new_stop_loss=new_stop_loss,
        created_at=datetime.now().isoformat(),
    )
    records = _load_all()
    records.append(record)
    _write_all(records)
    return record


def get_active_partial_exits() -> list[PartialExit]:
    """
    Today's partial exits only. Entries from a previous calendar day are
    dropped and the file is rewritten WITHOUT them (not just filtered on
    read) -- the "auto-clear" the API's GET handler relies on, so the
    file doesn't grow forever across trading days.
    """
    records = _load_all()
    today = date.today()
    active = [r for r in records if datetime.fromisoformat(r.created_at).date() == today]
    if len(active) != len(records):
        _write_all(active)
    return active
