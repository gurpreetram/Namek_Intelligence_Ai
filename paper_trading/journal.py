"""
Persistent, append-only trade journal -- every signal the system
generates, every trade it actually enters, every exit, and every signal
that got voided before fill (slippage guard) gets logged here as JSON
lines. This is your record of "what would this system have done," fully
reviewable before you ever risk real money on it.

Format: JSON Lines (one JSON object per line) in a local file -- simple,
append-friendly, human-readable if you just open the file, and trivially
loadable into pandas for analysis later.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import Signal
from config.settings import PROJECT_ROOT

DEFAULT_JOURNAL_PATH = PROJECT_ROOT / "paper_trading" / "journal.jsonl"


class TradeJournal:
    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_JOURNAL_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        record["logged_at"] = datetime.now().isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_signal(self, symbol: str, signal: Signal, at_time: pd.Timestamp) -> None:
        self._write({
            "event": "signal_generated",
            "symbol": symbol,
            "direction": signal.direction.value,
            "confidence": signal.confidence,
            "stop_loss": signal.stop_loss,
            "target": signal.target,
            "reasoning": signal.reasoning,
            "signal_time": at_time,
        })

    def log_voided_signal(self, symbol: str, signal: Signal,
                           at_time: pd.Timestamp, attempted_fill_price: float) -> None:
        """A signal that got voided at fill time because slippage between
        signal generation and the next bar's open invalidated the stop/
        target relationship -- see engine.py's slippage guard. Logging
        this is important: if you see many voided signals in the journal,
        that's a sign the strategy's stops are too tight for how fast this
        symbol actually moves, not a system bug to ignore."""
        self._write({
            "event": "signal_voided_by_slippage",
            "symbol": symbol,
            "direction": signal.direction.value,
            "attempted_fill_price": attempted_fill_price,
            "stop_loss": signal.stop_loss,
            "target": signal.target,
            "voided_at": at_time,
        })

    def log_entry(self, trade: Trade) -> None:
        self._write({
            "event": "trade_entered",
            "symbol": trade.symbol,
            "strategy_name": trade.strategy_name,
            "direction": trade.direction.value,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "stop_loss": trade.stop_loss,
            "target": trade.target,
            "reasoning": trade.reasoning,
        })

    def log_exit(self, trade: Trade) -> None:
        self._write({
            "event": "trade_exited",
            "symbol": trade.symbol,
            "strategy_name": trade.strategy_name,
            "direction": trade.direction.value,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "exit_time": trade.exit_time,
            "exit_price": trade.exit_price,
            "exit_reason": trade.exit_reason,
            "exit_reasoning": trade.exit_reasoning,
            "pnl_points": trade.pnl_points,
            "r_multiple": trade.r_multiple,
        })

    def load_all(self) -> pd.DataFrame:
        """Loads the full journal as a DataFrame for review/analysis."""
        if not self.path.exists():
            return pd.DataFrame()
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def summary(self) -> dict:
        """Quick counts -- how many signals fired, how many became real
        trades, how many voided, current win rate on closed paper trades."""
        df = self.load_all()
        if df.empty:
            return {"signals": 0, "entries": 0, "voided": 0, "exits": 0, "win_rate": None}

        signals = (df["event"] == "signal_generated").sum()
        entries = (df["event"] == "trade_entered").sum()
        voided = (df["event"] == "signal_voided_by_slippage").sum()
        exits_df = df[df["event"] == "trade_exited"]
        win_rate = None
        if not exits_df.empty and "pnl_points" in exits_df.columns:
            win_rate = float((exits_df["pnl_points"] > 0).mean())

        return {
            "signals": int(signals),
            "entries": int(entries),
            "voided": int(voided),
            "exits": int(len(exits_df)),
            "win_rate": win_rate,
        }
