"""
Paper Trading Engine: runs a Strategy against data ONE BAR AT A TIME, the
same way it would encounter live/incoming data -- as opposed to
BacktestEngine, which processes a whole historical date range in one call.

Why this is a separate engine and not just "call BacktestEngine every
minute": the whole point of paper trading is to prove the system behaves
correctly when fed data incrementally, the way it actually will once
connected to a live feed. A backtest processing a full day in one pass
can hide subtle state-management bugs that only show up when bars arrive
one at a time in real time -- this engine is deliberately built to be fed
that way, bar by bar, so it's a true dry run of the live code path, not
just the same historical logic run again.

Every signal AND every trade outcome is persisted to a local JSON journal
(paper_trading/journal.py) -- this is also your Trade Journal: a running
record of what the system would have done, with full reasoning, that you
can review before ever risking real capital on it.

Reuses the exact same fill/exit mechanics as BacktestEngine (next-bar-open
fill, high/low stop checks, slippage-invalidation guard) so paper trading
and backtesting are governed by identical rules -- if they diverged, a
strategy that looked good in backtest could behave differently in paper
trading for no real reason, which would make the whole exercise pointless.

Liquidity gate + gap-risk guard (added alongside the live-monitoring/
watchlist system, found while validating Phase B against a direct
backtest -- see examples/run_replay_validation.py): this engine was
missing BOTH the illiquid-stock rejection gate and the fill-gap/slippage-
magnitude guard that backtest/engine.py's run()/run_multi() already had
(the fixes that came from the real BANG +180R and AFFLE +46R cases
documented there). Without them, PaperTradingEngine could journal --
and NOTIFY the user about -- exactly the same class of fake trade: a
thin-liquidity print faking an absurd R-multiple, or a fill gap
collapsing realized risk to a sliver of what the strategy intended.
Suspected (not yet confirmed at the time this fix was written) to be the
cause of a real replay-validation mismatch: ULTRACEMCO/validated_breakout
produced an extra trade in replay that the backtest comparison (which
already had both guards) did not -- re-running replay validation after
this fix should confirm or rule this out. Both guards now mirror
backtest/engine.py exactly either way, since the gap is real regardless
of whether it explains this specific mismatch.

Incremental resampling (added alongside the live-monitoring/watchlist
system): on_new_bar() used to call multi_timeframe_view() on the FULL
truncated history on every single call -- the exact same O(bars^2)-prone
antipattern already fixed in backtest/engine.py's run()/run_multi() via
IncrementalResampler + per-day history_mtf caching (see that module's
docstring). It matters even more here: on_new_bar() is meant to be called
once per real-time bar, all day, across every watchlist symbol, for the
entire market session -- not once per backtest run. Completed days are
resampled ONCE (cached in _history_mtf), and the current day's higher-
timeframe buckets are updated in O(1) per bar via IncrementalResampler
(_resamplers), exactly mirroring backtest/engine.py's approach, just
triggered by day-rollover detection (_current_day) instead of an explicit
groupby loop, since on_new_bar() is driven by the caller's live feed one
bar at a time rather than a bounded historical DataFrame processed in one
call.

EARLY EXIT WARNING SYSTEM (_check_early_exit(), _check_momentum_reversal()):
before this, an open position only ever exited at target, stop_loss, or
EOD square-off -- no monitoring of whether the setup that justified entry
was still healthy in between. This adds two checks, run on every bar an
open position exists, BEFORE the hard stop-loss/target check:
  1. Setup invalidation (per-strategy, via an OPTIONAL check_invalidation()
     method on the Strategy -- called via getattr() duck-typing, NOT part
     of the Strategy ABC, so strategies without one just skip this path).
     Each strategy's own docstring explains its specific condition.
  2. Momentum reversal (generic, every strategy): the position moved
     favorably by a meaningful amount, then the last few bars reversed
     back with volume elevated in the adverse direction.
Either firing exits with a distinct exit_reason ("early_warning_setup_
invalidated" / "early_warning_momentum_reversal") and a specific, human-
readable exit_reasoning list, distinct from stop_loss/target/eod_square_off.
This REQUIRED building mtf_view (multi-timeframe data) even while a
position is open -- previously only built when flat, evaluating a new
signal -- a real, accepted per-bar cost, not a free addition.
VALIDATED via historical replay before going live -- see examples/
validate_early_exit_replay.py for the false-positive-rate methodology and
results; do not trust this system's thresholds without reading that first.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from backtest.engine import MIN_REALIZED_RISK_FRACTION, Trade
from backtest.strategy_base import Signal, SignalDirection, Strategy
from data.resample import IncrementalResampler, multi_timeframe_view
from detectors.breakout_validator import volume_zscore
from detectors.liquidity_analysis import analyze_liquidity
from detectors.pick_verification import (
    VERIFIED_STRATEGIES,
    check_continuation,
    check_minimum_move,
    check_volume_confirmation,
)
from detectors.support_resistance import average_true_range
from notifications.notification_engine import Alert, NotificationEngine
from paper_trading.journal import TradeJournal

# Early-exit warning thresholds (see _check_momentum_reversal()). ATR-
# relative, not a fixed point value -- this project's universe spans
# stocks from ~Rs 100 to Rs 30,000+, where a fixed "+3 to +6 points"
# threshold would mean wildly different things (huge for a Rs 100 stock,
# noise for a Rs 20,000 one). 1x ATR of favorable movement is a
# comparable bar across that whole range, using the same ATR-relative
# convention every stop-loss/target in this project already uses.
EARLY_EXIT_MOMENTUM_MIN_PROFIT_ATR_MULT = 1.0
EARLY_EXIT_MOMENTUM_LOOKBACK_BARS = 3
# Same volume-elevation bar detectors/breakout_validator.py already uses
# for "meaningfully above recent average, not just noise" -- reused
# rather than picking a second, potentially-inconsistent threshold.
EARLY_EXIT_MOMENTUM_VOLUME_ZSCORE_THRESHOLD = 0.5


class PaperTradingEngine:
    def __init__(
        self,
        strategy: Strategy,
        timeframes: list[int],
        journal: TradeJournal | None = None,
        eod_square_off_time: str = "15:15",
        notifier: NotificationEngine | None = None,
        live_mode: bool = False,
    ):
        self.strategy = strategy
        self.timeframes = sorted(set(timeframes) | {1})
        self.higher_timeframes = [tf for tf in self.timeframes if tf != 1]
        self.eod_square_off_time = eod_square_off_time
        self.journal = journal or TradeJournal()
        # Optional -- when set, every confirmed entry/exit is pushed
        # through this same NotificationEngine used elsewhere (Telegram/
        # console/file channels), right alongside the existing journal
        # write, instead of a separate alerting path (see paper_trading/
        # live_poller.py for how the live watchlist poller wires this up).
        self.notifier = notifier
        # False (default): unchanged original behavior -- a fired Signal
        # goes straight to _pending_signals with its own real target,
        # same as every caller before this existed (e.g. examples/
        # validate_early_exit_replay.py, examples/run_replay_validation.py).
        # True: paper_trading/live_poller.py's actual live trading path --
        # applies CHANGE 2 (pre-entry momentum/volume/no-pullback
        # verification, see detectors/pick_verification.py) for strategies
        # in VERIFIED_STRATEGIES, via the _verifying wait-state below. A
        # signal that passes verification (or belongs to a non-verified
        # strategy) still gets its own real Signal.target, unchanged --
        # CHANGE 1 (fixed-point target) was also applied here before being
        # reverted 2026-07-22 (see detectors/scanner.py's
        # FIXED_TARGET_POINTS docstring for why: a 400-symbol backtest
        # showed it hurt 3 of 4 verified strategies).
        self.live_mode = live_mode

        # Per-symbol trade state, persists across on_new_bar calls -- this
        # is what makes bar-by-bar operation possible (BacktestEngine keeps
        # this state as local variables inside one run() call; here it
        # has to survive between separate calls).
        self._open_trades: dict[str, Trade] = {}
        self._pending_signals: dict[str, object] = {}
        # live_mode + VERIFIED_STRATEGIES only: a signal that fired and
        # passed the INSTANT checks (min move, volume) but is still
        # waiting on check_continuation()'s multi-bar "no immediate
        # pullback" verdict -- re-checked on every subsequent real bar
        # until it resolves to passed/failed (see on_new_bar() step 3).
        # Unlike detectors/scanner.py's fast-scan path (which has to
        # INFER when the trigger happened, no cross-call memory), this is
        # the genuinely rigorous version: the signal was captured at the
        # exact bar it fired, so there's no approximation here.
        self._verifying: dict[str, dict] = {}

        # Per-symbol incremental resampling state -- see module docstring.
        self._current_day: dict[str, date] = {}  # symbol -> date currently being accumulated
        self._history_mtf: dict[str, dict[int, pd.DataFrame]] = {}  # symbol -> {tf: completed-days-only frame}
        self._resamplers: dict[str, dict[int, IncrementalResampler]] = {}  # symbol -> {tf: in-progress resampler}

    def on_new_bar(self, symbol: str, full_1min_history: pd.DataFrame,
                   current_index: pd.Timestamp) -> None:
        """
        Call this once per new completed bar for a symbol -- e.g. every
        time your data feed delivers a new 1-minute candle. full_1min_history
        should include all bars up to and including current_index (the
        engine does not fetch data itself; that's the caller's job, same
        separation as BacktestEngine).
        """
        if current_index not in full_1min_history.index:
            raise ValueError(f"current_index {current_index} not found in provided history")

        current_bar = full_1min_history.loc[current_index]
        current_date = current_index.date()

        # Bootstrap or day-rollover: (re)build the completed-days cache
        # ONCE per day (not per bar) and start fresh incremental
        # resamplers for the new day, exactly like backtest/engine.py's
        # per-day loop -- just triggered by detecting the date changed
        # since the last call for this symbol, since there's no bounded
        # groupby() to drive it here.
        if self._current_day.get(symbol) != current_date:
            truncated = full_1min_history.loc[:current_index]
            before_today = truncated[truncated.index.date != current_date]
            if before_today.empty:
                self._history_mtf[symbol] = {tf: truncated.iloc[0:0] for tf in self.timeframes}
            else:
                self._history_mtf[symbol] = multi_timeframe_view(before_today, self.timeframes)

            today_so_far = truncated[truncated.index.date == current_date]
            day_origin = today_so_far.index[0]
            self._resamplers[symbol] = {
                tf: IncrementalResampler(tf, day_origin) for tf in self.higher_timeframes
            }
            # Feed every bar of today seen so far, not just this one --
            # covers the case where full_1min_history already contains
            # several of today's bars the first time this symbol is seen
            # this day (e.g. the poller starting mid-morning, or catching
            # up after a gap), not just the single newest bar. This
            # already includes current_bar (today_so_far's slice is
            # inclusive of current_index), so no separate update below.
            for ts, bar in today_so_far.iterrows():
                for tf in self.higher_timeframes:
                    self._resamplers[symbol][tf].update(ts, bar)
            self._current_day[symbol] = current_date
        else:
            for tf in self.higher_timeframes:
                self._resamplers[symbol][tf].update(current_index, current_bar)

        # 1. Fill any pending signal from the previous bar.
        pending = self._pending_signals.get(symbol)
        if pending is not None and symbol not in self._open_trades:
            fill_price = float(current_bar["open"])
            valid_long = (
                pending.direction == SignalDirection.LONG
                and pending.stop_loss < fill_price < pending.target
            )
            valid_short = (
                pending.direction == SignalDirection.SHORT
                and pending.target < fill_price < pending.stop_loss
            )

            # Gap-risk guard -- see backtest/engine.py's run()/run_multi()
            # for the full derivation and the real AFFLE +46R case that
            # exposed this: direction sanity alone doesn't catch an
            # adverse gap that collapses realized risk (fill to stop) to a
            # sliver of the strategy's intended risk while still landing
            # technically inside [stop, target]. RR read per-strategy
            # (default 2.0) rather than hardcoded, same as backtest/engine.py.
            rr = getattr(self.strategy, "risk_reward_multiple", 2.0)
            intended_risk = abs(pending.target - pending.stop_loss) / (1.0 + rr)
            realized_risk = abs(fill_price - pending.stop_loss)
            risk_intact = intended_risk <= 0 or realized_risk >= MIN_REALIZED_RISK_FRACTION * intended_risk

            if (valid_long or valid_short) and risk_intact:
                trade = Trade(
                    symbol=symbol,
                    direction=pending.direction,
                    entry_time=current_index,
                    entry_price=fill_price,
                    stop_loss=pending.stop_loss,
                    target=pending.target,
                    reasoning=pending.reasoning,
                    confidence=pending.confidence,
                    strategy_name=self.strategy.name,
                )
                self._open_trades[symbol] = trade
                self.journal.log_entry(trade)
                if self.notifier is not None:
                    self.notifier.notify(self._entry_alert(trade))
            else:
                self.journal.log_voided_signal(symbol, pending, current_index, fill_price)
            del self._pending_signals[symbol]

        # Build the multi-timeframe view unconditionally -- needed both for
        # managing an open position's early-exit checks (step 2) and for
        # evaluating a new signal (step 3), unlike before this feature
        # existed, when it was only built in step 3 (guarded by "symbol is
        # flat"). Cost note: this now runs every bar even while a position
        # is open, not just when flat -- pd.concat() isn't free, and this
        # is a real, accepted tradeoff for live setup-health monitoring,
        # not a free change (see module docstring's early-exit section).
        history_mtf = self._history_mtf[symbol]
        today_1min = full_1min_history.loc[:current_index]
        today_1min = today_1min[today_1min.index.date == current_date]
        mtf_view: dict[int, pd.DataFrame] = {
            1: pd.concat([history_mtf[1], today_1min]) if not history_mtf[1].empty else today_1min,
        }
        for tf in self.higher_timeframes:
            today_resampled = self._resamplers[symbol][tf].as_dataframe()
            mtf_view[tf] = (
                pd.concat([history_mtf[tf], today_resampled]) if not history_mtf[tf].empty
                else today_resampled
            )

        # 2. Manage an open position. Early-exit checks run FIRST, before
        # the hard stop-loss/target -- see _check_early_exit()'s docstring
        # for why (this is what lets an early exit fire before price ever
        # reaches the stop, not just resolve to the same outcome).
        open_trade = self._open_trades.get(symbol)
        if open_trade is not None:
            self._check_early_exit(open_trade, mtf_view, current_index)
            if open_trade.is_open:
                self._check_exit(open_trade, current_bar, current_index)
            if not open_trade.is_open:
                self.journal.log_exit(open_trade)
                if self.notifier is not None:
                    self.notifier.notify(self._exit_alert(open_trade))
                del self._open_trades[symbol]
                open_trade = None

        if open_trade is not None and current_index.strftime("%H:%M") >= self.eod_square_off_time:
            self.force_close(symbol, float(current_bar["close"]), "eod_square_off", current_index)
            return

        # 3. Ask the strategy for a new signal, same no-lookahead guarantee
        #    as BacktestEngine: history (cached at day-rollover above) plus
        #    today's bars-so-far (incrementally resampled above), instead
        #    of re-resampling the full truncated history from scratch.
        if symbol not in self._open_trades and symbol not in self._pending_signals:
            # 3a. CHANGE 2 (live_mode + VERIFIED_STRATEGIES only): continue
            # an in-progress "no immediate pullback" wait for a signal that
            # already fired and passed its instant move/volume checks.
            # Re-runs the SAME check_continuation() detectors/scanner.py's
            # fast-scan path uses, just with a real trigger_time (no
            # inference needed) and real new bars arriving one at a time.
            if symbol in self._verifying:
                state = self._verifying[symbol]
                status, reason = check_continuation(
                    today_1min, state["trigger_time"], state["signal"].trigger_level, state["direction"],
                )
                if status == "pending":
                    return
                del self._verifying[symbol]
                if status == "failed":
                    self.journal.log_voided_signal(symbol, state["signal"], current_index, float(current_bar["close"]))
                    return
                # "passed" -- promote to a real pending signal with its
                # ORIGINAL Signal.target (real 2x-risk-reward) untouched.
                # CHANGE 1 (fixed-point target) applied here too before
                # being reverted 2026-07-22 -- see detectors/scanner.py's
                # FIXED_TARGET_POINTS docstring for why.
                self._pending_signals[symbol] = state["signal"]
                self.journal.log_signal(symbol, state["signal"], current_index)
                return

            # Liquidity gate -- see backtest/engine.py's run()/run_multi()
            # for the full explanation and the real BANG +180R case that
            # exposed this: a strategy is only asked for a signal when the
            # last ~30 bars (no lookahead) look liquid enough to actually
            # trade. None means "not enough bars yet to judge" -- don't
            # block on that, only block on an actual illiquid reading.
            liquidity = analyze_liquidity(mtf_view[1])
            if liquidity is not None and not liquidity.is_liquid:
                return

            signal = self.strategy.evaluate(symbol, mtf_view, current_index)
            if signal is None or signal.direction == SignalDirection.NONE:
                return

            # live_mode: CHANGE 2 (the multi-bar no-pullback wait) applies
            # only to VERIFIED_STRATEGIES; every other strategy (and every
            # live_mode=False caller) goes straight to a pending signal
            # with its real Signal.target, exactly as before CHANGE 1
            # existed. reference_price (signal-time bar's close) is only
            # needed for the instant move/volume checks below now.
            if self.live_mode and self.strategy.name in VERIFIED_STRATEGIES:
                reference_price = float(current_bar["close"])
                move_ok, move_reason = check_minimum_move(today_1min, reference_price, signal.direction)
                vol_ok, vol_reason = check_volume_confirmation(today_1min)
                if not (move_ok and vol_ok):
                    self.journal.log_voided_signal(symbol, signal, current_index, reference_price)
                    return
                self._verifying[symbol] = {
                    "signal": signal, "trigger_time": current_index,
                    "direction": signal.direction, "reference_price": reference_price,
                }
                return

            self._pending_signals[symbol] = signal
            self.journal.log_signal(symbol, signal, current_index)

    @staticmethod
    def _entry_alert(trade: Trade) -> Alert:
        return Alert(
            title=f"ENTRY: {trade.symbol} {trade.direction.value.upper()} ({trade.strategy_name})",
            message=f"Entry {trade.entry_price:.2f} | Stop {trade.stop_loss:.2f} | Target {trade.target:.2f}",
            reasoning=trade.reasoning,
            urgency="high",
        )

    @staticmethod
    def _exit_alert(trade: Trade) -> Alert:
        early_exit = trade.exit_reason in ("early_warning_momentum_reversal", "early_warning_setup_invalidated")
        if early_exit:
            reason_label = "momentum reversed" if trade.exit_reason == "early_warning_momentum_reversal" else "setup invalidated"
            title = f"EARLY EXIT WARNING: {trade.symbol} {trade.direction.value.upper()} -- {reason_label}"
            message = (
                f"Exiting {trade.symbol} {trade.direction.value} early ({reason_label}, before target/stop) -- "
                f"exit {trade.exit_price:.2f} | P&L {trade.pnl_points:+.2f} pts ({trade.r_multiple:+.2f}R)"
            )
        else:
            title = f"EXIT: {trade.symbol} {trade.direction.value.upper()} -- {trade.exit_reason}"
            message = f"Exit {trade.exit_price:.2f} | P&L {trade.pnl_points:+.2f} pts ({trade.r_multiple:+.2f}R)"
        return Alert(
            title=title,
            message=message,
            reasoning=trade.exit_reasoning,
            # stop_loss and early-exit-warning exits are the ones worth an
            # unmissable alert -- a target or eod_square_off exit is good/
            # neutral news that can wait to be read; a stop-out or an
            # early-exit warning are the ones you'd want to know about
            # immediately (e.g. to check nothing else needs attention).
            urgency="high" if trade.exit_reason == "stop_loss" or early_exit else "normal",
        )

    @staticmethod
    def _check_exit(trade: Trade, bar: pd.Series, current_time: pd.Timestamp) -> None:
        high, low = float(bar["high"]), float(bar["low"])
        if trade.direction == SignalDirection.LONG:
            if low <= trade.stop_loss:
                trade.exit_time, trade.exit_price, trade.exit_reason = current_time, trade.stop_loss, "stop_loss"
            elif high >= trade.target:
                trade.exit_time, trade.exit_price, trade.exit_reason = current_time, trade.target, "target"
        else:
            if high >= trade.stop_loss:
                trade.exit_time, trade.exit_price, trade.exit_reason = current_time, trade.stop_loss, "stop_loss"
            elif low <= trade.target:
                trade.exit_time, trade.exit_price, trade.exit_reason = current_time, trade.target, "target"

    def _check_early_exit(
        self, trade: Trade, mtf_view: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> None:
        """
        "Early exit warning" system: checked on every bar an open
        position exists, BEFORE the hard stop-loss/target (_check_exit)
        -- lets a deteriorating setup exit before price ever reaches
        those levels, not just resolve to the same stop-loss outcome
        later. Two independent checks, either can fire:

          1. Setup invalidation (per-strategy, via check_invalidation() --
             see each Strategy subclass for its specific condition: a
             broken breakout level re-crossed, market regime flipped,
             VWAP re-crossed, an order-block zone failed, a deeper
             Fibonacci level breached). Checked via getattr() duck-typing,
             NOT part of the Strategy ABC -- a strategy without this
             method (e.g. ReversalDetection, whose natural invalidation
             price is already identical to its stop_loss -- see that
             strategy's docstring for why a separate check there
             would be unreachable) simply never triggers this path.
          2. Momentum reversal (generic, every strategy) -- see
             _check_momentum_reversal().

        Sets trade.exit_time/exit_price/exit_reason/exit_reasoning
        in-place (same mutation pattern _check_exit uses) rather than
        returning a value -- caller checks trade.is_open afterward.
        """
        check_invalidation = getattr(self.strategy, "check_invalidation", None)
        if check_invalidation is not None:
            reasons = check_invalidation(trade, mtf_view, current_index)
            if reasons:
                self._fire_early_exit(trade, mtf_view, current_index, "early_warning_setup_invalidated", reasons)
                return

        reasons = self._check_momentum_reversal(trade, mtf_view, current_index)
        if reasons:
            self._fire_early_exit(trade, mtf_view, current_index, "early_warning_momentum_reversal", reasons)

    @staticmethod
    def _fire_early_exit(
        trade: Trade, mtf_view: dict[int, pd.DataFrame], current_index: pd.Timestamp,
        reason: str, reasons: list[str],
    ) -> None:
        current_close = float(mtf_view[1].loc[:current_index]["close"].iloc[-1])
        trade.exit_time = current_index
        trade.exit_price = current_close
        trade.exit_reason = reason
        trade.exit_reasoning = reasons

    def _check_momentum_reversal(
        self, trade: Trade, mtf_view: dict[int, pd.DataFrame], current_index: pd.Timestamp,
    ) -> Optional[list[str]]:
        """
        Generic (strategy-independent) early-exit check: has this
        position already moved favorably by a meaningful amount (>=
        EARLY_EXIT_MOMENTUM_MIN_PROFIT_ATR_MULT x ATR), and are the last
        EARLY_EXIT_MOMENTUM_LOOKBACK_BARS bars now reversing back toward
        entry/stop, with volume elevated in the adverse direction (reuses
        detectors.breakout_validator.volume_zscore, same threshold that
        module uses for "meaningfully above recent average")? Both
        conditions must hold -- reversing on quiet volume is treated as
        ordinary pullback noise, not a real reversal.
        """
        df = mtf_view.get(1)
        if df is None or df.empty:
            return None
        day_bars = df[df.index.date == current_index.date()]
        day_bars = day_bars.loc[:current_index]
        if len(day_bars) < EARLY_EXIT_MOMENTUM_LOOKBACK_BARS + 21:
            return None

        sign = 1.0 if trade.direction == SignalDirection.LONG else -1.0

        # Track the best favorable price seen so far -- reuses
        # Trade.peak_price (see that field's docstring for why reusing it
        # here, rather than a trailing-stop use, is safe).
        favorable_extreme = (
            float(day_bars["high"].iloc[-1]) if trade.direction == SignalDirection.LONG
            else float(day_bars["low"].iloc[-1])
        )
        if trade.peak_price is None:
            trade.peak_price = trade.entry_price
        if sign * (favorable_extreme - trade.peak_price) > 0:
            trade.peak_price = favorable_extreme

        profit_so_far = sign * (trade.peak_price - trade.entry_price)
        atr = average_true_range(day_bars.iloc[-30:])
        if atr <= 0 or profit_so_far < EARLY_EXIT_MOMENTUM_MIN_PROFIT_ATR_MULT * atr:
            return None  # hasn't moved favorably enough yet for a reversal to be meaningful

        recent_closes = day_bars["close"].iloc[-EARLY_EXIT_MOMENTUM_LOOKBACK_BARS:].to_numpy()
        if len(recent_closes) < EARLY_EXIT_MOMENTUM_LOOKBACK_BARS:
            return None
        reversing = all(
            sign * (recent_closes[i] - recent_closes[i - 1]) < 0 for i in range(1, len(recent_closes))
        )
        if not reversing:
            return None

        recent_volumes = day_bars["volume"].iloc[-21:-1]
        if len(recent_volumes) < 20:
            return None
        current_volume = float(day_bars["volume"].iloc[-1])
        vz = volume_zscore(current_volume, recent_volumes)
        if vz <= EARLY_EXIT_MOMENTUM_VOLUME_ZSCORE_THRESHOLD:
            return None

        return [
            f"Price reversed {EARLY_EXIT_MOMENTUM_LOOKBACK_BARS} bars against the {trade.direction.value} "
            f"position after reaching {profit_so_far:.2f} pts favorable ({profit_so_far / atr:.1f}x ATR)",
            f"Adverse-direction volume elevated: z-score {vz:.2f} "
            f"(> {EARLY_EXIT_MOMENTUM_VOLUME_ZSCORE_THRESHOLD} threshold)",
        ]

    def open_positions(self) -> dict[str, Trade]:
        return dict(self._open_trades)

    def force_close(self, symbol: str, price: float, reason: str, at_time: pd.Timestamp) -> None:
        """
        Immediately closes symbol's open trade (if any) at `price`,
        bypassing the normal stop/target check entirely -- the single
        shared code path for every kind of forced exit this engine needs:
        the per-bar EOD-square-off check above, the live poller's wall-
        clock-based EOD safeguard (paper_trading/live_poller.py, robust
        to a slow/delayed poll cycle missing the per-bar check), and a
        graceful-shutdown close (examples/run_live_session.py, using
        exit_reason="shutdown_close" to stay honestly distinct from a
        real EOD square-off in the journal). No-op if there's no open
        trade for this symbol.
        """
        open_trade = self._open_trades.get(symbol)
        if open_trade is None:
            return
        open_trade.exit_time = at_time
        open_trade.exit_price = price
        open_trade.exit_reason = reason
        self.journal.log_exit(open_trade)
        if self.notifier is not None:
            self.notifier.notify(self._exit_alert(open_trade))
        del self._open_trades[symbol]
