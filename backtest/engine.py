"""
Event-driven intraday backtest engine.

The single most important property of this engine, more important than
speed or features: **no lookahead bias**. At the moment a Strategy is asked
to evaluate bar N, it only ever sees data up to and including bar N. This
is enforced structurally (by slicing the DataFrames before passing them to
the strategy) rather than trusted to strategy authors, because lookahead
bias is the single most common way retail backtests lie to their author --
a "95% win rate" backtest that secretly peeked at the next candle is worse
than useless, it's actively dangerous, because it will fail exactly when
real money is on the line.

Other correctness properties this enforces:
  - A position is entered at the NEXT bar's open after a signal fires, not
    the signal bar's close (you can't trade a close you haven't seen yet
    in live conditions).
  - Stop-loss and target are checked using the bar's high/low, not close,
    so a wick that hits your stop counts as a stop-out even if price
    recovered by the close (matches how a real stop-loss order behaves).
  - Only one open position per symbol at a time (matches "avoid overtrading"
    being one of your own stated loss patterns -- the engine won't let a
    strategy pyramid into chaos by default).
  - A strategy is only asked for a signal when detectors.liquidity_analysis
    .analyze_liquidity() says the last 30 bars (up to and including the
    current bar -- no lookahead) are liquid enough to actually trade. A
    single-digit-share print on a near-dead stock can sit arbitrarily close
    to a breakout level by pure microstructure noise, producing a stop
    distance small enough to fake an absurd R-multiple (confirmed on real
    data: BANG produced a reported +180R off a 2-share fill and a 2-paise
    stop). This gate stops that at the source rather than trusting the
    ATR-based stop floor alone to catch it -- ATR computed from mostly
    zero-volume, flat-OHLC bars is itself near-zero, so that floor can be
    fooled by the exact same illiquidity that makes the fill meaningless.
  - A fill is voided if the gap between signal and fill has eaten more than
    half of the strategy's INTENDED risk, not just checked for direction
    sanity. The pre-existing valid_long/valid_short check alone only
    catches a gap that flips the setup's direction entirely -- it does
    NOT catch a smaller adverse gap that leaves the fill technically
    between stop and target but with realized risk (fill to stop) a
    fraction of what the strategy computed (fill to intended entry).
    Confirmed on real, perfectly liquid data: AFFLE filled 1.1 points
    against its own signal, collapsing a ~1.18-point intended risk down
    to 0.075 -- and reporting +46R off an ordinary move as a result.
    Liquidity alone doesn't prevent this; it's a fill-gap problem, not a
    thin-stock problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from backtest.strategy_base import Signal, SignalDirection, Strategy
from config.settings import MARKET_CLOSE, MARKET_OPEN
from data import storage
from data.resample import IncrementalResampler, multi_timeframe_view
from detectors.liquidity_analysis import analyze_liquidity

# A fill is voided if the signal-to-fill gap ate more than this fraction of
# the strategy's intended risk (see the gap-risk guard in run()/run_multi()).
MIN_REALIZED_RISK_FRACTION = 0.5


@dataclass
class Trade:
    symbol: str
    direction: SignalDirection
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    target: float
    reasoning: list[str]
    confidence: float = 0.5
    strategy_name: str = "unknown"
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "stop_loss" | "target" | "trailing_stop" | "eod_square_off" |
    # "shutdown_close" | "early_warning_momentum_reversal" | "early_warning_setup_invalidated"
    # (the last two: paper_trading/engine.py's early-exit warning system)
    # Specific, human-readable reason(s) behind an early-exit warning exit
    # (see paper_trading/engine.py's _check_early_exit()) -- distinct from
    # `reasoning` above, which is the ENTRY reasoning. Empty for every
    # other exit_reason (stop_loss/target/eod_square_off are already
    # self-explanatory from the price alone).
    exit_reasoning: list = field(default_factory=list)
    # Trailing-stop management state. initial_stop_loss freezes the ORIGINAL
    # stop at entry so risk_points/r_multiple stay computed against the risk
    # the strategy actually signaled, even after stop_loss itself is moved
    # to breakeven and then trailed (see _check_exit_trailing). Left None
    # for ordinary fixed-target trades, where stop_loss never moves anyway
    # and risk_points falling back to the live stop_loss is identical.
    initial_stop_loss: Optional[float] = None
    # Best favorable price seen since entry. Two independent uses, never
    # both active on the same Trade: BacktestEngine's trailing-stop mode
    # (_check_exit_trailing), and paper_trading/engine.py's momentum-
    # reversal early-exit check (which PaperTradingEngine always runs,
    # trailing mode or not -- see that module for why the same field is
    # safe to reuse rather than adding a near-duplicate one).
    peak_price: Optional[float] = None
    breakeven_moved: bool = False
    trailing_active: bool = False

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def pnl_points(self) -> float:
        if self.exit_price is None:
            return 0.0
        sign = 1 if self.direction == SignalDirection.LONG else -1
        return sign * (self.exit_price - self.entry_price)

    @property
    def risk_points(self) -> float:
        base_stop = self.initial_stop_loss if self.initial_stop_loss is not None else self.stop_loss
        return abs(self.entry_price - base_stop)

    @property
    def r_multiple(self) -> float:
        if self.risk_points == 0:
            return 0.0
        return self.pnl_points / self.risk_points


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    # run_multi()'s verification_config only: how many raw signals this
    # strategy fired that never became a trade because they were rejected
    # by CHANGE 2's pre-entry verification (min move / volume / no-
    # pullback) -- see detectors/pick_verification.py. (CHANGE 1's
    # fixed-target R:R filter used to reject here too before being
    # reverted 2026-07-22 -- verification_config now only ever applies
    # CHANGE 2, target is always the strategy's real Signal.target.)
    # Always 0 for run()/run_multi() calls that don't pass
    # verification_config, and for any strategy verification_config
    # doesn't name -- not a sign nothing was rejected, just that this
    # counter wasn't active for that run.
    rejected_signal_count: int = 0


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        timeframes: list[int],
        eod_square_off_time: str = "15:15",
        # Optional trailing-stop mode: OFF by default, so every existing
        # caller keeps the original fixed-target behavior unchanged. When
        # enabled, an open trade's target is ignored entirely -- the stop
        # is moved to breakeven once profit reaches breakeven_trigger_r
        # times the trade's INITIAL risk, then trailed at trail_distance_r
        # times that same initial risk behind the best price seen so far
        # once profit reaches trail_activate_r times initial risk. The
        # trade rides until the trailing stop is hit or EOD square-off --
        # see _check_exit_trailing for the exact bar-by-bar mechanics and
        # why this is still no-lookahead.
        use_trailing_stop: bool = False,
        breakeven_trigger_r: float = 1.0,
        trail_activate_r: float = 1.5,
        trail_distance_r: float = 0.5,
    ):
        self.strategy = strategy
        self.timeframes = sorted(set(timeframes) | {1})  # always need 1min base
        self.eod_square_off_time = eod_square_off_time
        self.use_trailing_stop = use_trailing_stop
        self.breakeven_trigger_r = breakeven_trigger_r
        self.trail_activate_r = trail_activate_r
        self.trail_distance_r = trail_distance_r

    def run(self, symbol: str, start_date: date, end_date: date,
            df_1min: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        df_1min: optional pre-loaded 1-minute data (e.g. from
        data.storage.load_combined(), which merges the permanent archive
        with the rolling cache -- see backtest/walk_forward.py). Omit
        (the default) to keep the original behavior of loading fresh via
        data.storage.load(symbol). Still filtered to [start_date,
        end_date] either way, so a caller can safely pass a full combined
        history and let this method narrow it down.
        """
        if df_1min is None:
            df_1min = storage.load(symbol)
        if df_1min.empty:
            raise ValueError(
                f"No cached data for {symbol}. Fetch it first via data/fetcher.py "
                f"and data/storage.py before backtesting."
            )

        df_1min = df_1min[
            (df_1min.index.date >= start_date) & (df_1min.index.date <= end_date)
        ]
        if df_1min.empty:
            raise ValueError(f"No cached data for {symbol} in range {start_date}..{end_date}")

        result = BacktestResult()
        open_trade: Optional[Trade] = None
        pending_signal: Optional[Signal] = None  # fires at NEXT bar's open

        # Perf note (two layers, both matter at 1-min resolution over weeks
        # of history): rebuilding multi_timeframe_view from df_1min.loc[:current_time]
        # on every bar, as this loop used to, re-resamples the ENTIRE history
        # -- including every day that's already fully in the past and can
        # never change -- from scratch, on every single bar (O(bars^2)).
        # Completed days are cached here ONCE, right after each day finishes.
        # Within the current day, calling resample() fresh on every bar is
        # STILL expensive even bounded to ~375 rows -- pandas' groupby+resample
        # machinery has real per-call overhead regardless of input size, and
        # that overhead times ~375 calls/day dominates once the cross-day
        # blowup above is fixed. IncrementalResampler updates the current
        # higher-timeframe bucket in O(1) per bar instead, only touching all
        # buckets-so-far (a much smaller number than 375) when the caller
        # actually reads the frame. Net effect: same data, same no-lookahead
        # guarantee, drastically fewer redundant aggregation passes.
        history_mtf: dict[int, pd.DataFrame] = {tf: df_1min.iloc[0:0] for tf in self.timeframes}
        higher_timeframes = [tf for tf in self.timeframes if tf != 1]

        for trade_date, day_df in df_1min.groupby(df_1min.index.date):
            market_hours_today = day_df.between_time(MARKET_OPEN, MARKET_CLOSE)
            day_origin = market_hours_today.index[0] if not market_hours_today.empty else day_df.index[0]
            resamplers = {tf: IncrementalResampler(tf, day_origin) for tf in higher_timeframes}

            for i, current_time in enumerate(day_df.index):
                current_bar = day_df.iloc[i]

                # Feed every bar into the running higher-timeframe buckets
                # unconditionally, regardless of whether the strategy is
                # actually queried this bar -- the incremental state must
                # reflect every 1-min bar seen so far, not just queried ones.
                for tf in higher_timeframes:
                    resamplers[tf].update(current_time, current_bar)

                # 1. Fill any pending signal at this bar's open (signal was
                #    generated on the previous bar -- see no-lookahead note above).
                if pending_signal is not None and open_trade is None:
                    fill_price = float(current_bar["open"])

                    # Slippage guard: stop/target were computed off the signal
                    # bar's close, but we fill at the NEXT bar's open. If price
                    # moved enough in that gap that the target/stop no longer
                    # make sense relative to the actual fill (e.g. a short's
                    # "target" ends up above the entry price because of an
                    # adverse gap), a real trader would recognize the setup
                    # invalidated before fill and skip it rather than take a
                    # trade with an inverted or zero risk-reward. Void it here
                    # instead of silently opening a broken trade.
                    valid_long = (
                        pending_signal.direction == SignalDirection.LONG
                        and pending_signal.stop_loss < fill_price < pending_signal.target
                    )
                    valid_short = (
                        pending_signal.direction == SignalDirection.SHORT
                        and pending_signal.target < fill_price < pending_signal.stop_loss
                    )

                    # Direction sanity alone isn't enough: an adverse gap can
                    # leave the fill sitting just barely inside [stop, target]
                    # while the REALIZED risk (fill_price to stop_loss) has
                    # collapsed to a sliver of what the strategy actually
                    # intended -- tight enough to fake a huge R-multiple off
                    # an ordinary move (confirmed on real data: AFFLE filled
                    # at 1433.10 against a stop of 1433.175 -- 0.075 points --
                    # when the strategy's own risk model called for ~1.18
                    # points, reporting +46R). Most strategies use a fixed
                    # target = entry +/- RR*risk convention (RR=2 by
                    # default), so stop_loss and target alone are enough to
                    # recover the INTENDED risk without this engine needing
                    # to know anything about how it was computed (ATR,
                    # level distance, etc.): |target - stop_loss| ==
                    # (1+RR) * intended_risk. RR is read off the strategy
                    # instance (defaulting to 2.0, the original hardcoded
                    # assumption) rather than hardcoded here, since
                    # ValidatedBreakout's risk_reward_multiple can now vary
                    # per instance (see detectors/validated_breakout_strategy.py) --
                    # hardcoding /3.0 unconditionally would silently
                    # miscompute intended_risk for any non-default RR.
                    rr = getattr(self.strategy, "risk_reward_multiple", 2.0)
                    intended_risk = abs(pending_signal.target - pending_signal.stop_loss) / (1.0 + rr)
                    realized_risk = abs(fill_price - pending_signal.stop_loss)
                    risk_intact = intended_risk <= 0 or realized_risk >= MIN_REALIZED_RISK_FRACTION * intended_risk

                    if (valid_long or valid_short) and risk_intact:
                        open_trade = Trade(
                            symbol=symbol,
                            direction=pending_signal.direction,
                            entry_time=current_time,
                            entry_price=fill_price,
                            stop_loss=pending_signal.stop_loss,
                            target=pending_signal.target,
                            reasoning=pending_signal.reasoning,
                            confidence=pending_signal.confidence,
                            strategy_name=self.strategy.name,
                            initial_stop_loss=pending_signal.stop_loss if self.use_trailing_stop else None,
                        )
                    pending_signal = None

                # 2. Manage open position: check stop-loss / target against
                #    this bar's high/low (not close -- see docstring).
                if open_trade is not None:
                    if self.use_trailing_stop:
                        self._check_exit_trailing(
                            open_trade, current_bar, current_time,
                            self.breakeven_trigger_r, self.trail_activate_r, self.trail_distance_r,
                        )
                    else:
                        self._check_exit(open_trade, current_bar, current_time)
                    if not open_trade.is_open:
                        result.trades.append(open_trade)
                        open_trade = None

                # 3. EOD square-off -- never hold an intraday position
                #    overnight, regardless of what the strategy wants.
                if open_trade is not None and current_time.strftime("%H:%M") >= self.eod_square_off_time:
                    open_trade.exit_time = current_time
                    open_trade.exit_price = float(current_bar["close"])
                    open_trade.exit_reason = "eod_square_off"
                    result.trades.append(open_trade)
                    open_trade = None
                    continue

                # 4. Ask the strategy for a signal using ONLY data up to and
                #    including current_time -- this slice is the lookahead guard.
                if open_trade is None and pending_signal is None:
                    today_1min = day_df.loc[:current_time]
                    mtf_view: dict[int, pd.DataFrame] = {
                        1: pd.concat([history_mtf[1], today_1min]) if not history_mtf[1].empty else today_1min,
                    }
                    for tf in higher_timeframes:
                        today_resampled = resamplers[tf].as_dataframe()
                        mtf_view[tf] = (
                            pd.concat([history_mtf[tf], today_resampled]) if not history_mtf[tf].empty
                            else today_resampled
                        )

                    # Liquidity gate: analyze_liquidity() itself slices to its
                    # own trailing lookback_bars (default 30) of whatever we
                    # pass it, so handing it the full no-lookahead mtf_view[1]
                    # is safe and cheap -- it never sees beyond current_time.
                    # None means "not enough bars yet to judge" -- don't block
                    # on that, only block on an actual illiquid reading.
                    liquidity = analyze_liquidity(mtf_view[1])
                    if liquidity is None or liquidity.is_liquid:
                        signal = self.strategy.evaluate(symbol, mtf_view, current_time)
                        if signal is not None and signal.direction != SignalDirection.NONE:
                            pending_signal = signal

            # Day fully complete -- fold it into the cache ONCE (not per-bar),
            # so it's never re-resampled again for the rest of this backtest.
            day_mtf = multi_timeframe_view(day_df, self.timeframes)
            for tf in self.timeframes:
                history_mtf[tf] = pd.concat([history_mtf[tf], day_mtf[tf]])

        return result

    @staticmethod
    def _check_exit(trade: Trade, bar: pd.Series, current_time: pd.Timestamp) -> None:
        high, low = float(bar["high"]), float(bar["low"])
        if trade.direction == SignalDirection.LONG:
            if low <= trade.stop_loss:
                trade.exit_time, trade.exit_price, trade.exit_reason = (
                    current_time, trade.stop_loss, "stop_loss",
                )
            elif high >= trade.target:
                trade.exit_time, trade.exit_price, trade.exit_reason = (
                    current_time, trade.target, "target",
                )
        else:  # SHORT
            if high >= trade.stop_loss:
                trade.exit_time, trade.exit_price, trade.exit_reason = (
                    current_time, trade.stop_loss, "stop_loss",
                )
            elif low <= trade.target:
                trade.exit_time, trade.exit_price, trade.exit_reason = (
                    current_time, trade.target, "target",
                )

    @staticmethod
    def _check_exit_trailing(
        trade: Trade,
        bar: pd.Series,
        current_time: pd.Timestamp,
        breakeven_trigger_r: float,
        trail_activate_r: float,
        trail_distance_r: float,
    ) -> None:
        """
        Trailing-stop exit management -- no fixed target at all, the trade
        rides until the (possibly-moved) stop is hit or EOD square-off.

        No-lookahead discipline: exactly like _check_exit, the stop-hit
        check happens FIRST on this bar's high/low, using whatever stop
        level was already in effect BEFORE this bar (i.e. as of the prior
        bar's close) -- only after that check passes do we let this bar's
        own favorable move (its high for a long, its low for a short)
        advance the peak and possibly ratchet the stop forward for FUTURE
        bars. This mirrors the existing "stop takes priority over target
        within the same bar" ordering in _check_exit: a bar that both
        would have triggered a stop AND made a new high is resolved as a
        stop-out, the conservative assumption for intrabar sequencing.

        risk is always measured against initial_stop_loss (the risk the
        strategy actually signaled at entry), never the live (possibly
        already-moved) stop_loss -- otherwise moving to breakeven would
        shrink "risk" and make the breakeven/trail triggers fire earlier
        and earlier as the stop moves, which isn't the intended behavior.
        """
        high, low = float(bar["high"]), float(bar["low"])
        if trade.initial_stop_loss is None:
            return  # trailing mode always sets this at Trade creation; guard anyway
        risk = abs(trade.entry_price - trade.initial_stop_loss)
        if risk <= 0:  # degenerate (zero risk) -- nothing sensible to trail against
            return

        if trade.direction == SignalDirection.LONG:
            if low <= trade.stop_loss:
                reason = "trailing_stop" if (trade.breakeven_moved or trade.trailing_active) else "stop_loss"
                trade.exit_time, trade.exit_price, trade.exit_reason = current_time, trade.stop_loss, reason
                return

            trade.peak_price = high if trade.peak_price is None else max(trade.peak_price, high)
            favorable_move = trade.peak_price - trade.entry_price

            if not trade.breakeven_moved and favorable_move >= breakeven_trigger_r * risk:
                trade.stop_loss = max(trade.stop_loss, trade.entry_price)
                trade.breakeven_moved = True
            if favorable_move >= trail_activate_r * risk:
                trade.trailing_active = True
            if trade.trailing_active:
                trade.stop_loss = max(trade.stop_loss, trade.peak_price - trail_distance_r * risk)
        else:  # SHORT
            if high >= trade.stop_loss:
                reason = "trailing_stop" if (trade.breakeven_moved or trade.trailing_active) else "stop_loss"
                trade.exit_time, trade.exit_price, trade.exit_reason = current_time, trade.stop_loss, reason
                return

            trade.peak_price = low if trade.peak_price is None else min(trade.peak_price, low)
            favorable_move = trade.entry_price - trade.peak_price

            if not trade.breakeven_moved and favorable_move >= breakeven_trigger_r * risk:
                trade.stop_loss = min(trade.stop_loss, trade.entry_price)
                trade.breakeven_moved = True
            if favorable_move >= trail_activate_r * risk:
                trade.trailing_active = True
            if trade.trailing_active:
                trade.stop_loss = min(trade.stop_loss, trade.peak_price + trail_distance_r * risk)


@dataclass
class _StrategyRunState:
    """Per-strategy trade-management state for run_multi() -- everything
    BacktestEngine.run() tracks in local variables (open_trade, pending_signal,
    the result), just one instance per strategy instead of per call."""
    strategy: Strategy
    timeframes: list[int]  # this strategy's own declared timeframes, incl. 1
    result: BacktestResult = field(default_factory=BacktestResult)
    open_trade: Optional[Trade] = None
    pending_signal: Optional[Signal] = None
    # Same optional trailing-stop mode as BacktestEngine -- per-strategy so
    # run_multi can compare a fixed-target variant against a trailing
    # variant of the same underlying strategy in one shared-data pass (see
    # examples/run_validated_breakout_variants.py). Defaults keep every
    # existing run_multi() caller's fixed-target behavior unchanged.
    use_trailing_stop: bool = False
    breakeven_trigger_r: float = 1.0
    trail_activate_r: float = 1.5
    trail_distance_r: float = 0.5
    # verification_config only (see run_multi()'s docstring): mirrors
    # paper_trading/engine.py's PaperTradingEngine live_mode + _verifying
    # exactly -- the RIGOROUS wait-state version, not detectors/scanner.py's
    # fast-scan approximation, since run_multi() already has true bar-by-
    # bar iteration with real cross-bar state (no trigger-time inference
    # needed). verifying holds {"signal", "trigger_time", "reference_price"}
    # for a signal that fired and passed its instant move/volume checks
    # but is still waiting on check_continuation()'s multi-bar verdict.
    verification_mode: bool = False
    verifying: Optional[dict] = None
    rejected_signal_count: int = 0


def run_multi(
    symbol: str,
    strategies: dict[str, tuple[Strategy, list[int]]],
    start_date: date,
    end_date: date,
    eod_square_off_time: str = "15:15",
    trailing_config: Optional[dict[str, dict]] = None,
    df_1min: Optional[pd.DataFrame] = None,
    verification_config: Optional[dict[str, bool]] = None,
) -> dict[str, BacktestResult]:
    """
    Runs MULTIPLE strategies against the SAME symbol in a single pass,
    sharing the one-time storage.load() and multi-timeframe resampling
    across all of them, instead of each strategy redoing both from scratch.

    Why this exists: BacktestEngine.run() is one strategy, one symbol, one
    disk load, one set of resample buckets -- correct in isolation, but
    examples/run_universe_backtest.py calls it once per (symbol, strategy)
    pair to compare strategies across a universe. Since 3 of the 4 project
    strategies share overlapping timeframe sets ([1,5,15], [1,5,15], [1,15],
    [1]), running them as separate BacktestEngine.run() calls means the same
    symbol's 1-min data gets loaded from disk up to 4x, the 5-min resample
    gets rebuilt independently twice, and the 15-min resample gets rebuilt
    independently three times -- all producing identical output each time.
    This function loads and resamples once (for the UNION of timeframes any
    strategy needs) and evaluates every strategy off that same shared data
    each bar, while keeping each strategy's open_trade/pending_signal/result
    fully independent -- only the price data is shared, never trade state.

    strategies: {name: (Strategy instance, its required timeframes)}.
    trailing_config: optional {name: {"use_trailing_stop": True,
        "breakeven_trigger_r": ..., "trail_activate_r": ..., "trail_distance_r": ...}},
        keyed by the SAME names as `strategies`. Omit a name (or the whole
        argument) to keep that strategy's existing fixed-target behavior --
        this only turns trailing ON for names explicitly listed.
    df_1min: optional pre-loaded 1-minute data (e.g. from
        data.storage.load_combined() -- see backtest/walk_forward.py).
        Omit (the default) to keep the original behavior of loading fresh
        via data.storage.load(symbol). Still filtered to [start_date,
        end_date] either way.
    verification_config: optional {name: True}, keyed by the SAME names
        as `strategies` -- turns on CHANGE 2 (pre-entry momentum/volume/
        no-pullback verification, for strategies in detectors.
        pick_verification.VERIFIED_STRATEGIES) for exactly the strategies
        listed, using the SAME logic paper_trading/engine.py's
        PaperTradingEngine live_mode uses -- the rigorous bar-by-bar
        wait-state version, not detectors/scanner.py's fast-scan
        approximation (this function already has true bar-by-bar
        iteration with real cross-bar state, so there's no need to infer
        a trigger bar after the fact). Target is ALWAYS the strategy's
        real 2x-risk-reward Signal.target, verification_config or not --
        CHANGE 1 (fixed-point target) used to also apply here before
        being reverted 2026-07-22 across every live_mode/
        verification_config path (a 400-symbol backtest showed it hurt 3
        of 4 verified strategies; see detectors/scanner.py's
        FIXED_TARGET_POINTS docstring). Omit a name (or the whole
        argument) to keep that strategy's original behavior (no
        verification either) completely unchanged -- this is how
        examples/run_universe_backtest.py and every other existing caller
        keeps producing its original, on-record numbers.
    Returns {name: BacktestResult}, one per input strategy. BacktestResult.
        rejected_signal_count is only meaningful for names verification_
        config turned on.
    """
    if df_1min is None:
        df_1min = storage.load(symbol)
    if df_1min.empty:
        raise ValueError(
            f"No cached data for {symbol}. Fetch it first via data/fetcher.py "
            f"and data/storage.py before backtesting."
        )
    df_1min = df_1min[(df_1min.index.date >= start_date) & (df_1min.index.date <= end_date)]
    if df_1min.empty:
        raise ValueError(f"No cached data for {symbol} in range {start_date}..{end_date}")

    trailing_config = trailing_config or {}
    verification_config = verification_config or {}
    states = {
        name: _StrategyRunState(
            strategy=strat, timeframes=sorted(set(tfs) | {1}),
            verification_mode=bool(verification_config.get(name, False)),
            **{k: v for k, v in trailing_config.get(name, {}).items()},
        )
        for name, (strat, tfs) in strategies.items()
    }

    # Lazy import (function-level, not module-level): detectors/scanner.py
    # imports strategies (e.g. detectors/fibonacci_strategy.py) that
    # themselves do `from backtest.engine import Trade` for type hints --
    # a module-level import of anything that pulls in detectors.scanner
    # here would risk a circular import while THIS module is still being
    # loaded (Trade not yet defined). By the time run_multi() actually
    # executes, backtest.engine is fully loaded, so this is safe -- same
    # reasoning as detectors/scanner.py's own lazy imports of composite_score/
    # sector_momentum/pick_verification.
    if any(state.verification_mode for state in states.values()):
        from detectors.pick_verification import (
            VERIFIED_STRATEGIES,
            check_continuation,
            check_minimum_move,
            check_volume_confirmation,
        )
    all_timeframes = sorted(set(tf for state in states.values() for tf in state.timeframes))
    higher_timeframes = [tf for tf in all_timeframes if tf != 1]

    history_mtf: dict[int, pd.DataFrame] = {tf: df_1min.iloc[0:0] for tf in all_timeframes}

    for trade_date, day_df in df_1min.groupby(df_1min.index.date):
        market_hours_today = day_df.between_time(MARKET_OPEN, MARKET_CLOSE)
        day_origin = market_hours_today.index[0] if not market_hours_today.empty else day_df.index[0]
        resamplers = {tf: IncrementalResampler(tf, day_origin) for tf in higher_timeframes}

        for i, current_time in enumerate(day_df.index):
            current_bar = day_df.iloc[i]

            for tf in higher_timeframes:
                resamplers[tf].update(current_time, current_bar)

            today_1min = day_df.loc[:current_time]
            full_mtf_view: dict[int, pd.DataFrame] = {
                1: pd.concat([history_mtf[1], today_1min]) if not history_mtf[1].empty else today_1min,
            }
            for tf in higher_timeframes:
                today_resampled = resamplers[tf].as_dataframe()
                full_mtf_view[tf] = (
                    pd.concat([history_mtf[tf], today_resampled]) if not history_mtf[tf].empty
                    else today_resampled
                )

            # Liquidity gate (see module docstring for why): computed ONCE
            # per bar here, not once per strategy below -- it's a property
            # of the symbol's data at this bar, not of any one strategy, so
            # there's no reason to recompute it 4x. Same no-lookahead
            # discipline as everything else: analyze_liquidity() only ever
            # sees full_mtf_view[1], which is truncated to current_time.
            liquidity = analyze_liquidity(full_mtf_view[1])
            is_liquid_now = liquidity is None or liquidity.is_liquid

            for state in states.values():
                # 1. Fill any pending signal at this bar's open (same
                #    slippage-void guard as BacktestEngine.run()).
                if state.pending_signal is not None and state.open_trade is None:
                    ps = state.pending_signal
                    fill_price = float(current_bar["open"])
                    valid_long = ps.direction == SignalDirection.LONG and ps.stop_loss < fill_price < ps.target
                    valid_short = ps.direction == SignalDirection.SHORT and ps.target < fill_price < ps.stop_loss

                    # Gap-risk guard -- see BacktestEngine.run()'s matching
                    # comment for the full explanation and the real-data case
                    # (AFFLE, +46R) that exposed this. Same derivation, RR
                    # read per-strategy (default 2.0) rather than hardcoded:
                    # |target - stop_loss| == (1+RR) * intended_risk.
                    rr = getattr(state.strategy, "risk_reward_multiple", 2.0)
                    intended_risk = abs(ps.target - ps.stop_loss) / (1.0 + rr)
                    realized_risk = abs(fill_price - ps.stop_loss)
                    risk_intact = intended_risk <= 0 or realized_risk >= MIN_REALIZED_RISK_FRACTION * intended_risk

                    if (valid_long or valid_short) and risk_intact:
                        state.open_trade = Trade(
                            symbol=symbol, direction=ps.direction, entry_time=current_time,
                            entry_price=fill_price, stop_loss=ps.stop_loss, target=ps.target,
                            reasoning=ps.reasoning, confidence=ps.confidence,
                            strategy_name=state.strategy.name,
                            initial_stop_loss=ps.stop_loss if state.use_trailing_stop else None,
                        )
                    state.pending_signal = None

                # 2. Manage open position.
                if state.open_trade is not None:
                    if state.use_trailing_stop:
                        BacktestEngine._check_exit_trailing(
                            state.open_trade, current_bar, current_time,
                            state.breakeven_trigger_r, state.trail_activate_r, state.trail_distance_r,
                        )
                    else:
                        BacktestEngine._check_exit(state.open_trade, current_bar, current_time)
                    if not state.open_trade.is_open:
                        state.result.trades.append(state.open_trade)
                        state.open_trade = None

                # 3. EOD square-off.
                if state.open_trade is not None and current_time.strftime("%H:%M") >= eod_square_off_time:
                    state.open_trade.exit_time = current_time
                    state.open_trade.exit_price = float(current_bar["close"])
                    state.open_trade.exit_reason = "eod_square_off"
                    state.result.trades.append(state.open_trade)
                    state.open_trade = None
                    continue

                # 4. Ask this strategy for a signal, using only the
                #    timeframes it declared needing -- a subset view of the
                #    shared data computed once above, not recomputed here.
                if state.open_trade is None and state.pending_signal is None and is_liquid_now:
                    # 4a. verification_config only: continue an in-progress
                    # "no immediate pullback" wait for a signal that already
                    # fired and passed its instant move/volume checks. Real
                    # trigger_time (no inference needed, unlike detectors/
                    # scanner.py's fast-scan path) -- this loop already
                    # visits every real bar in order.
                    if state.verification_mode and state.verifying is not None:
                        status, _ = check_continuation(
                            today_1min, state.verifying["trigger_time"],
                            state.verifying["signal"].trigger_level, state.verifying["signal"].direction,
                        )
                        if status == "pending":
                            continue
                        vsignal = state.verifying["signal"]
                        state.verifying = None
                        if status == "failed":
                            state.rejected_signal_count += 1
                            continue
                        # "passed" -- promote with its ORIGINAL Signal.target
                        # (real 2x-risk-reward) untouched. CHANGE 1 (fixed-
                        # point target) applied here too before being
                        # reverted 2026-07-22 across every live_mode/
                        # verification_config path -- see detectors/
                        # scanner.py's FIXED_TARGET_POINTS docstring for why
                        # (a 400-symbol backtest showed it hurt 3 of 4
                        # verified strategies).
                        state.pending_signal = vsignal
                        continue

                    mtf_view = {tf: full_mtf_view[tf] for tf in state.timeframes}
                    signal = state.strategy.evaluate(symbol, mtf_view, current_time)
                    if signal is None or signal.direction == SignalDirection.NONE:
                        continue

                    # verification_config: CHANGE 2 (the multi-bar wait)
                    # applies only to VERIFIED_STRATEGIES; every other
                    # strategy goes straight to a pending signal with its
                    # real Signal.target, exactly as if verification_config
                    # weren't set at all. reference_price (signal-time bar's
                    # close) is only needed for the instant move/volume
                    # checks below.
                    if state.verification_mode and state.strategy.name in VERIFIED_STRATEGIES:
                        reference_price = float(current_bar["close"])
                        move_ok, _ = check_minimum_move(today_1min, reference_price, signal.direction)
                        vol_ok, _ = check_volume_confirmation(today_1min)
                        if move_ok and vol_ok:
                            state.verifying = {
                                "signal": signal, "trigger_time": current_time, "reference_price": reference_price,
                            }
                        else:
                            state.rejected_signal_count += 1
                        continue

                    state.pending_signal = signal

        day_mtf = multi_timeframe_view(day_df, all_timeframes)
        for tf in all_timeframes:
            history_mtf[tf] = pd.concat([history_mtf[tf], day_mtf[tf]])

    for state in states.values():
        state.result.rejected_signal_count = state.rejected_signal_count
    return {name: state.result for name, state in states.items()}
