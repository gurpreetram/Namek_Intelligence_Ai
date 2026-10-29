"""
Walk-forward / out-of-sample CONSISTENCY testing -- not classic walk-
forward OPTIMIZATION. None of this project's 4 strategies have tunable
free parameters being searched over (each detector module's own docstring
documents fixed, non-optimized thresholds), so there is nothing to
re-optimize per period the way classic walk-forward analysis does. What
this module answers instead: does a strategy's performance (or lack of
it) hold up consistently across two different time windows, or does it
swing wildly between them -- a sign that any apparent edge seen in a
single backtest window is more likely noise/luck than a real, time-stable
pattern. This is the same instinct behind this project's own history of
re-checking a promising result on a bigger sample (e.g. the trailing-stop
400-symbol-subset-vs-full-2373-universe comparison, where TrendContinuation's
avg R-multiple flipped sign between the two) -- this module makes that
kind of check a reusable, structured tool instead of a one-off script.

Uses data.storage.load_combined() (the permanent archive merged with the
rolling ~30-day cache -- see examples/daily_archive.py) so the usable
history, and therefore how meaningful a result here can be, grows as the
archive accumulates more days over time.

HONEST LIMITATION as of 2026-07-13: the archive currently holds only ONE
day (2026-07-10) beyond what the ~30-day rolling cache already covers, so
a run today is a MECHANISM validation only -- confirming the split/run/
compare/verdict pipeline works correctly end-to-end -- NOT a trustworthy
consistency verdict. A real verdict needs the in-sample and out-of-sample
periods to be drawn from meaningfully different market conditions, which
requires weeks/months more archived history, not an arbitrary cut through
the same ~30-day window. See run_consistency_check()'s `mechanism_test_only`
return flag and format_report()'s handling of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from backtest.engine import run_multi
from backtest.metrics import PerformanceReport, compute
from data import storage
from detectors.reversal_strategy import ReversalDetection
from detectors.trend_continuation_strategy import TrendContinuation
from detectors.validated_breakout_strategy import ValidatedBreakout
from examples.opening_range_breakout import OpeningRangeBreakout

# Same builder pattern as examples/run_universe_backtest.py's
# STRATEGY_BUILDERS -- the actually-backtested default parameters, not
# loosened for this module. Zero-arg lambdas (not shared instances) so a
# fresh, state-free strategy object is built for every period/symbol --
# sharing one instance across periods would leak per-instance state
# (_traded_today, _pending_confirmation, etc.) between them, the same
# trap detectors/scanner.py's own docstring warns about for repeated
# scans of one symbol.
STRATEGY_BUILDERS = {
    "opening_range_breakout": (lambda: OpeningRangeBreakout(), [1, 5, 15]),
    "validated_breakout": (lambda: ValidatedBreakout(), [1, 5, 15]),
    "trend_continuation": (lambda: TrendContinuation(), [1, 15]),
    "reversal_detection": (lambda: ReversalDetection(), [1]),
}

# Matches portfolio/analytics.py's own default (min_reliable_sample=20) --
# same "is this sample big enough to say anything" bar used everywhere
# else in this project's reporting, not a new number invented for this
# module specifically.
DEFAULT_MIN_RELIABLE_SAMPLE = 20

# Below this many archived-only trading days (beyond the rolling cache),
# treat any result as a mechanism test rather than a real verdict -- see
# module docstring. 10 trading days is a deliberately low bar (about two
# weeks) chosen only to distinguish "we've barely started archiving" from
# "there's real extra history now"; it is NOT a claim that 10 days is
# enough for a trustworthy verdict, only enough to stop calling it a pure
# mechanism test. Real trustworthiness needs far more (weeks/months).
MIN_ARCHIVE_DAYS_FOR_REAL_VERDICT = 10


@dataclass
class PeriodSplit:
    in_sample_start: date
    in_sample_end: date
    out_sample_start: date
    out_sample_end: date


def split_date_range(start_date: date, end_date: date, split_ratio: float = 0.7) -> PeriodSplit:
    """
    Splits [start_date, end_date] into an earlier in-sample period and a
    later out-of-sample period, proportioned by CALENDAR days (not
    trading days -- simpler, and immaterial for a ratio split since
    trading-day density is roughly constant regardless of where within
    the range you cut).

    split_ratio: fraction of the range assigned to in-sample. Default
    0.70 (70% in-sample / 30% out-of-sample) is the conventional
    walk-forward default, not tuned for this project specifically.
    """
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
    if start_date >= end_date:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")

    total_days = (end_date - start_date).days
    split_offset = max(1, int(round(total_days * split_ratio)))
    in_sample_end = start_date + timedelta(days=split_offset)
    out_sample_start = in_sample_end + timedelta(days=1)

    return PeriodSplit(
        in_sample_start=start_date, in_sample_end=in_sample_end,
        out_sample_start=out_sample_start, out_sample_end=end_date,
    )


@dataclass
class ConsistencyResult:
    strategy_name: str
    in_sample: PerformanceReport
    out_sample: PerformanceReport
    verdict: str  # "CONSISTENT" | "INCONSISTENT" | "INSUFFICIENT_DATA"
    reasoning: list[str] = field(default_factory=list)


def _classify_consistency(
    in_sample: PerformanceReport, out_sample: PerformanceReport, min_reliable_sample: int,
) -> tuple[str, list[str]]:
    reasoning = [
        f"In-sample: {in_sample.total_trades} trades, {in_sample.win_rate:.1%} win rate, "
        f"{in_sample.avg_r_multiple:+.3f}R avg. Out-of-sample: {out_sample.total_trades} trades, "
        f"{out_sample.win_rate:.1%} win rate, {out_sample.avg_r_multiple:+.3f}R avg.",
    ]

    if in_sample.total_trades < min_reliable_sample or out_sample.total_trades < min_reliable_sample:
        reasoning.append(
            f"Need >= {min_reliable_sample} trades in BOTH periods to judge consistency -- "
            f"too few in at least one period."
        )
        return "INSUFFICIENT_DATA", reasoning

    # Win rate and avg R-multiple are the two headline numbers this
    # project already reports everywhere else (portfolio/analytics.py,
    # the dashboard's ConfidenceWarning) -- reused here rather than a new
    # composite metric, so "consistent" means the same thing a reader of
    # those other reports would already expect.
    win_rate_gap = abs(in_sample.win_rate - out_sample.win_rate)
    avg_r_gap = abs(in_sample.avg_r_multiple - out_sample.avg_r_multiple)
    sign_flipped = (in_sample.avg_r_multiple > 0) != (out_sample.avg_r_multiple > 0)

    # Thresholds below are NOT statistically derived (that would need many
    # more such comparisons than this project has run) -- they're matched
    # to the sign-flip pattern this project has already observed and
    # flagged as evidence of instability (TrendContinuation's avg R went
    # +0.06 -> -0.03 between a 400-symbol subset and the full 2373-symbol
    # universe). A >15 percentage-point win-rate swing, an avg-R gap over
    # 0.3, or an outright sign flip are treated as "meaningfully different".
    if win_rate_gap > 0.15 or sign_flipped or avg_r_gap > 0.3:
        reasoning.append(
            f"Win rate gap {win_rate_gap:.1%}, avg-R gap {avg_r_gap:.3f}"
            + (", avg R-multiple flips sign between periods" if sign_flipped else "")
            + " -- meaningfully different between periods."
        )
        return "INCONSISTENT", reasoning

    reasoning.append(f"Win rate gap {win_rate_gap:.1%}, avg-R gap {avg_r_gap:.3f} -- within a stable range.")
    return "CONSISTENT", reasoning


def _archived_trading_days(symbols: list[str]) -> int:
    """
    How many DISTINCT dates exist in the permanent archive alone (not the
    rolling cache) across a sample of symbols -- used only to decide
    whether to label a run a mechanism test. Checks up to 5 symbols and
    takes the max, since not every symbol is guaranteed to be archived
    yet (examples/daily_archive.py skips unresolvable/failed symbols).
    """
    best = 0
    for symbol in symbols[:5]:
        archive = storage.load_archive(symbol)
        if not archive.empty:
            best = max(best, len(set(archive.index.date)))
    return best


def run_consistency_check(
    symbols: list[str], start_date: date, end_date: date,
    split_ratio: float = 0.7, min_reliable_sample: int = DEFAULT_MIN_RELIABLE_SAMPLE,
) -> tuple[dict[str, ConsistencyResult], bool]:
    """
    Runs all 4 strategies across `symbols`, split into in-sample/out-of-
    sample periods (see split_date_range()), pooling every symbol's
    trades together per strategy per period (same aggregation style as
    backtest/metrics.py's compute() -- one PerformanceReport per pool, not
    a per-symbol breakdown). Works identically for a single symbol
    (pass a 1-element list) or a full universe.

    Pulls each symbol's data via data.storage.load_combined() (archive +
    rolling cache) exactly once, then reuses that same DataFrame for both
    periods via backtest.engine.run_multi()'s df_1min parameter -- no
    redundant re-fetching or re-loading between the two runs.

    Returns (results, mechanism_test_only) -- see this module's docstring
    for what mechanism_test_only means and why it matters.
    """
    split = split_date_range(start_date, end_date, split_ratio)
    mechanism_test_only = _archived_trading_days(symbols) < MIN_ARCHIVE_DAYS_FOR_REAL_VERDICT

    in_sample_trades: dict[str, list] = {name: [] for name in STRATEGY_BUILDERS}
    out_sample_trades: dict[str, list] = {name: [] for name in STRATEGY_BUILDERS}
    skipped = 0

    for symbol in symbols:
        df_1min = storage.load_combined(symbol)
        if df_1min.empty:
            skipped += 1
            continue
        try:
            in_specs = {name: (build(), tfs) for name, (build, tfs) in STRATEGY_BUILDERS.items()}
            in_results = run_multi(symbol, in_specs, split.in_sample_start, split.in_sample_end, df_1min=df_1min)
            out_specs = {name: (build(), tfs) for name, (build, tfs) in STRATEGY_BUILDERS.items()}
            out_results = run_multi(symbol, out_specs, split.out_sample_start, split.out_sample_end, df_1min=df_1min)
        except ValueError:
            # No data for this symbol in one or both periods (e.g. it was
            # only added to the cache partway through the range) -- a
            # symbol without data in BOTH periods can't inform a
            # consistency check either way. Same "one bad symbol shouldn't
            # kill the whole run" principle as detectors/scanner.py's
            # scan_universe().
            skipped += 1
            continue

        for name in STRATEGY_BUILDERS:
            in_sample_trades[name].extend(in_results[name].trades)
            out_sample_trades[name].extend(out_results[name].trades)

    results: dict[str, ConsistencyResult] = {}
    for name in STRATEGY_BUILDERS:
        in_perf = compute(in_sample_trades[name])
        out_perf = compute(out_sample_trades[name])
        verdict, reasoning = _classify_consistency(in_perf, out_perf, min_reliable_sample)
        if skipped:
            reasoning.append(f"{skipped}/{len(symbols)} symbols skipped (no usable data in one or both periods).")
        results[name] = ConsistencyResult(
            strategy_name=name, in_sample=in_perf, out_sample=out_perf,
            verdict=verdict, reasoning=reasoning,
        )
    return results, mechanism_test_only


def format_report(results: dict[str, ConsistencyResult], mechanism_test_only: bool) -> str:
    lines = []
    if mechanism_test_only:
        lines.append("=" * 72)
        lines.append("MECHANISM TEST ONLY -- NOT A REAL CONSISTENCY VERDICT")
        lines.append("The archive currently holds too little history beyond the rolling")
        lines.append("~30-day cache for the in-sample/out-of-sample periods to represent")
        lines.append("genuinely different market conditions. This run only confirms the")
        lines.append("split/run/compare/verdict pipeline works correctly end-to-end -- re-run")
        lines.append("after examples/daily_archive.py has accumulated weeks/months more")
        lines.append("history before treating any verdict below as trustworthy.")
        lines.append("=" * 72)
        lines.append("")

    for name, r in results.items():
        lines.append(f"{name}: {r.verdict}")
        lines.append(
            f"  in-sample:     {r.in_sample.total_trades:>4d} trades, {r.in_sample.win_rate:>6.1%} win rate, "
            f"{r.in_sample.avg_r_multiple:+.3f}R avg, PF {r.in_sample.profit_factor:.2f}"
        )
        lines.append(
            f"  out-of-sample: {r.out_sample.total_trades:>4d} trades, {r.out_sample.win_rate:>6.1%} win rate, "
            f"{r.out_sample.avg_r_multiple:+.3f}R avg, PF {r.out_sample.profit_factor:.2f}"
        )
        for reason in r.reasoning:
            lines.append(f"  - {reason}")
        lines.append("")

    return "\n".join(lines)
