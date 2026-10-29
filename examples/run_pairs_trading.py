"""
Tests statistical pairs trading -- structurally distinct from every prior
hypothesis test in this project: instead of predicting a single
instrument's own direction, this bets on the RELATIONSHIP between two
correlated, cointegrated stocks reverting to its historical norm, held
market-neutral (equal rupee-weighted long one leg / short the other).

PRE-REGISTERED BARS (same as every other hypothesis test in this
project): day-level Sharpe (non-annualized) >= 0.5 AND day-level win
rate >= 60%, evaluated SEPARATELY for in-sample and out-of-sample --
per the task brief, only the OUT-OF-SAMPLE result is the real test;
in-sample is reported for reference/comparison only, exactly as every
other walk-forward-validated finding in quant research is presented.

STEP 1 -- PAIR SELECTION, IN-SAMPLE ONLY (a deliberate, documented
departure from a literal "correlate across all cached history" reading):
correlation AND cointegration AND the hedge ratio are all fit using ONLY
the first IN_SAMPLE_FRACTION of each candidate pair's common cached
trading days. Selecting pairs using correlation computed over the FULL
history (including the out-of-sample test window) would itself be a
look-ahead/overfitting leak -- exactly the "in-sample overfitting of the
spread parameters" the task brief flags as the reason a walk-forward
split is critical in the first place. Fitting selection AND trading
parameters on the same in-sample-only window keeps the out-of-sample
test genuinely blind.

Liquidity filter (detectors.liquidity_analysis.analyze_liquidity())
applied FROM THE START: both legs of a pair must be liquid BEFORE any
correlation/cointegration work happens for that pair (universe-level
screen, mirroring examples/run_etf_mean_reversion.py's Step 1 pattern),
AND liquidity is re-checked on each leg's own PRIOR day before any
specific trading day is even considered eligible (same per-day gate
every other script in this project applies).

Sectors: config/sector_mapping.json (15 sectors, ~181 symbols) -- pairs
are only formed WITHIN a sector, never across.

Correlation: Pearson correlation of DAILY returns (last 1-min close of
each cached day), >= CORRELATION_THRESHOLD, computed in-sample only.

Cointegration: statsmodels.tsa.stattools.coint() (Engle-Granger) on
LOG daily closing prices, in-sample only, p-value < 0.05. HONEST
CAVEAT, stated up front and repeated in the final report: with an
in-sample window of ~15-20 daily observations (Upstox's ~29-33 day
rolling cache split 60/40), this test has genuinely low statistical
power by any normal quant-research standard (cointegration studies
conventionally want 50-100+ observations) -- results are reported
honestly regardless, but a "cointegrated" verdict here is far less
reliable than the same test run on a proper multi-year sample.

Hedge ratio: OLS slope of log(Price_A) ~ const + hedge_ratio*log(Price_B),
in-sample only (statsmodels.api.OLS) -- the SAME regression coint()
uses internally, refit here to also recover the coefficient (coint()
itself only returns the test statistic/p-value, not the hedge ratio).

STEP 2 -- SPREAD, Z-SCORE, ENTRY/EXIT: spread = log(Price_A) -
hedge_ratio*log(Price_B) (no intercept term in the TRADED spread, per
the task's literal formula -- the rolling mean-subtraction in the
z-score absorbs any constant level offset anyway, so omitting it here
changes nothing about the trading signal, only where the "centering"
happens). Z-score = (today's current spread - trailing 20-TRADING-DAY
mean) / (trailing 20-day std), where "trailing" means the 20 days
strictly BEFORE the day being traded, computed from EOD daily spread
values (no lookahead) -- this rolling normalization is allowed to look
back across the in-sample/out-of-sample boundary (e.g. an out-of-sample
day's 20-day window can include some in-sample days) because it's a
descriptive rolling statistic recomputed fresh from already-past data
every day, not a FITTED parameter -- the walk-forward discipline that
matters is that correlation/cointegration/hedge-ratio (STEP 1) are
NEVER re-estimated using in-sample-vs-out-of-sample information, which
this script enforces (they're computed once, in-sample only, and reused
unchanged for every day including the out-of-sample tail).

Entry: |z| >= 2.0 on a bar's close -> next COMMON bar's open for BOTH
legs (no-lookahead, matching backtest/engine.py's convention). z > 0
(A rich vs B): SHORT A, LONG B. z < 0: LONG A, SHORT B. Exit: |z| <= 0.5
on a later bar's close -> next common bar's open for both legs, OR EOD
square-off (15:15) for both legs at that bar's own close (same EOD
convention as every other script, hand-rolled here rather than reusing
detectors.cross_sectional_ranking.bar_at_or_before() since that helper
operates on one symbol at a time and this needs the LAST bar timestamp
common to BOTH legs, not each leg's own independent last bar). One
trade per pair per day (first qualifying entry only), matching this
project's standing "first qualifying signal per day" convention.

EQUAL RUPEE-WEIGHTED positions: each pair-trade's return is the equal-
weighted AVERAGE of the two legs' own signed %% returns (mathematically
equivalent to sizing both legs to the same rupee notional at entry,
without needing to track actual share counts).

STEP 4 -- COSTS: COST_PCT_PER_LEG (0.075%%, same assumption as every
other cost-aware script in this project) applied to EACH leg, so 0.15%%
total per pair-trade -- reported gross and net.

Run: python -m examples.run_pairs_trading
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from config.settings import PROJECT_ROOT
from data import storage
from detectors.liquidity_analysis import analyze_liquidity
from examples.run_crash_detection import MIN_SHARPE_BAR, MIN_WINRATE_BAR, _day_stats, _verdict

CORRELATION_THRESHOLD = 0.70
COINTEGRATION_PVALUE_THRESHOLD = 0.05
MIN_DAILY_OBS = 10  # minimum overlapping cached days before a pair is even attempted
ZSCORE_ROLLING_DAYS = 20
ENTRY_ZSCORE = 2.0
EXIT_ZSCORE = 0.5
EOD_SQUARE_OFF_TIME = "15:15"
COST_PCT_PER_LEG = 0.075
IN_SAMPLE_FRACTION = 0.60
FROZEN_RUN_CHECK_BARS = 10
DATA_GAP_EARLY_EXIT_MINUTES = 30


@dataclass
class Pair:
    symbol_a: str
    symbol_b: str
    sector: str
    correlation: float
    coint_pvalue: float
    hedge_ratio: float
    n_in_sample_obs: int
    in_sample_days: list
    out_sample_days: list


@dataclass
class PairTrade:
    symbol_a: str
    symbol_b: str
    sector: str
    day: date
    period: str  # "in_sample" | "out_sample"
    direction: str  # "short_a_long_b" | "long_a_short_b"
    entry_zscore: float
    entry_time: pd.Timestamp
    entry_price_a: float
    entry_price_b: float
    exit_time: pd.Timestamp
    exit_price_a: float
    exit_price_b: float
    exit_reason: str  # "reverted" | "eod_square_off"
    avg_traded_value_a: float
    avg_traded_value_b: float
    frozen_a: bool
    frozen_b: bool


def _frozen(bar: pd.Series) -> bool:
    return float(bar["open"]) == float(bar["high"]) == float(bar["low"]) == float(bar["close"])


def _daily_close_series(df: pd.DataFrame) -> pd.Series:
    return df.groupby(df.index.date)["close"].last()


def _load_liquid_sector_universe() -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    import json
    mapping_path = PROJECT_ROOT / "config" / "sector_mapping.json"
    sector_mapping = json.loads(mapping_path.read_text())

    symbol_data: dict[str, pd.DataFrame] = {}
    liquid_mapping: dict[str, str] = {}
    for symbol, sector in sector_mapping.items():
        df = storage.load(symbol)
        if df.empty:
            continue
        last_day = sorted(set(df.index.date))[-1]
        reading = analyze_liquidity(df[df.index.date == last_day])
        if reading is not None and reading.is_liquid:
            symbol_data[symbol] = df
            liquid_mapping[symbol] = sector
    return liquid_mapping, symbol_data


def _find_qualifying_pairs(liquid_mapping: dict[str, str], symbol_data: dict[str, pd.DataFrame]) -> list[Pair]:
    by_sector: dict[str, list[str]] = {}
    for symbol, sector in liquid_mapping.items():
        by_sector.setdefault(sector, []).append(symbol)

    daily_closes = {s: _daily_close_series(df) for s, df in symbol_data.items()}

    pairs: list[Pair] = []
    tested = 0
    for sector, members in by_sector.items():
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                common_days = sorted(set(daily_closes[a].index) & set(daily_closes[b].index))
                if len(common_days) < MIN_DAILY_OBS:
                    continue
                tested += 1

                split_idx = int(len(common_days) * IN_SAMPLE_FRACTION)
                in_sample_days = common_days[:split_idx]
                out_sample_days = common_days[split_idx:]
                if len(in_sample_days) < 5 or not out_sample_days:
                    continue

                close_a = daily_closes[a].loc[in_sample_days]
                close_b = daily_closes[b].loc[in_sample_days]
                ret_a = close_a.pct_change().dropna()
                ret_b = close_b.pct_change().dropna()
                common_ret_days = ret_a.index.intersection(ret_b.index)
                if len(common_ret_days) < 4:
                    continue
                correlation = float(np.corrcoef(ret_a.loc[common_ret_days], ret_b.loc[common_ret_days])[0, 1])
                if not (correlation >= CORRELATION_THRESHOLD):
                    continue

                log_a = np.log(close_a.values)
                log_b = np.log(close_b.values)
                try:
                    _, pvalue, _ = coint(log_a, log_b)
                except Exception:
                    continue
                if pvalue >= COINTEGRATION_PVALUE_THRESHOLD:
                    continue

                model = sm.OLS(log_a, sm.add_constant(log_b)).fit()
                hedge_ratio = float(model.params[1])

                pairs.append(Pair(
                    symbol_a=a, symbol_b=b, sector=sector, correlation=correlation, coint_pvalue=float(pvalue),
                    hedge_ratio=hedge_ratio, n_in_sample_obs=len(in_sample_days),
                    in_sample_days=in_sample_days, out_sample_days=out_sample_days,
                ))

    print(f"Sector pairs attempted (>= {MIN_DAILY_OBS} overlapping cached days): {tested}")
    return pairs


def _simulate_pair_trades(pair: Pair, symbol_data: dict[str, pd.DataFrame]) -> list[PairTrade]:
    df_a, df_b = symbol_data[pair.symbol_a], symbol_data[pair.symbol_b]
    daily_a, daily_b = _daily_close_series(df_a), _daily_close_series(df_b)
    all_days = pair.in_sample_days + pair.out_sample_days
    in_sample_set = set(pair.in_sample_days)

    # Full daily spread series (all common days, chronological) for the
    # rolling 20-day baseline -- see module docstring for why this is
    # allowed to look back across the in-sample/out-of-sample boundary.
    spread_by_day: dict[date, float] = {}
    for day in all_days:
        if day in daily_a.index and day in daily_b.index:
            spread_by_day[day] = float(np.log(daily_a.loc[day]) - pair.hedge_ratio * np.log(daily_b.loc[day]))
    ordered_days = sorted(spread_by_day.keys())

    trades: list[PairTrade] = []
    for idx, day in enumerate(ordered_days):
        if idx < ZSCORE_ROLLING_DAYS:
            continue
        trailing = [spread_by_day[d] for d in ordered_days[idx - ZSCORE_ROLLING_DAYS:idx]]
        baseline_mean, baseline_std = float(np.mean(trailing)), float(np.std(trailing))
        if baseline_std == 0:
            continue

        day_idx_in_all = all_days.index(day) if day in all_days else None
        if day_idx_in_all is None or day_idx_in_all == 0:
            continue
        prev_day = all_days[day_idx_in_all - 1]
        prev_a = df_a[df_a.index.date == prev_day]
        prev_b = df_b[df_b.index.date == prev_day]
        if prev_a.empty or prev_b.empty:
            continue
        liq_a, liq_b = analyze_liquidity(prev_a), analyze_liquidity(prev_b)
        if liq_a is None or not liq_a.is_liquid or liq_b is None or not liq_b.is_liquid:
            continue

        day_a = df_a[df_a.index.date == day]
        day_b = df_b[df_b.index.date == day]
        common_times = day_a.index.intersection(day_b.index).sort_values()
        if len(common_times) < 2:
            continue

        period = "in_sample" if day in in_sample_set else "out_sample"
        eod_ts = pd.Timestamp.combine(day, pd.Timestamp(EOD_SQUARE_OFF_TIME).time()).tz_localize(common_times.tz)
        eod_candidates = common_times[common_times <= eod_ts]
        if eod_candidates.empty:
            continue
        eod_time = eod_candidates[-1]

        # Find entry: first common bar whose CLOSE-based z-score fires.
        entry_time = None
        direction = None
        entry_z = None
        for t in common_times:
            price_a, price_b = float(day_a.loc[t, "close"]), float(day_b.loc[t, "close"])
            spread = np.log(price_a) - pair.hedge_ratio * np.log(price_b)
            z = (spread - baseline_mean) / baseline_std
            if abs(z) >= ENTRY_ZSCORE:
                after = common_times[common_times > t]
                if after.empty:
                    break
                entry_time = after[0]
                direction = "short_a_long_b" if z > 0 else "long_a_short_b"
                entry_z = float(z)
                break
        if entry_time is None or entry_time > eod_time:
            continue

        entry_price_a = float(day_a.loc[entry_time, "open"])
        entry_price_b = float(day_b.loc[entry_time, "open"])

        # Find exit: first common bar (strictly after entry) whose CLOSE-based
        # z-score reverts within EXIT_ZSCORE, else EOD.
        exit_time, exit_reason = None, "eod_square_off"
        after_entry = common_times[(common_times > entry_time) & (common_times <= eod_time)]
        for t in after_entry:
            price_a, price_b = float(day_a.loc[t, "close"]), float(day_b.loc[t, "close"])
            spread = np.log(price_a) - pair.hedge_ratio * np.log(price_b)
            z = (spread - baseline_mean) / baseline_std
            if abs(z) <= EXIT_ZSCORE:
                after = common_times[(common_times > t) & (common_times <= eod_time)]
                if not after.empty:
                    exit_time, exit_reason = after[0], "reverted"
                break
        if exit_time is None:
            exit_time = eod_time

        exit_price_a = float(day_a.loc[exit_time, "open" if exit_reason == "reverted" else "close"])
        exit_price_b = float(day_b.loc[exit_time, "open" if exit_reason == "reverted" else "close"])
        if exit_time <= entry_time:
            continue

        entry_bar_a, entry_bar_b = day_a.loc[entry_time], day_b.loc[entry_time]
        trades.append(PairTrade(
            symbol_a=pair.symbol_a, symbol_b=pair.symbol_b, sector=pair.sector, day=day, period=period,
            direction=direction, entry_zscore=entry_z, entry_time=entry_time,
            entry_price_a=entry_price_a, entry_price_b=entry_price_b,
            exit_time=exit_time, exit_price_a=exit_price_a, exit_price_b=exit_price_b, exit_reason=exit_reason,
            avg_traded_value_a=liq_a.avg_traded_value, avg_traded_value_b=liq_b.avg_traded_value,
            frozen_a=_frozen(entry_bar_a), frozen_b=_frozen(entry_bar_b),
        ))

    return trades


def _pair_trade_return_pct(t: PairTrade, cost_pct_per_leg: float = 0.0) -> float:
    sign_a = -1.0 if t.direction == "short_a_long_b" else 1.0
    sign_b = 1.0 if t.direction == "short_a_long_b" else -1.0
    leg_a = sign_a * (t.exit_price_a - t.entry_price_a) / t.entry_price_a * 100.0 - cost_pct_per_leg
    leg_b = sign_b * (t.exit_price_b - t.entry_price_b) / t.entry_price_b * 100.0 - cost_pct_per_leg
    return 0.5 * (leg_a + leg_b)


def _max_drawdown_pct(day_returns: np.ndarray) -> float:
    if len(day_returns) == 0:
        return 0.0
    equity = np.cumsum(day_returns)
    running_max = np.maximum.accumulate(equity)
    return float((running_max - equity).max())


def _report_period(label: str, trades: list[PairTrade], cost_pct_per_leg: float) -> tuple[float, float, int]:
    if not trades:
        print(f"\n--- {label} ---\nNo qualifying pair-trades.")
        return 0.0, 0.0, 0
    gross = np.array([_pair_trade_return_pct(t, 0.0) for t in trades])
    net = np.array([_pair_trade_return_pct(t, cost_pct_per_leg) for t in trades])
    by_day_gross: dict[date, list[float]] = {}
    by_day_net: dict[date, list[float]] = {}
    for t, g, n in zip(trades, gross, net):
        by_day_gross.setdefault(t.day, []).append(g)
        by_day_net.setdefault(t.day, []).append(n)
    day_gross = np.array([float(np.mean(v)) for v in by_day_gross.values()])
    day_net = np.array([float(np.mean(v)) for v in by_day_net.values()])
    g_win, g_mean, g_std, g_sharpe = _day_stats(day_gross)
    n_win, n_mean, n_std, n_sharpe = _day_stats(day_net)
    g_dd, n_dd = _max_drawdown_pct(day_gross), _max_drawdown_pct(day_net)

    print(f"\n--- {label} ---")
    print(f"Pair-trades: {len(trades)}  |  Days used: {len(by_day_gross)}")
    print(f"Position-level -- GROSS win rate: {(gross > 0).mean():.1%}  avg: {gross.mean():+.3f}%   "
          f"NET win rate: {(net > 0).mean():.1%}  avg: {net.mean():+.3f}%")
    print(f"Day-level GROSS -- win rate: {g_win:.1%}  avg daily return: {g_mean:+.3f}%  std: {g_std:.3f}%  "
          f"Sharpe: {g_sharpe:.2f}  max drawdown: {g_dd:.3f}%   [{_verdict(g_sharpe, g_win)}]")
    print(f"Day-level NET   -- win rate: {n_win:.1%}  avg daily return: {n_mean:+.3f}%  std: {n_std:.3f}%  "
          f"Sharpe: {n_sharpe:.2f}  max drawdown: {n_dd:.3f}%   [{_verdict(n_sharpe, n_win)}]")
    return n_sharpe, n_win, len(by_day_gross)


def _data_gap_flag(t: PairTrade) -> str | None:
    if t.exit_reason != "eod_square_off":
        return None
    eod_minutes = 15 * 60 + 15
    exit_minutes = t.exit_time.hour * 60 + t.exit_time.minute
    if eod_minutes - exit_minutes >= DATA_GAP_EARLY_EXIT_MINUTES:
        return (f"POSSIBLE DATA GAP: EOD exit found at {t.exit_time.strftime('%H:%M')}, "
                f"{eod_minutes - exit_minutes} min before the 15:15 target -- treat with caution")
    return None


def _spot_check(trades: list[PairTrade], n: int) -> None:
    print(f"\n--- SPOT-CHECK: {min(n, len(trades))} real pair-trades, largest |entry z-score| first ---")
    for t in sorted(trades, key=lambda t: abs(t.entry_zscore), reverse=True)[:n]:
        print(f"  {t.symbol_a}/{t.symbol_b} ({t.sector})  {t.day}  {t.period}  {t.direction}  "
              f"entry_z={t.entry_zscore:+.2f}")
        print(f"    {t.symbol_a}: entry={t.entry_price_a:.2f} exit={t.exit_price_a:.2f}  "
              f"liquidity=Rs.{t.avg_traded_value_a:,.0f}/min  {'FROZEN' if t.frozen_a else 'ok'}")
        print(f"    {t.symbol_b}: entry={t.entry_price_b:.2f} exit={t.exit_price_b:.2f}  "
              f"liquidity=Rs.{t.avg_traded_value_b:,.0f}/min  {'FROZEN' if t.frozen_b else 'ok'}")
        print(f"    entry={t.entry_time.strftime('%H:%M')}  exit={t.exit_time.strftime('%H:%M')} ({t.exit_reason})  "
              f"pair_return_gross={_pair_trade_return_pct(t):+.3f}%")
        gap = _data_gap_flag(t)
        if gap:
            print(f"    {gap}")


def main() -> None:
    print("=" * 70)
    print("STEP 1: liquid sector universe + pair selection (correlation + cointegration, IN-SAMPLE ONLY)")
    print("=" * 70)
    t0 = time.monotonic()
    liquid_mapping, symbol_data = _load_liquid_sector_universe()
    print(f"Sector-mapped universe: {len(symbol_data)} liquid symbols "
          f"(of {len(set(liquid_mapping) | set(symbol_data))} candidates) in {time.monotonic() - t0:.1f}s")

    pairs = _find_qualifying_pairs(liquid_mapping, symbol_data)
    print(f"\nQualifying pairs (correlation >= {CORRELATION_THRESHOLD}, cointegration p < "
          f"{COINTEGRATION_PVALUE_THRESHOLD}, in-sample only): {len(pairs)}")
    by_sector_count: dict[str, int] = {}
    for p in pairs:
        by_sector_count[p.sector] = by_sector_count.get(p.sector, 0) + 1
        print(f"  {p.symbol_a}/{p.symbol_b} ({p.sector})  corr={p.correlation:.3f}  "
              f"coint_p={p.coint_pvalue:.4f}  hedge_ratio={p.hedge_ratio:.3f}  "
              f"in_sample_days={p.n_in_sample_obs} (out_sample_days={len(p.out_sample_days)})")
    print(f"\nSectors represented: {by_sector_count}")
    print("=" * 70)
    print("SAMPLE-SIZE CAVEAT: correlation/cointegration/hedge-ratio are all fit on ~10-20 daily "
          "observations (60% of Upstox's ~29-33 day rolling cache). Cointegration tests conventionally "
          "want 50-100+ observations -- treat ANY 'cointegrated' verdict here as preliminary at best.")
    print("=" * 70)

    if not pairs:
        print("\nNo qualifying pairs found -- nothing further to test.")
        return

    print("\n" + "=" * 70)
    print("STEP 2/3: intraday backtest (walk-forward: in-sample vs out-of-sample)")
    print("=" * 70)
    all_trades: list[PairTrade] = []
    for pair in pairs:
        trades = _simulate_pair_trades(pair, symbol_data)
        all_trades.extend(trades)
        print(f"  {pair.symbol_a}/{pair.symbol_b}: {len(trades)} pair-trades "
              f"({sum(1 for t in trades if t.period == 'in_sample')} in-sample, "
              f"{sum(1 for t in trades if t.period == 'out_sample')} out-of-sample)")

    in_sample_trades = [t for t in all_trades if t.period == "in_sample"]
    out_sample_trades = [t for t in all_trades if t.period == "out_sample"]

    cost_total = 0.0
    _report_period("IN-SAMPLE (reference only)", in_sample_trades, cost_total)
    print("\n" + "-" * 70)
    print(f"STEP 4: costs -- {COST_PCT_PER_LEG:.3f}% per leg, {COST_PCT_PER_LEG*2:.3f}% total per pair-trade")
    print("-" * 70)
    _report_period("IN-SAMPLE, gross vs net", in_sample_trades, COST_PCT_PER_LEG)
    out_sharpe, out_win, out_days = _report_period("OUT-OF-SAMPLE (the real test), gross vs net", out_sample_trades, COST_PCT_PER_LEG)

    print("\n" + "=" * 70)
    print("STEP 5: spot-check (BEFORE final verdict)")
    print("=" * 70)
    spot_pool = out_sample_trades if out_sample_trades else all_trades
    if spot_pool:
        _spot_check(spot_pool, min(10, len(spot_pool)))
        n_frozen = sum(1 for t in spot_pool if t.frozen_a or t.frozen_b)
        n_gap = sum(1 for t in spot_pool if _data_gap_flag(t) is not None)
        print(f"\nAcross {len(spot_pool)} trades: {n_frozen} with a frozen leg, {n_gap} likely data-gap early exits.")
    else:
        print("\nNo trades available to spot-check.")

    print("\n" + "=" * 70)
    print("STEP 6: HONEST VERDICT (pre-registered bars: Sharpe>=0.5 AND win rate>=60%, NET of costs)")
    print("=" * 70)
    print(f"Out-of-sample: {out_days} days, {len(out_sample_trades)} pair-trades, "
          f"NET Sharpe={out_sharpe:.2f}, NET win rate={out_win:.1%}  -->  {_verdict(out_sharpe, out_win)}")
    print(f"\nQualifying pairs: {len(pairs)} across {len(by_sector_count)} sectors. "
          f"In-sample days used for fitting: {[p.n_in_sample_obs for p in pairs]}")
    print("REMINDER: only the OUT-OF-SAMPLE result above is the real test -- in-sample numbers reported "
          "earlier are a reference/sanity-check only and should NOT be read as evidence of an edge.")


if __name__ == "__main__":
    main()
