"""
Regime-conditional analysis: tags each already-backtested trade (reused
from the raw full-universe results already on disk from earlier work --
NOT a new backtest) with the whole-market breadth reading (% advancing,
detectors/market_breadth.py) at/near that trade's entry time, then splits
each strategy's trades into "high breadth" vs "low breadth" regimes and
compares win rate / avg R-multiple between them.

Trade sources reused (raw JSONL already on disk, full 2373-symbol
universe, no new backtest run):
  - logs/orb_partial_results.jsonl          -> opening_range_breakout
  - logs/smc_vwap_partial_results.jsonl     -> smart_money, vwap_reclaim
  - logs/trailing_stop_partial_results.jsonl, variant=="baseline_1_2"
                                             -> validated_breakout (the
    standard, non-trailing 1:2 R:R version -- the other two variants in
    that file are a separate trailing-stop experiment, not this
    strategy's registered default, so they're excluded here)
trend_continuation, reversal_detection, and fibonacci_retracement have no
full-universe raw trade file on disk (never run at that scale) and are
therefore NOT included in this analysis -- reported explicitly, not
silently.

BREADTH-AT-ENTRY-TIME COMPUTATION: computing detectors.market_breadth.
compute_market_breadth() fresh for every individual trade's exact entry
minute would mean tens of thousands of full-universe passes -- not
tractable. Instead, entry times are rounded DOWN to the nearest 15-minute
bucket (09:15, 09:30, ..., 15:15) and breadth is computed ONCE per
distinct (day, bucket) pair actually needed, then reused for every trade
that shares it. This is a reasonable approximation: breadth doesn't swing
wildly minute-to-minute, and this is a REGIME classification, not a
precise point-in-time reading.

HONEST SAMPLE-SIZE CAVEAT (restated in the printed report, not just
here): breadth is computed from the same ~29-30 day cached universe as
everything else in this project -- a SMALL number of distinct trading
DAYS underlies every trade here, regardless of how many total trades
exist. Two strategies both firing thousands of trades doesn't mean two
independent samples of market regimes; it means the same ~26 days'
regimes, sliced more finely.

Run: python -m examples.analyze_breadth_regime
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from backtest.engine import Trade
from backtest.strategy_base import SignalDirection
from config.settings import PROJECT_ROOT, STORAGE_ROOT
from data import storage
from detectors.market_breadth import compute_market_breadth

BUCKET_MINUTES = 15
BREADTH_THRESHOLD_PCT = 60.0  # see report for whether the data supports this specific split


def _to_trade(d: dict) -> Trade:
    return Trade(
        symbol=d["symbol"], direction=SignalDirection(d["direction"]),
        entry_time=pd.Timestamp(d["entry_time"]), entry_price=d["entry_price"],
        stop_loss=d["stop_loss"], target=d["target"], reasoning=d.get("reasoning", []),
        confidence=d.get("confidence", 0.5), strategy_name=d["strategy_name"],
        exit_time=pd.Timestamp(d["exit_time"]) if d.get("exit_time") else None,
        exit_price=d.get("exit_price"), exit_reason=d.get("exit_reason"),
        initial_stop_loss=d.get("initial_stop_loss"),
    )


def load_trades() -> dict[str, list[Trade]]:
    by_strategy: dict[str, list[Trade]] = defaultdict(list)

    orb_path = PROJECT_ROOT / "logs" / "orb_partial_results.jsonl"
    with open(orb_path) as f:
        for line in f:
            row = json.loads(line)
            for t in row["trades"]:
                by_strategy[t["strategy_name"]].append(_to_trade(t))

    smc_vwap_path = PROJECT_ROOT / "logs" / "smc_vwap_partial_results.jsonl"
    with open(smc_vwap_path) as f:
        for line in f:
            row = json.loads(line)
            for t in row["trades"]:
                by_strategy[t["strategy_name"]].append(_to_trade(t))

    validated_path = PROJECT_ROOT / "logs" / "trailing_stop_partial_results.jsonl"
    with open(validated_path) as f:
        for line in f:
            row = json.loads(line)
            if row["variant"] != "baseline_1_2":
                continue
            for t in row["trades"]:
                by_strategy[t["strategy_name"]].append(_to_trade(t))

    return dict(by_strategy)


def bucket_time(ts: pd.Timestamp) -> pd.Timestamp:
    minute_of_day = ts.hour * 60 + ts.minute
    bucket_minute = (minute_of_day // BUCKET_MINUTES) * BUCKET_MINUTES
    return ts.replace(hour=bucket_minute // 60, minute=bucket_minute % 60, second=0, microsecond=0)


def main() -> None:
    print("Loading full cached universe for breadth computation...")
    symbols = sorted(p.stem.replace("_1min", "") for p in STORAGE_ROOT.glob("*_1min.parquet"))
    symbol_data = {s: df for s in symbols if not (df := storage.load(s)).empty}
    print(f"Loaded {len(symbol_data)} symbols.\n")

    print("Loading existing full-universe trade results (no new backtest)...")
    by_strategy = load_trades()
    for name, trades in by_strategy.items():
        print(f"  {name}: {len(trades)} trades")
    print()

    all_trades = [(name, t) for name, trades in by_strategy.items() for t in trades]
    unique_buckets = sorted({bucket_time(t.entry_time) for _, t in all_trades})
    print(f"Distinct (day, {BUCKET_MINUTES}-min bucket) combinations needed: {len(unique_buckets)}")
    distinct_days = sorted({b.date() for b in unique_buckets})
    print(f"Distinct trading days covered by these trades: {len(distinct_days)}  ({distinct_days[0]} .. {distinct_days[-1]})\n")

    print("Computing breadth once per bucket...")
    breadth_by_bucket: dict[pd.Timestamp, float] = {}
    for i, b in enumerate(unique_buckets, 1):
        reading = compute_market_breadth(symbol_data, as_of=b)
        breadth_by_bucket[b] = reading.pct_advancing if reading is not None else float("nan")
        if i % 100 == 0:
            print(f"  {i}/{len(unique_buckets)}...", flush=True)
    print("Done.\n")

    breadth_values = np.array([v for v in breadth_by_bucket.values() if not np.isnan(v)])
    print("=" * 70)
    print("BREADTH DISTRIBUTION ACROSS ALL BUCKETS ACTUALLY USED")
    print("=" * 70)
    print(f"n={len(breadth_values)}  mean={breadth_values.mean():.1f}%  median={np.median(breadth_values):.1f}%  "
          f"std={breadth_values.std():.1f}%  min={breadth_values.min():.1f}%  max={breadth_values.max():.1f}%")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  p{pct}: {np.percentile(breadth_values, pct):.1f}%")
    frac_above_60 = (breadth_values >= 60.0).mean()
    print(f"\nFraction of buckets with breadth >= 60%: {frac_above_60:.1%}")
    print(f"Fraction of buckets with breadth >= median ({np.median(breadth_values):.1f}%): ~50% by construction")
    print()

    print("=" * 70)
    print(f"SAMPLE-SIZE CAVEAT: breadth regimes above are computed from {len(distinct_days)} distinct "
          f"trading days of cached data. This is a SMALL sample; treat all findings here as "
          f"preliminary, not validated conclusions.")
    print("=" * 70)
    print()

    for threshold_label, threshold in [
        (f"{BREADTH_THRESHOLD_PCT:.0f}% (task-suggested default)", BREADTH_THRESHOLD_PCT),
        (f"{np.median(breadth_values):.1f}% (data median, balanced split)", float(np.median(breadth_values))),
    ]:
        print("#" * 70)
        print(f"SPLIT: breadth >= {threshold_label}  vs  breadth < {threshold_label}")
        print("#" * 70)
        for name, trades in by_strategy.items():
            high, low = [], []
            for t in trades:
                b = breadth_by_bucket.get(bucket_time(t.entry_time))
                if b is None or np.isnan(b):
                    continue
                (high if b >= threshold else low).append(t)

            def _stats(group: list[Trade]) -> tuple[int, float, float]:
                if not group:
                    return 0, float("nan"), float("nan")
                wins = sum(1 for t in group if t.pnl_points > 0)
                win_rate = wins / len(group)
                avg_r = float(np.mean([t.r_multiple for t in group]))
                return len(group), win_rate, avg_r

            n_high, wr_high, r_high = _stats(high)
            n_low, wr_low, r_low = _stats(low)

            print(f"\n{name}:")
            print(f"  High breadth: n={n_high:5d}  win_rate={wr_high:.1%}  avg_R={r_high:+.3f}")
            print(f"  Low  breadth: n={n_low:5d}  win_rate={wr_low:.1%}  avg_R={r_low:+.3f}")
            if n_high >= 30 and n_low >= 30:
                wr_diff = wr_high - wr_low
                r_diff = r_high - r_low
                meaningful = abs(wr_diff) >= 0.05 or abs(r_diff) >= 0.10
                verdict = (
                    f"MEANINGFUL DIFFERENCE (win_rate diff {wr_diff:+.1%}, avg_R diff {r_diff:+.3f})"
                    if meaningful else
                    f"NO meaningful difference (win_rate diff {wr_diff:+.1%}, avg_R diff {r_diff:+.3f} -- within noise)"
                )
            else:
                verdict = f"INSUFFICIENT SAMPLE in one or both groups (n_high={n_high}, n_low={n_low}) -- no verdict"
            print(f"  VERDICT: {verdict}")
        print()


if __name__ == "__main__":
    main()
