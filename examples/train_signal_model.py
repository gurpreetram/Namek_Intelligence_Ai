"""
Trains and honestly evaluates a simple logistic-regression model that
learns which combinations of this project's existing detector-derived
features predict a WINNING trade (R-multiple > 0), pooling Smart Money
Concepts (order-block retest) + Opening Range Breakout trades -- the two
richest full-universe trade sets currently persisted (see
examples/run_smc_vwap_universe.py and examples/run_orb_universe.py).

SCOPE: this is explicitly "does combining our existing hand-computed
features predict outcomes better than the strategies' own hardcoded
entry rules," NOT a new trading strategy. There is no Strategy subclass,
no backtest-engine integration, no live signal generation here. If this
experiment finds a real, honest edge, that's the START of a case for
building a real strategy around it, not the strategy itself.

CRITICAL: time-ordered train/test split, not random. A random shuffle
split would let the model train on data chronologically AFTER some test
examples -- a lookahead leak at the DATASET level (even though each
trade's own FEATURES are already no-lookahead-safe per
detectors/feature_extraction.py) -- producing an optimistic test score
that wouldn't hold on genuinely future trades. Same discipline as
backtest/walk_forward.py, applied to a model instead of a whole strategy.

Run: python -m examples.train_signal_model
"""
from __future__ import annotations

import json
import multiprocessing
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from backtest.engine import Trade
from backtest.strategy_base import SignalDirection
from config.settings import PROJECT_ROOT
from data import storage
from detectors.feature_extraction import FEATURE_NAMES, extract_features_for_trade

SMC_VWAP_PATH = PROJECT_ROOT / "logs" / "smc_vwap_partial_results.jsonl"
ORB_PATH = PROJECT_ROOT / "logs" / "orb_partial_results.jsonl"
DATASET_CACHE_PATH = PROJECT_ROOT / "logs" / "signal_model_dataset.parquet"
BENCHMARK_SYMBOL = "NIFTYBEES"

# Same "is this sample big enough to say anything" bar used everywhere
# else in this project's reporting (risk/probability_calibration.py's
# min_reliable_sample=30, portfolio/analytics.py's =20) -- 30 here to
# match the calibration module specifically, since this experiment is
# closer in spirit to "is this probability estimate trustworthy" than to
# a simple per-symbol/per-strategy breakdown.
MIN_RELIABLE_SAMPLE = 30
TOP_PCT = 0.20


def _trade_from_dict(d: dict) -> Trade:
    return Trade(
        symbol=d["symbol"], direction=SignalDirection(d["direction"]),
        entry_time=pd.Timestamp(d["entry_time"]), entry_price=d["entry_price"],
        stop_loss=d["stop_loss"], target=d["target"], reasoning=d["reasoning"],
        confidence=d["confidence"], strategy_name=d["strategy_name"],
        exit_time=pd.Timestamp(d["exit_time"]) if d["exit_time"] is not None else None,
        exit_price=d["exit_price"], exit_reason=d["exit_reason"],
        initial_stop_loss=d.get("initial_stop_loss"), peak_price=d.get("peak_price"),
        breakeven_moved=d.get("breakeven_moved", False), trailing_active=d.get("trailing_active", False),
    )


def _load_smc_trades() -> list[Trade]:
    trades = []
    with open(SMC_VWAP_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["strategy"] != "smart_money":
                continue
            trades.extend(_trade_from_dict(t) for t in rec["trades"])
    return trades


def _load_orb_trades() -> list[Trade]:
    trades = []
    with open(ORB_PATH) as f:
        for line in f:
            rec = json.loads(line)
            trades.extend(_trade_from_dict(t) for t in rec["trades"])
    return trades


def _extract_for_symbol(args: tuple[str, list[Trade]]) -> list[dict]:
    symbol, trades = args
    df = storage.load(symbol)
    if df.empty:
        return []
    benchmark_df = storage.load(BENCHMARK_SYMBOL)
    rows = []
    for trade in trades:
        feat = extract_features_for_trade(trade, df, benchmark_df)
        if feat is None:
            continue
        row = dict(feat.features)
        row["symbol"] = feat.symbol
        row["entry_time"] = feat.entry_time
        row["strategy_name"] = feat.strategy_name
        row["label_win"] = feat.label_win
        row["r_multiple"] = feat.r_multiple
        rows.append(row)
    return rows


def build_dataset() -> pd.DataFrame:
    smc_trades = _load_smc_trades()
    orb_trades = _load_orb_trades()
    print(f"Loaded {len(smc_trades)} smart_money trades, {len(orb_trades)} opening_range_breakout trades")

    all_trades = smc_trades + orb_trades
    by_symbol: dict[str, list[Trade]] = {}
    for t in all_trades:
        by_symbol.setdefault(t.symbol, []).append(t)

    cores = os.cpu_count() or 1
    tasks = list(by_symbol.items())
    all_rows = []
    with multiprocessing.Pool(processes=cores) as pool:
        for rows in pool.imap_unordered(_extract_for_symbol, tasks):
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"Extracted features for {len(df)}/{len(all_trades)} trades "
          f"({len(all_trades) - len(df)} skipped, insufficient history before entry)")
    return df


def main() -> None:
    if DATASET_CACHE_PATH.exists():
        print(f"Loading cached dataset from {DATASET_CACHE_PATH}")
        df = pd.read_parquet(DATASET_CACHE_PATH)
    else:
        df = build_dataset()
        df.to_parquet(DATASET_CACHE_PATH)
        print(f"Cached dataset to {DATASET_CACHE_PATH}")

    df = df.sort_values("entry_time").reset_index(drop=True)

    # Categorical "which strategy fired" -- one-hot with one category
    # dropped (standard practice to avoid perfect collinearity with the
    # intercept in a linear model). is_smart_money=0 implicitly means
    # opening_range_breakout, since these are the only two strategies here.
    df["is_smart_money"] = (df["strategy_name"] == "smart_money").astype(float)
    feature_cols = FEATURE_NAMES + ["is_smart_money"]

    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"\nTotal dataset: {len(df)} trades")
    print(f"Train: {len(train_df)} trades ({train_df['entry_time'].min()} to {train_df['entry_time'].max()})")
    print(f"Test:  {len(test_df)} trades ({test_df['entry_time'].min()} to {test_df['entry_time'].max()})")

    X_train, y_train = train_df[feature_cols].values, train_df["label_win"].values
    X_test, y_test = test_df[feature_cols].values, test_df["label_win"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    test_probs = model.predict_proba(X_test_scaled)[:, 1]

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    test_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))

    print()
    print("=" * 70)
    print("MODEL COEFFICIENTS (standardized features -- magnitude ~ importance)")
    print("=" * 70)
    coefs = sorted(zip(feature_cols, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coefs:
        print(f"  {name:22s} {coef:+.4f}")
    print(f"  {'intercept':22s} {model.intercept_[0]:+.4f}")

    print()
    print("=" * 70)
    print("HONEST EVALUATION (TEST SET ONLY)")
    print("=" * 70)
    print(f"Train AUC: {train_auc:.3f}  (reference only, NOT the headline number)")
    print(f"Test AUC:  {test_auc:.3f}")
    print(f"Test accuracy: {test_acc:.1%}")
    print(f"Test set size: {len(test_df)} trades")

    overall_test_win_rate = float(y_test.mean())
    overall_test_avg_r = float(test_df["r_multiple"].mean())
    print(f"\nBaseline (unfiltered test set): win rate {overall_test_win_rate:.1%}, avg R {overall_test_avg_r:+.3f}")
    print(f"Coin-flip baseline: 50.0% win rate")

    n_top = max(1, int(len(test_df) * TOP_PCT))
    top_idx = np.argsort(-test_probs)[:n_top]
    top_win_rate = float(y_test[top_idx].mean())
    top_avg_r = float(test_df["r_multiple"].values[top_idx].mean())
    reliable = n_top >= MIN_RELIABLE_SAMPLE
    print(f"\nTop {TOP_PCT:.0%} most-confident predictions "
          f"({n_top} trades{'' if reliable else f' -- TOO FEW, need >= {MIN_RELIABLE_SAMPLE}'}):")
    print(f"  Win rate: {top_win_rate:.1%} (vs {overall_test_win_rate:.1%} unfiltered, "
          f"{(top_win_rate - overall_test_win_rate) * 100:+.1f}pp)")
    print(f"  Avg R:    {top_avg_r:+.3f} (vs {overall_test_avg_r:+.3f} unfiltered)")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    if test_auc < 0.55:
        print("Test AUC is close to 0.5 (coin flip) -- these features do NOT meaningfully")
        print("distinguish winning from losing trades at this scale. No real edge found.")
    elif not reliable:
        print(f"Model shows some discrimination (AUC {test_auc:.3f}), but the top-{TOP_PCT:.0%} subset "
              f"is only {n_top} trades -- too few to trust the win-rate/avg-R reading either way.")
    elif top_win_rate <= overall_test_win_rate and top_avg_r <= overall_test_avg_r:
        print(f"Model shows some discrimination (AUC {test_auc:.3f}), but filtering to its most confident "
              f"predictions does NOT actually improve realized win rate or avg R-multiple. Marginal at "
              f"best -- do not treat as a validated edge.")
    else:
        print(f"Model's top-confidence subset ({n_top} trades, sample-size reliable) shows a real "
              f"improvement over the unfiltered baseline. Worth further investigation -- still not "
              f"validated as a tradeable edge without confirmation beyond this one test split.")


if __name__ == "__main__":
    main()
