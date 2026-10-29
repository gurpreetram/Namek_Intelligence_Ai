"""
Investigates WHY the ML signal model's win-rate improvement in the top-20%-
confidence picks (examples/validate_signal_model_stability.py) doesn't
reliably translate into an avg R-multiple improvement (only 3/5 folds).

HYPOTHESIS being tested: the win-rate gain comes disproportionately from
small-margin EOD square-off "wins" (R-multiple slightly above zero) rather
than genuine target hits (fixed +2R) -- if so, win rate rises without avg
R following, since these wins contribute little value, while losses stay
concentrated at the full -1R stop-loss level (or worse via an adverse
EOD square-off).

exit_reason isn't in the cached feature dataset (logs/signal_model_dataset.parquet
only has symbol/entry_time/strategy_name/features/label_win/r_multiple) --
this script joins it back from the original trade records
(logs/smc_vwap_partial_results.jsonl, logs/orb_partial_results.jsonl) by
matching (symbol, entry_time, strategy_name), rather than re-running the
expensive feature extraction just to add one field.

Re-derives the EXACT same 5 TimeSeriesSplit folds and top-20% selections
as validate_signal_model_stability.py (same data, same code, deterministic
LogisticRegression fit) so this analysis is directly about THOSE folds,
not a new experiment.

Run: python -m examples.analyze_exit_reason_composition
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config.settings import PROJECT_ROOT
from detectors.feature_extraction import FEATURE_NAMES

DATASET_CACHE_PATH = PROJECT_ROOT / "logs" / "signal_model_dataset.parquet"
SMC_VWAP_PATH = PROJECT_ROOT / "logs" / "smc_vwap_partial_results.jsonl"
ORB_PATH = PROJECT_ROOT / "logs" / "orb_partial_results.jsonl"
N_SPLITS = 5
TOP_PCT = 0.20


def _build_exit_reason_lookup() -> dict[tuple[str, str, str], str]:
    """{(symbol, entry_time_iso, strategy_name): exit_reason}."""
    lookup: dict[tuple[str, str, str], str] = {}

    with open(SMC_VWAP_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["strategy"] != "smart_money":
                continue
            for t in rec["trades"]:
                lookup[(t["symbol"], t["entry_time"], t["strategy_name"])] = t["exit_reason"]

    with open(ORB_PATH) as f:
        for line in f:
            rec = json.loads(line)
            for t in rec["trades"]:
                lookup[(t["symbol"], t["entry_time"], t["strategy_name"])] = t["exit_reason"]

    return lookup


def main() -> None:
    df = pd.read_parquet(DATASET_CACHE_PATH)
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["is_smart_money"] = (df["strategy_name"] == "smart_money").astype(float)

    lookup = _build_exit_reason_lookup()
    df["entry_time_iso"] = df["entry_time"].apply(lambda t: t.isoformat())
    df["exit_reason"] = df.apply(
        lambda row: lookup.get((row["symbol"], row["entry_time_iso"], row["strategy_name"])), axis=1
    )
    matched = df["exit_reason"].notna().sum()
    print(f"Matched exit_reason for {matched}/{len(df)} trades ({len(df) - matched} unmatched)")
    print(f"Overall exit_reason distribution:\n{df['exit_reason'].value_counts()}\n")

    feature_cols = FEATURE_NAMES + ["is_smart_money"]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    exit_reasons = ["target", "stop_loss", "eod_square_off", "shutdown_close", "eod_square_off_backfilled"]

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        X_train, y_train = train_df[feature_cols].values, train_df["label_win"].values
        X_test, y_test = test_df[feature_cols].values, test_df["label_win"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = LogisticRegression(C=1.0, max_iter=1000).fit(X_train_s, y_train)
        test_probs = model.predict_proba(X_test_s)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)

        n_top = max(1, int(len(test_df) * TOP_PCT))
        top_pos = np.argsort(-test_probs)[:n_top]
        top_df = test_df.iloc[top_pos]

        baseline_avg_r = test_df["r_multiple"].mean()
        top_avg_r = top_df["r_multiple"].mean()

        print("=" * 100)
        print(f"FOLD {fold_num} -- test AUC {test_auc:.3f}, baseline avg R {baseline_avg_r:+.3f}, "
              f"top-20% avg R {top_avg_r:+.3f} ({'IMPROVED' if top_avg_r > baseline_avg_r else 'WORSE/FLAT'})")
        print("=" * 100)

        print(f"\n{'Exit reason':>26s} | {'Baseline %':>10s} {'Baseline N':>10s} {'Baseline avgR':>13s} | "
              f"{'Top20% %':>9s} {'Top20% N':>9s} {'Top20% avgR':>12s}")
        print("-" * 100)
        for reason in exit_reasons:
            base_mask = test_df["exit_reason"] == reason
            top_mask = top_df["exit_reason"] == reason
            base_n, top_n = base_mask.sum(), top_mask.sum()
            if base_n == 0 and top_n == 0:
                continue
            base_pct = base_n / len(test_df) * 100
            top_pct_val = top_n / len(top_df) * 100
            base_avg_r = test_df.loc[base_mask, "r_multiple"].mean() if base_n > 0 else float("nan")
            top_avg_r_reason = top_df.loc[top_mask, "r_multiple"].mean() if top_n > 0 else float("nan")
            print(f"{reason:>26s} | {base_pct:>9.1f}% {base_n:>10d} {base_avg_r:>+13.3f} | "
                  f"{top_pct_val:>8.1f}% {top_n:>9d} {top_avg_r_reason:>+12.3f}")
        print()


if __name__ == "__main__":
    main()
