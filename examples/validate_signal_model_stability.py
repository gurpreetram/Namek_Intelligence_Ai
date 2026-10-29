"""
Cross-split stability check for the ML signal-combination experiment
(examples/train_signal_model.py). That script's single Jun-15/Jul-1 split
found: test AUC 0.587, and the top-20%-confidence subset showed avg R
+0.035 vs -0.055 unfiltered. One split is not enough to trust that --
it could be a stable pattern, or it could be specific to whatever
happened to be different about that particular 3-week test window
(a regime shift, a handful of unusual symbols, etc.).

This re-runs the identical feature set / model / evaluation across 5
DIFFERENT chronological splits via sklearn's TimeSeriesSplit -- expanding
training window, non-overlapping sequential test folds, exactly the
walk-forward discipline already established in backtest/walk_forward.py
and examples/train_signal_model.py, just repeated 5 times instead of
once. Uses the SAME cached feature dataset (logs/signal_model_dataset.parquet)
built by train_signal_model.py -- no need to re-extract features, only
the train/test boundaries change per fold.

Run: python -m examples.validate_signal_model_stability
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config.settings import PROJECT_ROOT
from detectors.feature_extraction import FEATURE_NAMES

DATASET_CACHE_PATH = PROJECT_ROOT / "logs" / "signal_model_dataset.parquet"
N_SPLITS = 5
TOP_PCT = 0.20
MIN_RELIABLE_SAMPLE = 30  # same discipline as risk/probability_calibration.py


def main() -> None:
    df = pd.read_parquet(DATASET_CACHE_PATH)
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["is_smart_money"] = (df["strategy_name"] == "smart_money").astype(float)
    feature_cols = FEATURE_NAMES + ["is_smart_money"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        X_train, y_train = train_df[feature_cols].values, train_df["label_win"].values
        X_test, y_test = test_df[feature_cols].values, test_df["label_win"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(X_train_s, y_train)
        test_probs = model.predict_proba(X_test_s)[:, 1]

        test_auc = roc_auc_score(y_test, test_probs)
        unfiltered_win_rate = float(y_test.mean())
        unfiltered_avg_r = float(test_df["r_multiple"].mean())

        n_top = max(1, int(len(test_df) * TOP_PCT))
        top_idx = np.argsort(-test_probs)[:n_top]
        top_win_rate = float(y_test[top_idx].mean())
        top_avg_r = float(test_df["r_multiple"].values[top_idx].mean())
        reliable = n_top >= MIN_RELIABLE_SAMPLE

        coefs = dict(zip(feature_cols, model.coef_[0]))
        top_features = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        fold_results.append({
            "fold": fold_num,
            "train_n": len(train_df), "test_n": len(test_df),
            "train_start": train_df["entry_time"].min(), "train_end": train_df["entry_time"].max(),
            "test_start": test_df["entry_time"].min(), "test_end": test_df["entry_time"].max(),
            "test_auc": test_auc,
            "unfiltered_win_rate": unfiltered_win_rate, "unfiltered_avg_r": unfiltered_avg_r,
            "top_n": n_top, "top_win_rate": top_win_rate, "top_avg_r": top_avg_r, "reliable": reliable,
            "coefs": coefs, "top_features": top_features,
        })

    print("=" * 100)
    print(f"CROSS-SPLIT STABILITY CHECK -- {N_SPLITS} chronological folds (expanding train window, TimeSeriesSplit)")
    print("=" * 100)
    for r in fold_results:
        print(f"\n--- Fold {r['fold']} ---")
        print(f"  Train: {r['train_n']:>6d} trades, {r['train_start']} to {r['train_end']}")
        print(f"  Test:  {r['test_n']:>6d} trades, {r['test_start']} to {r['test_end']}")
        print(f"  Test AUC: {r['test_auc']:.3f}")
        print(f"  Unfiltered: win rate {r['unfiltered_win_rate']:.1%}, avg R {r['unfiltered_avg_r']:+.3f}")
        reliability_note = "" if r["reliable"] else f" -- TOO FEW, need >= {MIN_RELIABLE_SAMPLE}"
        print(f"  Top {TOP_PCT:.0%} ({r['top_n']} trades{reliability_note}): "
              f"win rate {r['top_win_rate']:.1%} ({(r['top_win_rate'] - r['unfiltered_win_rate']) * 100:+.1f}pp), "
              f"avg R {r['top_avg_r']:+.3f}")
        print(f"  Top 5 features by |coef|: " + ", ".join(f"{name}({coef:+.3f})" for name, coef in r["top_features"]))

    print()
    print("=" * 100)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 100)
    header = f"{'Fold':>4s} {'TestN':>7s} {'AUC':>6s} {'UnfiltWR':>9s} {'UnfiltR':>8s} {'Top20WR':>8s} {'Top20R':>8s} {'R Improved?':>12s}"
    print(header)
    r_improved_count = 0
    wr_improved_count = 0
    reliable_count = 0
    for r in fold_results:
        r_improved = r["top_avg_r"] > r["unfiltered_avg_r"]
        wr_improved = r["top_win_rate"] > r["unfiltered_win_rate"]
        r_improved_count += int(r_improved and r["reliable"])
        wr_improved_count += int(wr_improved and r["reliable"])
        reliable_count += int(r["reliable"])
        print(f"{r['fold']:>4d} {r['test_n']:>7d} {r['test_auc']:>6.3f} "
              f"{r['unfiltered_win_rate']:>9.1%} {r['unfiltered_avg_r']:>+8.3f} "
              f"{r['top_win_rate']:>8.1%} {r['top_avg_r']:>+8.3f} "
              f"{'YES' if r_improved else 'no':>12s}"
              + ("" if r["reliable"] else "  [LOW SAMPLE]"))

    print()
    print("=" * 100)
    print("FEATURE IMPORTANCE CONSISTENCY")
    print("=" * 100)
    all_features = FEATURE_NAMES + ["is_smart_money"]
    coef_matrix = pd.DataFrame([r["coefs"] for r in fold_results], index=[f"fold{r['fold']}" for r in fold_results])
    coef_matrix = coef_matrix[all_features]
    print(coef_matrix.round(3).T)
    # Sign consistency: for each feature, does its coefficient keep the
    # same sign across all folds? A feature that flips sign fold-to-fold
    # isn't a stable, trustworthy signal regardless of its average magnitude.
    sign_consistent = (coef_matrix.apply(lambda col: (col > 0).all() or (col < 0).all(), axis=0))
    print(f"\nFeatures with a CONSISTENT sign across all {N_SPLITS} folds: "
          f"{sign_consistent.sum()}/{len(all_features)}")
    print("  " + ", ".join(sign_consistent[sign_consistent].index.tolist()))
    print(f"Features that FLIP SIGN across folds (unstable): "
          f"{(~sign_consistent).sum()}/{len(all_features)}")
    print("  " + ", ".join(sign_consistent[~sign_consistent].index.tolist()))

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"Avg-R improvement (top 20% > unfiltered) held in {r_improved_count}/{reliable_count} reliable folds "
          f"({N_SPLITS - reliable_count} fold(s) had too few trades in the top-20% subset to count).")
    print(f"Win-rate improvement held in {wr_improved_count}/{reliable_count} reliable folds.")
    print(f"{sign_consistent.sum()}/{len(all_features)} features kept a consistent coefficient sign across all folds.")

    frac_r = r_improved_count / reliable_count if reliable_count else 0.0
    if frac_r >= 0.8 and sign_consistent.sum() >= len(all_features) * 0.7:
        print("\nSTABLE: the avg-R improvement from confidence-filtering held up across most folds, "
              "with mostly consistent feature signs. This is the strongest honest reading available -- "
              "still not a validated tradeable edge without further confirmation, but a real, "
              "reproducible pattern rather than a one-split fluke.")
    elif frac_r <= 0.4:
        print("\nUNSTABLE: the avg-R improvement only showed up in a minority of folds. The original "
              "single-split result looks more like it was specific to that one test window than a "
              "stable, reproducible pattern. Treat the earlier finding as NOT confirmed.")
    else:
        print("\nMIXED: the avg-R improvement showed up in some folds but not consistently across most "
              "of them. This is not a clean confirmation -- do not round this up to 'it works'. At best, "
              "this is inconclusive and would need a larger number of folds or more data to resolve.")


if __name__ == "__main__":
    main()
