"""
Regression version of the ML signal-combination experiment
(examples/train_signal_model.py trained a classifier on win/loss;
examples/analyze_exit_reason_composition.py found that classifier was
mechanically learning to avoid volatility -- filtering out both large
winners (target hits) and large losers (stop-losses) in favor of small-
margin EOD-square-off trades, since any R-multiple > 0 counted as an
equally-good "win". That is the wrong objective if the actual goal is
profitability.

This trains Ridge (L2-regularized linear regression) to predict the
ACTUAL R-multiple value directly -- the metric that actually matters --
using the SAME feature set and SAME 5 chronological TimeSeriesSplit folds
as the classifier experiments, so results are directly comparable.
Ranks test-set trades by PREDICTED R-multiple (not predicted win
probability) and evaluates the top 20% the same way as before: actual
avg R-multiple, win rate, and exit-reason composition vs the unfiltered
baseline, per fold.

Switching the training objective doesn't guarantee this works -- it just
points the model at the right target. Report honestly either way.

Run: python -m examples.train_regression_signal_model
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config.settings import PROJECT_ROOT
from detectors.feature_extraction import FEATURE_NAMES

DATASET_CACHE_PATH = PROJECT_ROOT / "logs" / "signal_model_dataset.parquet"
SMC_VWAP_PATH = PROJECT_ROOT / "logs" / "smc_vwap_partial_results.jsonl"
ORB_PATH = PROJECT_ROOT / "logs" / "orb_partial_results.jsonl"
N_SPLITS = 5
TOP_PCT = 0.20
MIN_RELIABLE_SAMPLE = 30
EXIT_REASONS = ["target", "stop_loss", "eod_square_off", "shutdown_close", "eod_square_off_backfilled"]


def _build_exit_reason_lookup() -> dict[tuple[str, str, str], str]:
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

    feature_cols = FEATURE_NAMES + ["is_smart_money"]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        X_train, y_train = train_df[feature_cols].values, train_df["r_multiple"].values
        X_test, y_test = test_df[feature_cols].values, test_df["r_multiple"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        train_pred = model.predict(X_train_s)
        test_pred = model.predict(X_test_s)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_corr = float(np.corrcoef(y_test, test_pred)[0, 1])

        win_labels_test = (y_test > 0).astype(int)
        unfiltered_win_rate = float(win_labels_test.mean())
        unfiltered_avg_r = float(y_test.mean())

        n_top = max(1, int(len(test_df) * TOP_PCT))
        top_pos = np.argsort(-test_pred)[:n_top]
        top_df = test_df.iloc[top_pos]
        top_win_rate = float(win_labels_test[top_pos].mean())
        top_avg_r = float(top_df["r_multiple"].mean())
        reliable = n_top >= MIN_RELIABLE_SAMPLE

        coefs = dict(zip(feature_cols, model.coef_))
        top_features = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        fold_results.append({
            "fold": fold_num, "test_n": len(test_df),
            "test_start": test_df["entry_time"].min(), "test_end": test_df["entry_time"].max(),
            "test_r2": test_r2, "train_r2": train_r2, "test_corr": test_corr,
            "unfiltered_win_rate": unfiltered_win_rate, "unfiltered_avg_r": unfiltered_avg_r,
            "top_n": n_top, "top_win_rate": top_win_rate, "top_avg_r": top_avg_r, "reliable": reliable,
            "coefs": coefs, "top_features": top_features, "top_df": top_df, "test_df": test_df,
        })

    print("=" * 100)
    print(f"RIDGE REGRESSION ON R-MULTIPLE -- {N_SPLITS} chronological folds")
    print("=" * 100)
    for r in fold_results:
        print(f"\n--- Fold {r['fold']} ---")
        print(f"  Test: {r['test_n']:>6d} trades, {r['test_start']} to {r['test_end']}")
        print(f"  Test R^2: {r['test_r2']:+.4f}  (train R^2: {r['train_r2']:+.4f}, reference only)")
        print(f"  Test correlation(predicted, actual R): {r['test_corr']:+.3f}")
        print(f"  Unfiltered: win rate {r['unfiltered_win_rate']:.1%}, avg R {r['unfiltered_avg_r']:+.3f}")
        reliability_note = "" if r["reliable"] else f" -- TOO FEW, need >= {MIN_RELIABLE_SAMPLE}"
        print(f"  Top {TOP_PCT:.0%} by PREDICTED R ({r['top_n']} trades{reliability_note}): "
              f"win rate {r['top_win_rate']:.1%} ({(r['top_win_rate'] - r['unfiltered_win_rate']) * 100:+.1f}pp), "
              f"avg R {r['top_avg_r']:+.3f}")
        print(f"  Top 5 features by |coef|: " + ", ".join(f"{name}({coef:+.3f})" for name, coef in r["top_features"]))

        # Exit-reason composition, same structure as analyze_exit_reason_composition.py
        test_df, top_df = r["test_df"], r["top_df"]
        print(f"\n  {'Exit reason':>24s} | {'Base %':>7s} {'Base N':>7s} {'Base avgR':>10s} | "
              f"{'Top20 %':>8s} {'Top20 N':>8s} {'Top20 avgR':>10s}")
        print("  " + "-" * 90)
        for reason in EXIT_REASONS:
            base_mask = test_df["exit_reason"] == reason
            top_mask = top_df["exit_reason"] == reason
            base_n, top_n_r = base_mask.sum(), top_mask.sum()
            if base_n == 0 and top_n_r == 0:
                continue
            base_pct = base_n / len(test_df) * 100
            top_pct_val = top_n_r / len(top_df) * 100
            base_avg_r = test_df.loc[base_mask, "r_multiple"].mean() if base_n > 0 else float("nan")
            top_avg_r_reason = top_df.loc[top_mask, "r_multiple"].mean() if top_n_r > 0 else float("nan")
            print(f"  {reason:>24s} | {base_pct:>6.1f}% {base_n:>7d} {base_avg_r:>+10.3f} | "
                  f"{top_pct_val:>7.1f}% {top_n_r:>8d} {top_avg_r_reason:>+10.3f}")

    print()
    print("=" * 100)
    print("SIDE-BY-SIDE SUMMARY (Ridge regression, ranked by predicted R-multiple)")
    print("=" * 100)
    print(f"{'Fold':>4s} {'TestN':>7s} {'TestR2':>8s} {'Corr':>6s} {'UnfiltWR':>9s} {'UnfiltR':>8s} "
          f"{'Top20WR':>8s} {'Top20R':>8s} {'R Improved?':>12s}")
    r_improved_count, wr_improved_count, reliable_count = 0, 0, 0
    for r in fold_results:
        r_improved = r["top_avg_r"] > r["unfiltered_avg_r"]
        wr_improved = r["top_win_rate"] > r["unfiltered_win_rate"]
        r_improved_count += int(r_improved and r["reliable"])
        wr_improved_count += int(wr_improved and r["reliable"])
        reliable_count += int(r["reliable"])
        print(f"{r['fold']:>4d} {r['test_n']:>7d} {r['test_r2']:>+8.4f} {r['test_corr']:>6.3f} "
              f"{r['unfiltered_win_rate']:>9.1%} {r['unfiltered_avg_r']:>+8.3f} "
              f"{r['top_win_rate']:>8.1%} {r['top_avg_r']:>+8.3f} "
              f"{'YES' if r_improved else 'no':>12s}" + ("" if r["reliable"] else "  [LOW SAMPLE]"))

    print()
    print("=" * 100)
    print("FEATURE IMPORTANCE CONSISTENCY (Ridge coefficients)")
    print("=" * 100)
    all_features = FEATURE_NAMES + ["is_smart_money"]
    coef_matrix = pd.DataFrame([r["coefs"] for r in fold_results], index=[f"fold{r['fold']}" for r in fold_results])
    coef_matrix = coef_matrix[all_features]
    print(coef_matrix.round(4).T)
    sign_consistent = coef_matrix.apply(lambda col: (col > 0).all() or (col < 0).all(), axis=0)
    print(f"\nFeatures with a CONSISTENT sign across all {N_SPLITS} folds: {sign_consistent.sum()}/{len(all_features)}")
    print("  " + ", ".join(sign_consistent[sign_consistent].index.tolist()))
    print(f"Features that FLIP SIGN across folds: {(~sign_consistent).sum()}/{len(all_features)}")
    print("  " + ", ".join(sign_consistent[~sign_consistent].index.tolist()))

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"Avg-R improvement (top 20% by predicted R > unfiltered) held in {r_improved_count}/{reliable_count} "
          f"reliable folds.")
    print(f"Win-rate improvement held in {wr_improved_count}/{reliable_count} reliable folds.")
    print(f"{sign_consistent.sum()}/{len(all_features)} features kept a consistent coefficient sign across all folds.")

    frac_r = r_improved_count / reliable_count if reliable_count else 0.0
    if frac_r >= 0.8 and sign_consistent.sum() >= len(all_features) * 0.7:
        print("\nSTABLE: switching to a regression objective produced a consistent avg-R improvement across "
              "most folds. This is a real, reproducible result -- still needs confirmation beyond these 5 "
              "folds before treating it as a validated edge, but this is the strongest honest reading so far.")
    elif frac_r <= 0.4:
        print("\nUNSTABLE: switching the training objective to directly predict R-multiple did NOT fix the "
              "instability -- the avg-R improvement still only shows up in a minority of folds. The "
              "classifier-vs-regressor framing was not the actual problem.")
    else:
        print("\nMIXED: switching to a regression objective changed the picture somewhat but did not produce "
              "a clean, consistent improvement across most folds. Do not round this up to 'fixed' -- still "
              "inconclusive.")


if __name__ == "__main__":
    main()
