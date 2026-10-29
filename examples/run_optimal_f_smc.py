"""
PART 2 of the "world-champion trader concepts" tests: Optimal-F position
sizing (Ralph Vince) -- NOT a new trading signal. Reuses the Smart Money
Concepts strategy's ALREADY-COMPUTED historical trades exactly as found
in logs/smart_money_verification_partial_results.jsonl (the full
1,644-symbol-universe "verified" SMC run -- 351 real trades, the
largest-sample SMC result on record in this project, produced by
examples/run_smart_money_verification_universe.py). Holding the entry
signal fixed, this asks a pure risk-management question: does the
position-SIZING method change the outcome on the identical trade
sequence?

IMPORTANT, STATED UP FRONT rather than discovered by surprise later:
this project's own prior verification work already found the "verified"
SMC edge did NOT hold at full scale (win rate 42.7%, avg R -0.04, Sharpe
-0.06, expectancy -1.12 pts/trade -- see
logs/smart_money_verification_stdout.log). Optimal-F computed honestly
on a NEGATIVE-historical-expectancy sequence is expected to come out
near zero -- the formula correctly says "don't size this up, there is no
edge here to compound," which is the OPPOSITE of "dangerously large."
That is itself the honest finding to report, not a result to
second-guess or swap out for a smaller, cherrier-picked, more-flattering
dataset.

METHODOLOGY: see risk/optimal_f.py's module docstring for why Optimal-F
is computed here as HPR_i(f) = 1 + f*R_i (no re-normalization by the
sample's worst trade) -- the R-multiples are already normalized to each
trade's own risk, matching this project's existing fixed-1%-risk
convention (risk/position_sizing.py), and re-dividing by the worst trade
again would silently change what "f" means, breaking the three-way
comparison below.

Trades are reconstructed via examples/run_smart_money_verification_universe.py's
own _trade_from_dict() (reused, not re-derived) so r_multiple comes from
backtest/engine.py's Trade.r_multiple property exactly as originally
computed -- not hand-rolled a second time, which would risk a silent
formula mismatch.

THREE-WAY COMPARISON on the SAME 351-trade sequence, same chronological
order, only f differs: (a) fixed 1% risk (current approach), (b)
computed Optimal-f, (c) half-Optimal-f (commonly recommended in practice
since full Optimal-f is very aggressive/high-drawdown). Reports total
return, max drawdown, Sharpe for each via backtest/metrics.py's
simulate_fixed_fraction_equity_curve() (reused, not reimplemented).

SAMPLE-SIZE SENSITIVITY, demonstrated empirically: rather than hunting
for a differently-filtered smaller trade set (which would conflate
"smaller sample" with "a stricter entry filter"), bootstraps 200 random
60-trade subsamples (with replacement) from the SAME 351-trade R-multiple
pool and reports how much the resulting Optimal-f estimate swings --
Optimal-F's well-documented real-world failure mode (a point estimate
fit to past trades, highly sensitive to exactly which trades happen to
be in the sample), shown with a real, reproducible number instead of an
assertion.

Run: python -m examples.run_optimal_f_smc
"""
from __future__ import annotations

import json

import numpy as np

from backtest.metrics import compute, simulate_fixed_fraction_equity_curve
from backtest.engine import Trade
from config.settings import PROJECT_ROOT
from examples.run_smart_money_verification_universe import _trade_from_dict
from risk.optimal_f import compute_optimal_f

PARTIAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "smart_money_verification_partial_results.jsonl"
FIXED_RISK_PCT = 0.01
N_BOOTSTRAP_DRAWS = 200
BOOTSTRAP_SAMPLE_SIZE = 60
RANDOM_SEED = 42
# Purely a reporting threshold for the honest-caveat section below -- not
# a claim from the formula itself, just a plain-language way to describe
# whatever number actually comes out.
DANGEROUSLY_LARGE_F_THRESHOLD = 0.10


def _load_all_trades() -> list[Trade]:
    trades: list[Trade] = []
    with open(PARTIAL_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for t in record["trades"]:
                trades.append(_trade_from_dict(t))
    return trades


def main() -> None:
    print(f"Loading trades from {PARTIAL_RESULTS_PATH.name}...")
    trades = _load_all_trades()
    trades.sort(key=lambda t: t.exit_time)
    print(f"Loaded {len(trades)} SMC trades (full 1,644-symbol universe, 'verified' variant), "
          f"sorted chronologically by exit_time.\n")

    report = compute(trades)
    print("=" * 70)
    print("BASELINE CHECK -- must match logs/smart_money_verification_stdout.log's already-known result")
    print("=" * 70)
    print(report)
    print()

    r_multiples = np.array([t.r_multiple for t in trades])

    print("=" * 70)
    print("OPTIMAL-F (full sample, chronological order)")
    print("=" * 70)
    result = compute_optimal_f(r_multiples)
    print(f"Trades: {result.n_trades}  |  Avg R: {r_multiples.mean():+.3f}  |  Worst single trade: {result.worst_r:+.2f}R")
    print(f"f_max (bounded by worst trade -- beyond this, that one trade alone wipes out equity): {result.f_max:.3f}")
    print(f"Optimal-f: {result.optimal_f:.4f}  ({result.optimal_f:.1%} of capital risked per trade)")
    print(f"Half-Optimal-f: {result.half_optimal_f:.4f}  ({result.half_optimal_f:.1%})")
    print(f"Terminal Wealth Relative at optimal-f: {result.twr_at_optimal:.4f}x starting capital")

    print("\n" + "=" * 70)
    print("THREE-WAY SIZING COMPARISON (same 351-trade sequence, same order, only f differs)")
    print("=" * 70)
    for label, f in (
        ("(a) Fixed 1% risk (current approach)", FIXED_RISK_PCT),
        ("(b) Optimal-f", result.optimal_f),
        ("(c) Half-Optimal-f", result.half_optimal_f),
    ):
        eq = simulate_fixed_fraction_equity_curve(r_multiples, f)
        print(f"{label:<38s} f={f:.4f}  total_return={eq.total_return_pct:+9.2f}%  "
              f"max_drawdown={eq.max_drawdown_pct:7.2f}%  Sharpe={eq.sharpe:+.2f}")

    print("\n" + "=" * 70)
    print(f"SAMPLE-SIZE SENSITIVITY: {N_BOOTSTRAP_DRAWS} bootstrap draws of {BOOTSTRAP_SAMPLE_SIZE} trades "
          f"each, WITH replacement, from the SAME {len(trades)}-trade pool")
    print("=" * 70)
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_fs = np.array([
        compute_optimal_f(rng.choice(r_multiples, size=BOOTSTRAP_SAMPLE_SIZE, replace=True)).optimal_f
        for _ in range(N_BOOTSTRAP_DRAWS)
    ])
    print(f"Optimal-f distribution: min={bootstrap_fs.min():.3f}  median={np.median(bootstrap_fs):.3f}  "
          f"max={bootstrap_fs.max():.3f}  mean={bootstrap_fs.mean():.3f}  std={bootstrap_fs.std():.3f}")

    print("\n" + "=" * 70)
    print("HONEST CAVEAT")
    print("=" * 70)
    size_word = "dangerously large" if result.optimal_f >= DANGEROUSLY_LARGE_F_THRESHOLD else "essentially zero, NOT dangerously large"
    print(f"Full-sample Optimal-f computes to {result.optimal_f:.1%} -- {size_word}.")
    print(f"This directly reflects the sample's historical expectancy (avg R {r_multiples.mean():+.3f}, "
          f"Sharpe {report.sharpe_ratio:+.2f}): {'the formula is correctly saying this strategy has no real edge to compound, so it should not be sized up at all -- the honest answer here is the ABSENCE of a sizing edge, not its magnitude.' if result.optimal_f < DANGEROUSLY_LARGE_F_THRESHOLD else 'a positive-looking historical edge is being amplified into an aggressive sizing recommendation.'}")
    print(f"Bootstrap spread ({bootstrap_fs.min():.1%} to {bootstrap_fs.max():.1%}, std {bootstrap_fs.std():.1%}) "
          f"across 60-trade subsamples of this SAME underlying population shows how much this number can swing "
          f"from pure sample-size/sequence noise, independent of any real change in the underlying edge -- "
          f"Optimal-F's well-known real-world failure mode: it is a point estimate fit to past trades, highly "
          f"sensitive to exactly which trades happen to be in the sample. Neither the full-sample number nor "
          f"any single bootstrap draw should be trusted as a stable, forward-looking sizing recommendation on "
          f"a sample this size.")


if __name__ == "__main__":
    main()
