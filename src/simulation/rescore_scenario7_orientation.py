"""Re-score stored FNN/RF replicate outputs against the re-oriented Scenario VII truth.

Scenario VII is the only scenario whose ground-truth ROC curve is built by Monte Carlo
(ground_truth_auc.true_roc_curve's `scenario == 7` branch) rather than by the closed-form
location-scale path. Until commit 95fc290 that branch used the arms in fixed order, while
_ab_from_means_stds() -- and every estimator in the study -- orients adaptively, putting
the higher-mean arm in the case role. Scenario VII was therefore the only scenario scored
against a target that could fall below AUC = 0.5.

This script recomputes the ROC-curve MSE of a stored run *without refitting anything*. It
rebuilds each method's predicted ROC curve from the per-subject `Predicted Mean` /
`Predicted Std Dev` columns that mlp_reg_data_simulation_multi.py and
rf_reg_data_simulation_multi_2.py already write to full_results_{healthy,diseased}_*.csv,
then scores it against both the old (fixed-order) and the new (oriented) ground truth,
drawn from the same seed the original run used.

The run is self-validating: the recomputed OLD mean MSE is compared against the stored
roc_mse_values.csv. If the two agree, the reconstruction is faithful and the NEW number
can be trusted; if they do not, the script says so and the NEW number should be ignored
in favour of a full re-run.

The linear, semiparametric-additive and naive baselines persist no per-subject output, so
they cannot be re-scored this way -- re-run their own scripts instead (they cost well
under a second per replicate plus ~15 s of ground truth).

Usage
-----
    python3 src/simulation/rescore_scenario7_orientation.py \
        -i output/fnn/scenario_7 -o rescore_fnn_s7.csv

    python3 src/simulation/rescore_scenario7_orientation.py \
        -i output/fnn_rf/scenario_7 -o rescore_rf_s7.csv

Pass -pt to restrict to one sample size, e.g. -pt '_5000_'.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ground_truth_auc
import true_dgp

EPSILON = 1e-8
P_GRID = np.linspace(0.001, 0.999, 100)   # identical to the grid used when scoring
CHUNK = 2000                              # identical to true_roc_curve's chunking


def predicted_roc(mean0, mean1, std0, std1, p=P_GRID):
    """Rebuild the estimator's conditional ROC curves exactly as the simulation scripts do.

    a/b use the adaptive orientation (higher-mean arm as the case arm), and the curve is
    the parametric Gaussian form 1 - Phi(Phi^-1(1-p) * b - a) -- not an empirical CDF of
    the residuals, which are stored squared and so cannot play that role.
    """
    higher_is_1 = mean1 > mean0
    a = np.where(higher_is_1,
                 (mean1 - mean0) / (std1 + EPSILON),
                 (mean0 - mean1) / (std0 + EPSILON))
    b = np.where(higher_is_1,
                 std0 / (std1 + EPSILON),
                 std1 / (std0 + EPSILON))
    z = norm.ppf(1 - p)                                  # (len_p,)
    return 1 - norm.cdf(z[None, :] * b[:, None] - a[:, None])


def truth_both_orientations(scenario, X0, X1, seed, p=P_GRID, n_mc=20000):
    """Ground truth under the old (fixed-order) and new (oriented) conventions.

    Both are derived from the *same* Monte Carlo draws, so the comparison isolates the
    orientation and carries no sampling difference. Draw order and chunking match
    true_roc_curve exactly, so the draws reproduce the original run's ground truth.
    """
    rng = np.random.default_rng(seed)

    if scenario != 7:
        # Closed-form path already orients; old and new coincide.
        roc = ground_truth_auc.true_roc_curve(scenario, X0, X1, p=p, rng=rng)
        return np.asarray(roc), np.asarray(roc)

    n = X0.shape[0]
    old_chunks, new_chunks = [], []
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        y0 = true_dgp.sample_conditional(7, 0, X0[start:end], n_mc=n_mc, rng=rng)
        y1 = true_dgp.sample_conditional(7, 1, X1[start:end], n_mc=n_mc, rng=rng)

        old_chunks.append(ground_truth_auc._roc_from_samples_batch(y0, y1, p))

        swap = y1.mean(axis=1) < y0.mean(axis=1)
        if swap.any():
            held = y0[swap]
            y0[swap] = y1[swap]
            y1[swap] = held
        new_chunks.append(ground_truth_auc._roc_from_samples_batch(y0, y1, p))

    return np.concatenate(old_chunks, axis=0), np.concatenate(new_chunks, axis=0)


def read_arm(path, cov_cols):
    df = pd.read_csv(path)
    missing = [c for c in list(cov_cols) + ["Predicted Mean", "Predicted Std Dev"]
               if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing column(s) {missing}; found {list(df.columns)}")
    return (df[list(cov_cols)].to_numpy(dtype=float),
            df["Predicted Mean"].to_numpy(dtype=float),
            df["Predicted Std Dev"].to_numpy(dtype=float))


def process(rep_dir, name, scenario, n_mc):
    cov_cols = true_dgp.covariate_columns(scenario)
    h = os.path.join(rep_dir, f"full_results_healthy_{name}.csv")
    d = os.path.join(rep_dir, f"full_results_diseased_{name}.csv")
    if not (os.path.exists(h) and os.path.exists(d)):
        return None

    X0, mean0, std0 = read_arm(h, cov_cols)
    X1, mean1, std1 = read_arm(d, cov_cols)
    if len(X0) != len(X1):
        k = min(len(X0), len(X1))
        print(f"  !! {name}: arms differ in length ({len(X0)} vs {len(X1)}), truncating to {k}")
        X0, mean0, std0 = X0[:k], mean0[:k], std0[:k]
        X1, mean1, std1 = X1[:k], mean1[:k], std1[:k]

    roc_hat = predicted_roc(mean0, mean1, std0, std1)
    roc_old, roc_new = truth_both_orientations(
        scenario, X0, X1, ground_truth_auc.seed_from_name(name), n_mc=n_mc)

    mse_old = ((roc_hat - roc_old) ** 2).mean(axis=1)
    mse_new = ((roc_hat - roc_new) ** 2).mean(axis=1)

    stored = np.nan
    sp = os.path.join(rep_dir, "roc_mse_values.csv")
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        col = "Mean Squared Error" if "Mean Squared Error" in s.columns else s.columns[-1]
        stored = float(s[col].mean())

    # share of subjects the orientation actually moves
    auc_old = np.array([simpson(c, P_GRID) for c in roc_old])
    return dict(replicate=name, n=len(X0),
                mse_old=float(mse_old.mean()), mse_new=float(mse_new.mean()),
                mse_stored=stored, frac_flipped=float((auc_old < 0.5).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input_dir", required=True,
                    help="scenario directory holding the per-replicate folders, "
                         "e.g. output/fnn/scenario_7")
    ap.add_argument("-o", "--output", default="rescore_s7.csv",
                    help="where to write the per-replicate table")
    ap.add_argument("-s", "--scenario", type=int, default=7)
    ap.add_argument("-pt", "--pattern", default="",
                    help="regex; only replicate folders matching it are processed "
                         "(e.g. '_5000_')")
    ap.add_argument("-nmc", "--n_mc", type=int, default=20000,
                    help="Monte Carlo draws per subject (must match the original run)")
    ap.add_argument("-l", "--limit", type=int, default=0,
                    help="stop after this many replicates (0 = all); useful for a "
                         "quick check before committing to the full pass")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"!! {args.input_dir} is not a directory")

    names = sorted(n for n in os.listdir(args.input_dir)
                   if os.path.isdir(os.path.join(args.input_dir, n)))

    # Hard guard, not merely a default: rf_reg_data_simulation_multi_2.py writes every
    # replicate of every scenario into one <var_to_group>_rf/ directory, with no
    # per-scenario level. A loose -pt such as '_5000_' would then sweep in Scenario I--IX
    # folders and score each against Scenario VII's Monte Carlo truth, silently producing
    # garbage. The folder name carries the scenario, so require it to agree.
    scen_rx = re.compile(rf"^scenario_{args.scenario}_")
    off_scenario = [n for n in names if not scen_rx.match(n)]
    names = [n for n in names if scen_rx.match(n)]
    if off_scenario:
        print(f"    ignoradas {len(off_scenario)} carpetas de otros escenarios "
              f"(p.ej. {off_scenario[0]})")

    if args.pattern:
        rx = re.compile(args.pattern)
        names = [n for n in names if rx.search(n)]
    if args.limit:
        names = names[:args.limit]
    if not names:
        sys.exit(f"!! no scenario-{args.scenario} replicate folders under "
                 f"{args.input_dir} matching {args.pattern!r}")

    print(f"=== re-scoring {len(names)} replicate(s) from {args.input_dir} ===")
    print(f"    scenario {args.scenario} | n_mc = {args.n_mc}\n")

    rows, skipped = [], []
    for i, name in enumerate(names, 1):
        try:
            r = process(os.path.join(args.input_dir, name), name, args.scenario, args.n_mc)
        except Exception as exc:                      # keep going; report at the end
            print(f"  [{i}/{len(names)}] {name}: FAILED -- {exc}")
            skipped.append(name)
            continue
        if r is None:
            skipped.append(name)
            continue
        rows.append(r)
        print(f"  [{i}/{len(names)}] {name}: vieja={r['mse_old']:.5f} "
              f"nueva={r['mse_new']:.5f} guardada={r['mse_stored']:.5f} "
              f"| {r['frac_flipped']:.1%} sujetos reorientados")

    if not rows:
        sys.exit("!! nothing could be re-scored (no full_results_* files found?)")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    old, new, sto = df.mse_old.mean(), df.mse_new.mean(), df.mse_stored.mean()
    print(f"\n{'='*66}")
    print(f"  replicas re-puntuadas : {len(df)}" + (f"  (saltadas: {len(skipped)})" if skipped else ""))
    print(f"  sujetos reorientados  : {df.frac_flipped.mean():.1%}")
    print(f"\n  MSE con la verdad VIEJA (orden fijo)   : {old:.5f}")
    print(f"  MSE guardada en roc_mse_values.csv     : {sto:.5f}")
    print(f"  MSE con la verdad NUEVA (orientada)    : {new:.5f}")

    # Self-check: does the reconstruction reproduce the published number?
    if np.isnan(sto):
        print("\n  ?? no roc_mse_values.csv found -- reconstruction could NOT be validated.")
    else:
        rel = abs(old - sto) / max(sto, 1e-12)
        if rel < 0.02:
            print(f"\n  OK: la reconstruccion reproduce la cifra publicada (desvio {rel:.2%}).")
            print(f"      => la MSE orientada {new:.5f} es fiable; sustituye a {sto:.5f}"
                  f" en la fila VII.")
        else:
            print(f"\n  !! la reconstruccion NO reproduce la cifra publicada "
                  f"({old:.5f} vs {sto:.5f}, desvio {rel:.1%}).")
            print("      => NO uses la MSE orientada. Algo mas difiere entre este script y")
            print("         el pipeline original; hay que relanzar el escenario entero.")

    print(f"\n  tabla por replica -> {args.output}")


if __name__ == "__main__":
    main()
