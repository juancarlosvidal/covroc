"""Pointwise bootstrap + cross-fitting coverage for the FNN two-stage estimator on the nine
simulation scenarios (Reviewer 2, Major Comment 4: "the simulation study should report
pointwise ... coverage"). --method selects which confidence-band engine to use:
  oob         -- bootstrap_crossfit_oob.bootstrap_crossfit_oob_paired (nonparametric
                 subject resampling + OOB), the original/default method, also what
                 notebooks/nhanes_hetero_residuos.ipynb uses for the real-data case study.
  parametric  -- bootstrap_crossfit_parametric.bootstrap_crossfit_parametric_paired
                 (parametric bootstrap of Y at fixed X + cross-fitting, Monte Carlo AUC).
  gaussian    -- bootstrap_crossfit_parametric.bootstrap_crossfit_gaussian_paired (same
                 parametric bootstrap, closed-form Gaussian AUC, no Monte Carlo).
See bootstrap_crossfit_parametric.py's module docstring for the full derivation of the
latter two and why their point estimate/CI split differs from the OOB path's.

For each scenario replicate CSV (same file layout mlp_reg_data_simulation_multi.py reads):
per subject i, the chosen method's CI (--ci_level, default 0.95) for the covariate-adjusted
AUC is compared against the ground-truth AUC(x_i) from ground_truth_auc.true_auc (exact for
the eight Gaussian scenarios, Monte Carlo for Scenario 7's non-Gaussian healthy arm).
--ci_level accepts a comma-separated list (e.g. "0.95,0.8") for parametric/gaussian --
useful for calibration checks (does empirical coverage track the nominal level at 80% as
well as at 95%?) -- computed from the SAME bootstrap replicates at no extra fit cost;
--method oob only accepts a single level per run. Writes coverage.csv (per-row true AUC /
estimated AUC / CI / n draws / covered, plus a Boot Mean AUC diagnostic column for
parametric/gaussian) and timing.csv, both consumed by
src/postprocessing/statistics_summary.py's coverage_summary.csv aggregation. The "Method"
column tags rows by method AND ci_level (e.g. "FNN-Coverage-Gaussian-CI80") so per-method,
per-level comparisons are just a groupby away -- no downstream script needs to change.

Scope (agreed with the paper's authors): pointwise coverage only, not simultaneous/uniform
bands; FNN only, matching the real-data notebook (Random Forest coverage can follow later).
"""
import argparse
import ast
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from data_simulation_reg import load_data
import true_dgp
import ground_truth_auc
from bootstrap_crossfit_oob import bootstrap_crossfit_oob_paired
from bootstrap_crossfit_parametric import bootstrap_crossfit_parametric_paired, bootstrap_crossfit_gaussian_paired

METHOD_LABELS = {
    "oob": "FNN-Coverage",
    "parametric": "FNN-Coverage-Parametric",
    "gaussian": "FNN-Coverage-Gaussian",
}


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" > Device: {device}")

    mlp_config = {
        "hidden_layers": config["hidden_layers"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "num_epochs": config["num_epochs"],
        "early_stop_patience": config["early_stop_patience"],
    }

    target = "Y_generated"
    var_to_group = "mortstat"

    onlyfiles = [f for f in os.listdir(config["input_dir"]) if os.path.isfile(os.path.join(config["input_dir"], f))]
    for f in onlyfiles:
        file = os.path.join(config["input_dir"], f)
        df = pd.read_csv(file)
        selected_combination = df.columns[df.columns.str.contains("x_D")].tolist()
        print(f"Running coverage analysis for file {file}")

        dataset_name = os.path.splitext(os.path.basename(f))[0]
        sceminario = dataset_name
        match = re.match(r"^(scenario_\d+_\d+_\d+)_data$", dataset_name)
        if match:
            sceminario = match.group(1)
        scenario_num = int(re.match(r"^scenario_(\d+)_", f).group(1))

        index_0 = df[df[var_to_group] == 0].copy()
        index_1 = df[df[var_to_group] == 1].copy()
        if "True_Mean_Y" in index_0.columns:
            del index_0["True_Mean_Y"]
        if "True_Mean_Y" in index_1.columns:
            del index_1["True_Mean_Y"]

        ground_truth_cols = true_dgp.covariate_columns(scenario_num)
        X0_true = df.loc[df[var_to_group] == 0, ground_truth_cols].to_numpy(dtype=float)
        X1_true = df.loc[df[var_to_group] == 1, ground_truth_cols].to_numpy(dtype=float)

        data_all, data_0, data_1 = load_data(file, selected_combination, target, var_to_group)
        X0, Y0, W0 = data_0["x"], data_0["y"], data_0["w"]
        X1, Y1, W1 = data_1["x"], data_1["y"], data_1["w"]

        rng = np.random.default_rng(ground_truth_auc.seed_from_name(sceminario))
        true_auc_vals = ground_truth_auc.true_auc(scenario_num, X0_true, X1_true, rng=rng)

        method = config["method"]
        ci_levels = config["ci_levels"]
        auc_boot_mean = None
        auc_reps = None
        t0 = time.perf_counter()
        if method == "oob":
            # bootstrap_crossfit_oob_paired only supports a single ci_level per call (its
            # OOB filtering, unlike the parametric/gaussian percentile aggregation below,
            # isn't set up to hand back multiple levels from one bootstrap run), and
            # doesn't expose raw replicate draws either (no PIT column for this method).
            if len(ci_levels) != 1:
                raise ValueError(f"--method oob only supports a single --ci_level, got {ci_levels}")
            auc_point, auc_lower, auc_upper, n_draws = bootstrap_crossfit_oob_paired(
                X0, Y0, W0, X1, Y1, W1, mlp_config,
                config["bootstrap_reps"], config["k_folds"], device,
                n_mc=config["n_mc"], ci_level=ci_levels[0],
            )
            ci_bounds = {ci_levels[0]: (auc_lower, auc_upper)}
        elif method == "parametric":
            auc_point, ci_bounds, n_draws, auc_boot_mean, auc_reps = bootstrap_crossfit_parametric_paired(
                X0, Y0, W0, X1, Y1, W1, mlp_config,
                config["bootstrap_reps"], config["k_folds"], device,
                n_mc=config["n_mc"], ci_levels=ci_levels,
            )
        elif method == "gaussian":
            auc_point, ci_bounds, n_draws, auc_boot_mean, auc_reps = bootstrap_crossfit_gaussian_paired(
                X0, Y0, W0, X1, Y1, W1, mlp_config,
                config["bootstrap_reps"], config["k_folds"], device,
                ci_levels=ci_levels,
            )
        else:
            raise ValueError(f"Unknown --method {method!r}")
        fit_seconds_total = time.perf_counter() - t0

        # PIT (probability integral transform): the fraction of the B bootstrap-replicate
        # AUCs at or below the ground-truth AUC, per subject. A calibration check that's
        # richer than any single nominal --ci_level -- if the bootstrap is well-calibrated,
        # these should be ~Uniform(0,1) across subjects, and it lets *any* level's coverage
        # be checked later (P(true inside a level-L CI) = P(L/2 < PIT < 1-L/2)) without
        # re-running the fit. Independent of --ci_level, so the same PIT column is repeated
        # across each ci_level's row block below. Not available for method=oob (see above).
        pit = np.full(len(true_auc_vals), np.nan) if auc_reps is None else (auc_reps <= true_auc_vals[None, :]).mean(axis=0)

        # NaN (not the boolean False a naive comparison against a NaN CI would give) for
        # rows without enough draws to form a CI (only possible for method=oob, where a
        # subject can end up with fewer than min_samples OOB draws -- parametric/gaussian
        # always get exactly bootstrap_reps draws), so the coverage rate below (and
        # statistics_summary.py's aggregation) can exclude them with nanmean instead of
        # silently counting them as "not covered".
        has_ci = ~np.isnan(auc_point)

        # One row block per requested ci_level, stacked into a single coverage.csv, with
        # the level baked into "Method" (e.g. "FNN-Coverage-Gaussian-CI80") so
        # statistics_summary.py's existing groupby("Scenario ID", "Method") separates them
        # with no changes needed -- all levels come from the SAME bootstrap replicates
        # (ci_bounds is computed once, per-level, inside _aggregate_bootstrap), so this
        # costs nothing extra over computing a single level.
        level_frames = []
        for level, (auc_lower, auc_upper) in ci_bounds.items():
            covered = np.full(len(true_auc_vals), np.nan)
            covered[has_ci] = (auc_lower[has_ci] <= true_auc_vals[has_ci]) & (true_auc_vals[has_ci] <= auc_upper[has_ci])
            level_frames.append(pd.DataFrame({
                "Scenario": sceminario,
                "Method": f"{METHOD_LABELS[method]}-CI{round(level * 100)}",
                "Row Index": np.arange(len(true_auc_vals)),
                "True AUC": true_auc_vals,
                "Estimated AUC Mean": auc_point,
                "CI Lower": auc_lower,
                "CI Upper": auc_upper,
                "N OOB": n_draws,
                "Covered": covered,
                "Boot Mean AUC": auc_boot_mean if auc_boot_mean is not None else np.nan,
                "PIT": pit,
            }))
        coverage_df = pd.concat(level_frames, ignore_index=True)

        op = f'{config["output_file"]}/{sceminario}'
        os.makedirs(op, exist_ok=True)
        coverage_df.to_csv(f"{op}/coverage.csv", index=False)

        timing_df = pd.DataFrame([{
            "Scenario": sceminario,
            "Method": METHOD_LABELS[method],
            "Fit Seconds Total": fit_seconds_total,
        }])
        timing_df.to_csv(f"{op}/timing.csv", index=False)

        if has_ci.any():
            for level in ci_bounds:
                lvl_covered = coverage_df.loc[coverage_df["Method"] == f"{METHOD_LABELS[method]}-CI{round(level * 100)}", "Covered"]
                print(f"  Pointwise coverage at {level:.0%} ({int(has_ci.sum())}/{len(has_ci)} rows with a CI): "
                      f"{np.nanmean(lvl_covered):.3f}")
        else:
            print("  No rows had enough draws to form a CI (increase --bootstrap_reps).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pointwise bootstrap+cross-fitting+OOB coverage for the FNN two-stage "
                     "estimator on the simulation scenarios (Reviewer 2, Major Comment 4)."
    )
    parser.add_argument("-i", "--input_dir", default="./data/simulation", help="Directorio de entrada")
    parser.add_argument("-o", "--output_file", default="./output", help="Directorio de salida")
    parser.add_argument("-b", "--bootstrap_reps", type=int, default=100, help="Número de réplicas bootstrap B")
    parser.add_argument("-k", "--k_folds", type=int, default=5, help="Número de folds de cross-fitting K")
    parser.add_argument("-e", "--num_epochs", type=int, default=50, help="Número de épocas por fold")
    parser.add_argument("-es", "--early_stop_patience", type=int, default=10, help="Early stopping")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4, help="Regularización L2 (weight decay)")
    parser.add_argument("-dr", "--dropout", type=float, default=0.2, help="Tasa de dropout")
    parser.add_argument("-hl", "--hidden_layers", type=ast.literal_eval, default="[64, 32, 16]",
                         help="Capas ocultas como '[64, 32, 16]'")
    parser.add_argument("-mc", "--n_mc", type=int, default=500, help="Número de muestras Monte Carlo por AUC")
    parser.add_argument("-ci", "--ci_level", type=str, default="0.95",
                         help="Nivel(es) de confianza del intervalo, separados por coma (p.ej. '0.95,0.8'). "
                              "--method oob solo admite un nivel por ejecucion; parametric/gaussian calculan "
                              "todos los niveles pedidos a partir de las mismas replicas bootstrap, sin coste extra.")
    parser.add_argument("-m", "--method", choices=["oob", "parametric", "gaussian"], default="oob",
                         help="Motor de bandas de confianza: oob (bootstrap de sujetos + OOB, por defecto), "
                              "parametric (bootstrap parametrico + AUC Monte Carlo) o gaussian (bootstrap "
                              "parametrico + AUC analitica via Phi). Ver bootstrap_crossfit_parametric.py.")
    args = parser.parse_args()

    config = {
        "input_dir": args.input_dir,
        "output_file": args.output_file,
        "bootstrap_reps": args.bootstrap_reps,
        "k_folds": args.k_folds,
        "num_epochs": args.num_epochs,
        "early_stop_patience": args.early_stop_patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "hidden_layers": args.hidden_layers,
        "n_mc": args.n_mc,
        "ci_levels": tuple(float(x) for x in args.ci_level.split(",")),
        "method": args.method,
    }

    main(config)
