"""Pointwise bootstrap + cross-fitting + OOB coverage for the FNN two-stage estimator on
the nine simulation scenarios (Reviewer 2, Major Comment 4: "the simulation study should
report pointwise ... coverage"). Reuses bootstrap_crossfit_oob.bootstrap_crossfit_oob_paired
-- the same engine notebooks/nhanes_hetero_residuos.ipynb uses for the real-data case study
-- so the confidence bands come from one procedure across both parts of the paper.

For each scenario replicate CSV (same file layout mlp_reg_data_simulation_multi.py reads):
per subject i, the bootstrap-OOB 95% CI for the covariate-adjusted AUC is compared against
the ground-truth AUC(x_i) from ground_truth_auc.true_auc (exact for the eight Gaussian
scenarios, Monte Carlo for Scenario 7's non-Gaussian healthy arm). Writes coverage.csv
(per-row true AUC / estimated mean / CI / n_oob / covered) and timing.csv, both consumed by
src/postprocessing/statistics_summary.py's coverage_summary.csv aggregation.

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

        true_auc_vals = ground_truth_auc.true_auc(scenario_num, X0_true, X1_true)

        t0 = time.perf_counter()
        auc_mean, auc_lower, auc_upper, n_oob = bootstrap_crossfit_oob_paired(
            X0, Y0, W0, X1, Y1, W1, mlp_config,
            config["bootstrap_reps"], config["k_folds"], device,
            n_mc=config["n_mc"], ci_level=config["ci_level"],
        )
        fit_seconds_total = time.perf_counter() - t0

        # NaN (not the boolean False a naive comparison against a NaN CI would give) for
        # rows without enough OOB draws to form a CI, so the coverage rate below (and
        # statistics_summary.py's aggregation) can exclude them with nanmean instead of
        # silently counting them as "not covered".
        has_ci = ~np.isnan(auc_mean)
        covered = np.full(len(true_auc_vals), np.nan)
        covered[has_ci] = (auc_lower[has_ci] <= true_auc_vals[has_ci]) & (true_auc_vals[has_ci] <= auc_upper[has_ci])

        coverage_df = pd.DataFrame({
            "Scenario": sceminario,
            "Method": "FNN-Coverage",
            "Row Index": np.arange(len(true_auc_vals)),
            "True AUC": true_auc_vals,
            "Estimated AUC Mean": auc_mean,
            "CI Lower": auc_lower,
            "CI Upper": auc_upper,
            "N OOB": n_oob,
            "Covered": covered,
        })

        op = f'{config["output_file"]}/{sceminario}'
        os.makedirs(op, exist_ok=True)
        coverage_df.to_csv(f"{op}/coverage.csv", index=False)

        timing_df = pd.DataFrame([{
            "Scenario": sceminario,
            "Method": "FNN-Coverage",
            "Fit Seconds Total": fit_seconds_total,
        }])
        timing_df.to_csv(f"{op}/timing.csv", index=False)

        if has_ci.any():
            print(f"  Pointwise coverage ({int(has_ci.sum())}/{len(has_ci)} rows with a CI): "
                  f"{np.nanmean(covered):.3f}")
        else:
            print("  No rows had enough OOB draws to form a CI (increase --bootstrap_reps).")


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
    parser.add_argument("-ci", "--ci_level", type=float, default=0.95, help="Nivel de confianza del intervalo")
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
        "ci_level": args.ci_level,
    }

    main(config)
