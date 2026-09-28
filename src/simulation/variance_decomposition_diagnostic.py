"""Diagnostic: decomposes the parametric/gaussian bootstrap's replicate-to-replicate AUC
variance (bootstrap_crossfit_parametric.py) into two components, to check a hypothesis
raised by the first real-scale validation (Scenario 1 severely OVER-covered at every CI
level tested, Scenario 4 severely UNDER-covered -- opposite failure modes that a single
"wrong AUC formula" explanation doesn't fit):

  1. Training noise: _fit_group_bootstrap re-initializes both networks (random weights)
     and re-shuffles the K folds (KFold(..., shuffle=True), no fixed seed) on EVERY call,
     even called twice on the exact same (X, Y, W) with no resampling at all. Any AUC
     spread from that alone is pure optimization/init randomness, not statistical
     uncertainty about mu/sigma -- and it doesn't shrink or grow with how hard the
     estimation problem is, only with how noisy training itself is (e.g. epoch budget,
     architecture, batch size).
  2. Full parametric bootstrap: the usual Y* regeneration + refit (as in
     bootstrap_crossfit_gaussian_paired), which mixes the same training noise together
     with genuine parametric-bootstrap resampling variance.

If (1) is close to (2), the reported bootstrap "confidence" bands are mostly measuring
training noise, not genuine estimation uncertainty -- which would explain over-coverage on
an easy scenario (little real uncertainty, so training noise dominates and inflates the
band) without contradicting under-coverage on a hard scenario (there, mu-bias from
underfitting the true mean function dominates over anything the bootstrap measures,
training noise included).

Uses the SAME n_repeat count for both arms so the two variances are directly comparable
(same number of independent draws), and reuses the exact fitting/AUC code the real
pipeline uses (_fit_group_bootstrap, _gaussian_auc, bootstrap_crossfit_gaussian_paired) so
this measures the actual deployed procedure, not a reimplementation of it.
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch

from data_simulation_reg import load_data
from bootstrap_crossfit_oob import _fit_group_bootstrap
from bootstrap_crossfit_parametric import _gaussian_auc, bootstrap_crossfit_gaussian_paired


def _training_noise_replicates(X, Y, W, mlp_config, K, device, n_repeat):
    """n_repeat independent calls to _fit_group_bootstrap on the SAME (X, Y, W) -- no
    resampling, no Y* regeneration, X_eval == X. Returns (mu_reps, sigma_reps), each
    (n_repeat, N) -- any row-to-row spread here is pure training-randomness noise.
    """
    mu_reps = np.empty((n_repeat, X.shape[0]))
    sigma_reps = np.empty((n_repeat, X.shape[0]))
    for r in range(n_repeat):
        mu, sigma, _ = _fit_group_bootstrap(X, Y, W, X, mlp_config, K, device)
        mu_reps[r] = mu
        sigma_reps[r] = sigma
    return mu_reps, sigma_reps


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
    n_repeat = config["n_repeat"]
    K = config["k_folds"]

    file = config["input_file"]
    df = pd.read_csv(file)
    selected_combination = df.columns[df.columns.str.contains("x_D")].tolist()
    print(f"Running variance decomposition for {file} (n_repeat={n_repeat}, K={K})")

    data_all, data_0, data_1 = load_data(file, selected_combination, target, var_to_group)
    X0, Y0, W0 = data_0["x"], data_0["y"], data_0["w"]
    X1, Y1, W1 = data_1["x"], data_1["y"], data_1["w"]

    print(" > Training-noise-only replicates (same Y, no bootstrap)...")
    mu0_r, sigma0_r = _training_noise_replicates(X0, Y0, W0, mlp_config, K, device, n_repeat)
    mu1_r, sigma1_r = _training_noise_replicates(X1, Y1, W1, mlp_config, K, device, n_repeat)
    auc_noise = np.stack([
        _gaussian_auc(mu0_r[r], mu1_r[r], sigma0_r[r], sigma1_r[r], adaptive_direction=True)
        for r in range(n_repeat)
    ])

    print(" > Full parametric (gaussian) bootstrap replicates (Y* regenerated each time)...")
    _, _, _, _, auc_boot = bootstrap_crossfit_gaussian_paired(
        X0, Y0, W0, X1, Y1, W1, mlp_config, n_repeat, K, device, ci_levels=(0.95,),
    )

    var_noise = auc_noise.var(axis=0)
    var_boot = auc_boot.var(axis=0)
    ratio = var_noise / np.clip(var_boot, 1e-12, None)

    print(f" > Median training-noise variance : {np.median(var_noise):.6f}")
    print(f" > Median full-bootstrap variance  : {np.median(var_boot):.6f}")
    print(f" > Median ratio (noise / full)     : {np.median(ratio):.3f}")
    print(f" > Mean ratio (noise / full)       : {np.mean(ratio):.3f}")

    os.makedirs(config["output_dir"], exist_ok=True)
    pd.DataFrame({
        "Row Index": np.arange(len(var_noise)),
        "Var Training Noise": var_noise,
        "Var Full Bootstrap": var_boot,
        "Ratio": ratio,
    }).to_csv(os.path.join(config["output_dir"], "variance_decomposition.csv"), index=False)
    print(f" > Wrote {config['output_dir']}/variance_decomposition.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decompose parametric/gaussian bootstrap AUC variance into training "
                     "noise (same data, repeated refits) vs. full bootstrap (Y* regenerated)."
    )
    parser.add_argument("-i", "--input_file", required=True, help="Un unico fichero de replica de escenario (CSV)")
    parser.add_argument("-o", "--output_dir", default="./output/variance_decomposition", help="Directorio de salida")
    parser.add_argument("-n", "--n_repeat", type=int, default=30, help="Numero de repeticiones en cada brazo")
    parser.add_argument("-k", "--k_folds", type=int, default=5, help="Numero de folds de cross-fitting K")
    parser.add_argument("-e", "--num_epochs", type=int, default=50, help="Numero de epocas por fold")
    parser.add_argument("-es", "--early_stop_patience", type=int, default=10, help="Early stopping")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Tamano de batch")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4, help="Regularizacion L2 (weight decay)")
    parser.add_argument("-dr", "--dropout", type=float, default=0.2, help="Tasa de dropout")
    parser.add_argument("-hl", "--hidden_layers", type=ast.literal_eval, default="[64, 32, 16]",
                         help="Capas ocultas como '[64, 32, 16]'")
    args = parser.parse_args()

    config = {
        "input_file": args.input_file,
        "output_dir": args.output_dir,
        "n_repeat": args.n_repeat,
        "k_folds": args.k_folds,
        "num_epochs": args.num_epochs,
        "early_stop_patience": args.early_stop_patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "hidden_layers": args.hidden_layers,
    }

    main(config)
