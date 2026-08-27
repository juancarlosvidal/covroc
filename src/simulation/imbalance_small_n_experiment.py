"""Robustness check for the FNN under a small, class-imbalanced sample (Reviewer 2,
Major Comment 5: "more realistic class imbalance and smaller samples"). Deliberately
kept separate from mlp_reg_data_simulation_multi.py/data_generation.py rather than
generalizing them to unequal group sizes: the whole pipeline (ground_truth_auc.py, the
baselines, bootstrap_crossfit_oob.py) assumes healthy/diseased covariate arrays of
equal length, row-paired for the ground-truth comparison, and reworking that assumption
throughout would be a much larger, riskier change than this one-off script needs.

Per replicate: draws a single dataset of n_total subjects split minority_frac/
(1-minority_frac) between the diseased and healthy arms (both train AND evaluate on
this same imbalanced draw, matching a real deployment where the minority class stays
scarce at both stages -- see conversation with the user). Trains the same two-stage FNN
estimator (MLP/train_model reused from mlp_reg_data_simulation_multi.py, with a single
80/20 train/val split instead of that module's nested K-fold CV, appropriate for the
much smaller n here) on the full imbalanced draw. Scores against the ground truth using
every minority-arm subject, each paired with an independently-sampled majority-arm
subject from the same draw -- reuses ground_truth_auc.true_roc_curve unmodified, since
both arrays end up the same length (the minority count) regardless of the imbalance
ratio, sidestepping the equal-length assumption above without changing it.

Only Scenarios I and IV are covered (both single-covariate, U(-1,1), Gaussian in both
arms -- no special-casing needed, unlike Scenario VII's mixture).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

import true_dgp
import ground_truth_auc
from mlp_reg_data_simulation_multi import MLP, train_model, compute_mean, compute_std, compute_residues

P_GRID = np.linspace(0.001, 0.999, 100)


def _make_loader(X, y, batch_size, shuffle):
    w = np.ones_like(y, dtype=np.float32)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _fit_stage(X, y, mlp_config, device, val_frac=0.2, rng=None):
    n = X.shape[0]
    idx = (rng or np.random.default_rng()).permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    model = MLP(X.shape[1], mlp_config["hidden_layers"], mlp_config["dropout"]).to(device)
    model = train_model(
        model,
        _make_loader(X[train_idx], y[train_idx], mlp_config["batch_size"], shuffle=True),
        _make_loader(X[val_idx], y[val_idx], mlp_config["batch_size"], shuffle=False),
        device, mlp_config["learning_rate"], mlp_config["weight_decay"],
        mlp_config["num_epochs"], mlp_config["early_stop_patience"],
    )
    return model


def run_replicate(scenario, name, n_total, minority_frac, mlp_config, device):
    rng = np.random.default_rng(ground_truth_auc.seed_from_name(name))

    n_diseased = max(1, int(round(n_total * minority_frac)))
    n_healthy = n_total - n_diseased

    X0 = rng.uniform(-1, 1, size=(n_healthy, 1))
    X1 = rng.uniform(-1, 1, size=(n_diseased, 1))
    Y0 = true_dgp.sample_conditional(scenario, 0, X0, n_mc=1, rng=rng).ravel().astype(np.float32)
    Y1 = true_dgp.sample_conditional(scenario, 1, X1, n_mc=1, rng=rng).ravel().astype(np.float32)
    X0, X1 = X0.astype(np.float32), X1.astype(np.float32)

    t0 = time.perf_counter()
    mean_model_0 = _fit_stage(X0, Y0, mlp_config, device, rng=rng)
    mean_model_1 = _fit_stage(X1, Y1, mlp_config, device, rng=rng)

    resid_0 = compute_residues({'x': X0, 'y': Y0.reshape(-1, 1)}, mean_model_0)
    resid_1 = compute_residues({'x': X1, 'y': Y1.reshape(-1, 1)}, mean_model_1)
    std_model_0 = _fit_stage(X0, resid_0.astype(np.float32), mlp_config, device, rng=rng)
    std_model_1 = _fit_stage(X1, resid_1.astype(np.float32), mlp_config, device, rng=rng)
    fit_seconds_total = time.perf_counter() - t0

    # Evaluation pairs: every minority-arm (diseased) subject, each paired with an
    # independently-sampled majority-arm (healthy) subject from the same draw --
    # X0_eval/X1_eval end up equal length (n_diseased), so ground_truth_auc.true_roc_curve
    # can be called exactly as everywhere else in the repository.
    eval_healthy_idx = rng.choice(n_healthy, size=n_diseased, replace=(n_healthy < n_diseased))
    X0_eval, X1_eval = X0[eval_healthy_idx], X1

    mean_0 = compute_mean({'x': X0_eval}, mean_model_0).ravel()
    mean_1 = compute_mean({'x': X1_eval}, mean_model_1).ravel()
    std_0 = compute_std({'x': X0_eval}, std_model_0).ravel()
    std_1 = compute_std({'x': X1_eval}, std_model_1).ravel()

    epsilon = 1e-8
    higher_is_1 = mean_1 > mean_0
    a_predicted = np.where(higher_is_1, (mean_1 - mean_0) / (std_1 + epsilon), (mean_0 - mean_1) / (std_0 + epsilon))
    b_predicted = np.where(higher_is_1, std_0 / (std_1 + epsilon), std_1 / (std_0 + epsilon))

    # Parametric (Gaussian) formula, matching every other roc_mse_values.csv-producing
    # script in the repository (mlp_reg_data_simulation_multi.py,
    # rf_reg_data_simulation_multi_2.py) -- resid_0/resid_1 here are squared residuals
    # (compute_residues), not usable for an empirical-CDF-based curve.
    from scipy.stats import norm
    roc_predicted = [1 - norm.cdf(norm.ppf(1 - P_GRID) * b_predicted[i] - a_predicted[i]) for i in range(n_diseased)]

    eval_rng = np.random.default_rng(ground_truth_auc.seed_from_name(name + "_eval"))
    roc_real = list(ground_truth_auc.true_roc_curve(scenario, X0_eval, X1_eval, p=P_GRID, rng=eval_rng))

    mse_values = [mean_squared_error(real, pred) for real, pred in zip(roc_real, roc_predicted)]

    return {
        "Scenario": f"scenario_{scenario}",
        "N Total": n_total,
        "Minority Frac": minority_frac,
        "N Diseased": n_diseased,
        "N Healthy": n_healthy,
        "Replicate": name,
        "Mean ROC MSE": float(np.mean(mse_values)),
        "Fit Seconds Total": fit_seconds_total,
    }


def main(scenarios, n_total, minority_frac, n_replicates, output_csv, mlp_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" > Device: {device}")

    rows = []
    for scenario in scenarios:
        for rep in range(n_replicates):
            name = f"imbalance_scenario_{scenario}_{n_total}_{minority_frac}_{rep}"
            print(f"=== {name} ===")
            rows.append(run_replicate(scenario, name, n_total, minority_frac, mlp_config, device))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small-n, class-imbalanced FNN robustness check (Reviewer 2, Major Comment 5)")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 4], help="Scenario numbers (only 1 and 4 supported: single-covariate, Gaussian-in-both-arms)")
    parser.add_argument("--n-total", type=int, default=500, help="Total subjects per replicate (both arms combined)")
    parser.add_argument("--minority-frac", type=float, default=0.1, help="Fraction of n_total in the diseased (minority) arm")
    parser.add_argument("--n-replicates", type=int, default=100, help="Number of replicates per scenario")
    parser.add_argument("--output-csv", default="output/imbalance_small_n/results.csv", help="Path to write per-replicate results")
    parser.add_argument("-e", "--num-epochs", type=int, default=800, help="Epochs per stage")
    parser.add_argument("-es", "--early-stop-patience", type=int, default=10)
    parser.add_argument("-bs", "--batch-size", type=int, default=32)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-4)
    parser.add_argument("-dr", "--dropout", type=float, default=0.2)
    parser.add_argument("-hl", "--hidden-layers", type=int, nargs="+", default=[64, 32, 16])
    args = parser.parse_args()

    mlp_config = {
        "hidden_layers": args.hidden_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "early_stop_patience": args.early_stop_patience,
    }
    main(args.scenarios, args.n_total, args.minority_frac, args.n_replicates, args.output_csv, mlp_config)
