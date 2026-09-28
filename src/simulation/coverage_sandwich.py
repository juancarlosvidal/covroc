"""Pointwise sandwich (robust M-estimation) + delta-method coverage for the covariate-
adjusted AUC -- a fast, non-bootstrap alternative to coverage_bootstrap_crossfit.py's
oob/parametric/gaussian methods. See sandwich_delta_ci.py's module docstring for the full
derivation (Murphy & Topel 1985 / Newey 1984 two-step M-estimation sandwich, applied to
the two-stage mean+variance network pair via autodiff Hessian-vector products -- no
retraining, no bootstrap replicates).

Key differences from coverage_bootstrap_crossfit.py:
  - Fits ONE model per group (sandwich_delta_ci.fit_single: no K-fold cross-fitting) --
    the sandwich/delta theory needs a single, cleanly-defined M-estimator theta_hat, not a
    fold-averaged ensemble. This reintroduces the in-sample residual-variance bias
    cross-fitting was built to avoid (Reviewer 2 Major Comment 3) -- a known trade-off.
  - Evaluates only a subsample of --n_points subjects per replicate (default 100), NOT
    every row: unlike a_x = grad eta(x) (cheap, one backward pass), the sandwich variance
    itself needs conjugate-gradient solves against the fitted model's Hessian, which cost
    about the same per query regardless of x (each CG iteration is a fresh forward+backward
    pass over the WHOLE group's data) -- benchmarked at roughly 0.25s/subject at n=500 on
    CPU (see project memory), so evaluating all ~5000 subjects would cost as much as the
    bootstrap methods this was meant to be a fast alternative to. A modest, reproducible
    subsample (matching src/postprocessing/simultaneous_coverage_summary.py's existing
    --n-points convention) keeps the coverage-rate estimate meaningful without that cost.
  - Reports a stationarity diagnostic (mean training-score norm per group) alongside the
    CI: the sandwich/delta approximation assumes theta_hat is close to a genuine stationary
    point of the training loss, which early stopping (based on validation-loss plateau, not
    a training-loss gradient check) does not guarantee -- see sandwich_delta_ci.py's module
    docstring. No safe threshold for this has been calibrated yet; it's reported for
    inspection, not yet used to flag/reject a given CI automatically.
  - No n_mc (no Monte Carlo anywhere in this method).
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
import sandwich_delta_ci as sdc


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
    ci_levels = config["ci_levels"]
    rng_points = np.random.default_rng(config["points_seed"])

    # --pattern keeps a single sequential run from also picking up the n=20000 files that
    # sit in the same scenario folder (4x the cost). Default matches n=5000 only.
    onlyfiles = sorted(f for f in os.listdir(config["input_dir"])
                       if os.path.isfile(os.path.join(config["input_dir"], f))
                       and re.search(config["pattern"], f))
    if not onlyfiles:
        raise SystemExit(f"No file in {config['input_dir']} matches --pattern {config['pattern']!r}")
    print(f" > {len(onlyfiles)} ficheros coinciden con --pattern {config['pattern']!r}")
    for f in onlyfiles:
        file = os.path.join(config["input_dir"], f)
        df = pd.read_csv(file)
        selected_combination = df.columns[df.columns.str.contains("x_D")].tolist()
        print(f"Running sandwich coverage for file {file}")

        dataset_name = os.path.splitext(os.path.basename(f))[0]
        sceminario = dataset_name
        match = re.match(r"^(scenario_\d+_\d+_\d+)_data$", dataset_name)
        if match:
            sceminario = match.group(1)
        scenario_num = int(re.match(r"^scenario_(\d+)_", f).group(1))

        # Resumable: a long sequential run (100 replicates x 9 scenarios) that dies part
        # way should pick up where it left off, not restart. --overwrite forces a redo.
        op_check = f'{config["output_file"]}/{sceminario}/coverage.csv'
        if os.path.exists(op_check) and not config["overwrite"]:
            print(f"   {op_check} ya existe -- saltando")
            continue

        ground_truth_cols = true_dgp.covariate_columns(scenario_num)
        X0_true = df.loc[df[var_to_group] == 0, ground_truth_cols].to_numpy(dtype=float)
        X1_true = df.loc[df[var_to_group] == 1, ground_truth_cols].to_numpy(dtype=float)

        data_all, data_0, data_1 = load_data(file, selected_combination, target, var_to_group)
        X0, Y0, W0 = data_0["x"], data_0["y"], data_0["w"]
        X1, Y1, W1 = data_1["x"], data_1["y"], data_1["w"]
        N0, N1 = X0.shape[0], X1.shape[0]
        assert N0 == N1, "X0/X1 must be row-paired (same length) -- see ground_truth_auc's convention"
        N = N0

        rng = np.random.default_rng(ground_truth_auc.seed_from_name(sceminario))
        true_auc_vals = ground_truth_auc.true_auc(scenario_num, X0_true, X1_true, rng=rng)

        # --- Fit: ONE mean-network + ONE variance-network per group, no cross-fitting of
        # the mean net itself. --residuals controls what the variance net is trained on. ---
        use_oof = config["residuals"] == "oof"
        t_fit0 = time.perf_counter()
        mu0_model = sdc.fit_single(X0, Y0, W0, mlp_config, device)
        mu1_model = sdc.fit_single(X1, Y1, W1, mlp_config, device)
        if use_oof:
            # Out-of-fold residuals (K auxiliary mean-net fits per group) remove the
            # downward sigma bias that the in-sample variant's residuals carry -- the
            # dominant source of under-coverage measured on the real-scale runs.
            resid0_sq = sdc.oof_squared_residuals(X0, Y0, W0, mlp_config, config["k_folds"], device)
            resid1_sq = sdc.oof_squared_residuals(X1, Y1, W1, mlp_config, config["k_folds"], device)
        else:
            resid0_sq = ((Y0.ravel() - mu0_model(torch.tensor(X0, dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()) ** 2).reshape(-1, 1).astype(np.float32)
            resid1_sq = ((Y1.ravel() - mu1_model(torch.tensor(X1, dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()) ** 2).reshape(-1, 1).astype(np.float32)
        sigma0_model = sdc.fit_single(X0, resid0_sq, W0, mlp_config, device)
        sigma1_model = sdc.fit_single(X1, resid1_sq, W1, mlp_config, device)
        fit_seconds = time.perf_counter() - t_fit0
        # Ratio of OOF to in-sample residual scale: a direct read on how much downward
        # sigma bias the in-sample variant carries (>1 means in-sample residuals are too
        # small, which inflates eta and pushes the AUC toward 0/1).
        insample0 = ((Y0.ravel() - mu0_model(torch.tensor(X0, dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()) ** 2).mean()
        insample1 = ((Y1.ravel() - mu1_model(torch.tensor(X1, dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()) ** 2).mean()
        resid_ratio0 = float(resid0_sq.mean() / insample0)
        resid_ratio1 = float(resid1_sq.mean() / insample1)

        X0_t = torch.tensor(X0, dtype=torch.float32).to(device)
        Y0_t = torch.tensor(Y0, dtype=torch.float32).view(-1).to(device)
        W0_t = torch.tensor(W0, dtype=torch.float32).view(-1).to(device)
        X1_t = torch.tensor(X1, dtype=torch.float32).to(device)
        Y1_t = torch.tensor(Y1, dtype=torch.float32).view(-1).to(device)
        W1_t = torch.tensor(W1, dtype=torch.float32).view(-1).to(device)

        # --- Build each group's sandwich context ONCE (x-independent quantities). ---
        t_ctx0 = time.perf_counter()
        # cross_term is False for the OOF variant: stage 2's targets no longer depend on
        # theta_mu, so H_sigma_mu is zero by construction (see oof_squared_residuals).
        resid0_t = torch.tensor(resid0_sq, dtype=torch.float32).view(-1).to(device)
        resid1_t = torch.tensor(resid1_sq, dtype=torch.float32).view(-1).to(device)
        ctx0 = sdc.GroupSandwichContext(mu0_model, sigma0_model, X0_t, Y0_t, W0_t, config["damping"],
                                         resid2_t=resid0_t, cross_term=not use_oof)
        ctx1 = sdc.GroupSandwichContext(mu1_model, sigma1_model, X1_t, Y1_t, W1_t, config["damping"],
                                         resid2_t=resid1_t, cross_term=not use_oof)
        ctx_seconds = time.perf_counter() - t_ctx0

        # --- Evaluate a reproducible subsample of subjects (module docstring: full N is
        # too expensive, CG cost/query rivals the bootstrap methods this replaces). ---
        if config["target"] == "avg":
            # AVERAGED functional: one number per replicate, averaged over ALL n covariate
            # rows (not the --n_points grid), and ONE CG solve instead of n_points of them.
            t_ci0 = time.perf_counter()
            abar, ci_bounds, se, parts = sdc.averaged_auc_ci(ctx0, ctx1, X0_t, X1_t, ci_levels=ci_levels)
            ci_seconds = time.perf_counter() - t_ci0
            true_abar = float(np.mean(true_auc_vals))
            var_theta, var_cov, cross = parts
            print(f"  Fit: {fit_seconds:.1f}s | Context: {ctx_seconds:.1f}s | AVG target: {ci_seconds:.1f}s | "
                  f"Abar_hat={abar:.4f} true={true_abar:.4f} SE={se:.5f} "
                  f"(var_theta={var_theta:.2e}, var_cov={var_cov:.2e}, cross={cross:.2e}) | "
                  f"score norm=({ctx0.mean_score_norm:.4f}, {ctx1.mean_score_norm:.4f})")
            frames = []
            for level, (lo, hi) in ci_bounds.items():
                frames.append(pd.DataFrame([{
                    "Scenario": sceminario, "Method": f"FNN-Sandwich-AVG-CI{round(level*100)}",
                    "Row Index": -1, "True AUC": true_abar, "Estimated AUC Mean": abar,
                    "CI Lower": lo, "CI Upper": hi, "SE Eta": se,
                    "Covered": float(lo <= true_abar <= hi),
                    "Var Theta": var_theta, "Var Cov": var_cov, "Cross": cross,
                }]))
            coverage_df = pd.concat(frames, ignore_index=True)
            op = f'{config["output_file"]}/{sceminario}'
            os.makedirs(op, exist_ok=True)
            coverage_df.to_csv(f"{op}/coverage.csv", index=False)
            pd.DataFrame([{"Scenario": sceminario, "Method": "FNN-Sandwich-AVG",
                           "Fit Seconds": fit_seconds, "Context Seconds": ctx_seconds,
                           "CI Seconds Total": ci_seconds, "CI Seconds Per Subject": ci_seconds,
                           "N Points": N, "Residuals": config["residuals"],
                           "Score Norm G0": ctx0.mean_score_norm, "Score Norm G1": ctx1.mean_score_norm,
                           "Fit Seconds Total": fit_seconds + ctx_seconds + ci_seconds,
                           }]).to_csv(f"{op}/timing.csv", index=False)
            for level in ci_levels:
                cv = coverage_df.loc[coverage_df["Method"].str.endswith(f"CI{round(level*100)}"), "Covered"].iloc[0]
                print(f"  Cobertura del promedio al {level:.0%}: {'SI' if cv else 'NO'}")
            continue

        n_points = min(config["n_points"], N)
        idx = rng_points.choice(N, size=n_points, replace=False)
        idx.sort()

        level_rows = {level: [] for level in ci_levels}
        se_etas = np.full(N, np.nan)
        score_norm0 = np.nan
        score_norm1 = np.nan
        t_ci0 = time.perf_counter()
        for i in idx:
            x0_row = X0_t[i:i + 1]
            x1_row = X1_t[i:i + 1]
            auc_point, ci_bounds, se_eta, s0n, s1n = sdc.sandwich_auc_ci(ctx0, ctx1, x0_row, x1_row, ci_levels=ci_levels)
            se_etas[i] = se_eta
            score_norm0, score_norm1 = s0n, s1n
            for level, (lower, upper) in ci_bounds.items():
                covered = lower <= true_auc_vals[i] <= upper
                level_rows[level].append({
                    "Row Index": i, "True AUC": true_auc_vals[i], "Estimated AUC Mean": auc_point,
                    "CI Lower": lower, "CI Upper": upper, "SE Eta": se_eta, "Covered": float(covered),
                })
        ci_seconds = time.perf_counter() - t_ci0
        print(f"  Fit: {fit_seconds:.1f}s | Context build: {ctx_seconds:.1f}s | "
              f"CI for {n_points} subjects: {ci_seconds:.1f}s ({ci_seconds / n_points:.3f}s/subject) | "
              f"mean training-score norm (group0, group1) = ({score_norm0:.4f}, {score_norm1:.4f}) | "
              f"residual scale vs in-sample (g0, g1) = ({resid_ratio0:.3f}, {resid_ratio1:.3f})")

        method_label = "FNN-Coverage-SandwichOOF" if use_oof else "FNN-Coverage-Sandwich"
        level_frames = []
        for level, rows in level_rows.items():
            level_df = pd.DataFrame(rows)
            level_df["Scenario"] = sceminario
            level_df["Method"] = f"{method_label}-CI{round(level * 100)}"
            level_frames.append(level_df)
        coverage_df = pd.concat(level_frames, ignore_index=True)
        coverage_df = coverage_df[["Scenario", "Method", "Row Index", "True AUC", "Estimated AUC Mean",
                                    "CI Lower", "CI Upper", "SE Eta", "Covered"]]

        op = f'{config["output_file"]}/{sceminario}'
        os.makedirs(op, exist_ok=True)
        coverage_df.to_csv(f"{op}/coverage.csv", index=False)

        timing_df = pd.DataFrame([{
            "Scenario": sceminario, "Method": method_label,
            "Fit Seconds": fit_seconds, "Context Seconds": ctx_seconds,
            "CI Seconds Total": ci_seconds, "CI Seconds Per Subject": ci_seconds / n_points,
            "N Points": n_points, "Residuals": config["residuals"],
            "Resid Ratio G0": resid_ratio0, "Resid Ratio G1": resid_ratio1,
            "Score Norm G0": score_norm0, "Score Norm G1": score_norm1,
            "Fit Seconds Total": fit_seconds + ctx_seconds + ci_seconds,  # for statistics_summary.py compatibility
        }])
        timing_df.to_csv(f"{op}/timing.csv", index=False)

        for level in ci_levels:
            covered = coverage_df.loc[coverage_df["Method"] == f"{method_label}-CI{round(level*100)}", "Covered"]
            print(f"  Pointwise coverage at {level:.0%} ({n_points} sampled subjects): {covered.mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pointwise sandwich+delta-method coverage for the covariate-adjusted "
                     "AUC -- fast, non-bootstrap alternative to coverage_bootstrap_crossfit.py."
    )
    parser.add_argument("-i", "--input_dir", default="./data/simulation", help="Directorio de entrada")
    parser.add_argument("-o", "--output_file", default="./output", help="Directorio de salida")
    parser.add_argument("-e", "--num_epochs", type=int, default=800, help="Numero de epocas por red")
    parser.add_argument("-es", "--early_stop_patience", type=int, default=10, help="Early stopping")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Tamano de batch")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4, help="Regularizacion L2 (weight decay)")
    parser.add_argument("-dr", "--dropout", type=float, default=0.2, help="Tasa de dropout")
    parser.add_argument("-hl", "--hidden_layers", type=ast.literal_eval, default="[64, 32, 16]",
                         help="Capas ocultas como '[64, 32, 16]'")
    parser.add_argument("-ci", "--ci_level", type=str, default="0.95",
                         help="Nivel(es) de confianza separados por coma (p.ej. '0.95,0.8,0.5'), "
                              "calculados del mismo SE sin coste extra por nivel.")
    parser.add_argument("-np", "--n_points", type=int, default=100,
                         help="Numero de sujetos (submuestra reproducible) evaluados por replica -- "
                              "ver el docstring del modulo, evaluar los N completos es tan caro como el bootstrap.")
    parser.add_argument("-ps", "--points_seed", type=int, default=0, help="Semilla para la submuestra de sujetos")
    parser.add_argument("-pt", "--pattern", type=str, default=r"_5000_.*_data\.csv$",
                         help="Regex sobre el nombre de fichero; por defecto solo n=5000.")
    parser.add_argument("-r", "--residuals", choices=["insample", "oof"], default="insample",
                         help="Objetivo de la red de varianza: 'insample' (residuos de la propia red de "
                              "media, sesga sigma a la baja) u 'oof' (residuos out-of-fold via K ajustes "
                              "auxiliares, corrige ese sesgo; ver sandwich_delta_ci.oof_squared_residuals)")
    parser.add_argument("-k", "--k_folds", type=int, default=5,
                         help="Folds para los residuos out-of-fold (solo con --residuals oof)")
    parser.add_argument("-dp", "--damping", type=float, default=1e-3,
                         help="Amortiguacion minima para los sistemas CG (se sube automaticamente si "
                              "hace falta para garantizar definitud positiva -- ver sandwich_delta_ci.py)")
    parser.add_argument("-tg", "--target", choices=["pointwise", "avg"], default="pointwise",
                         help="pointwise: un IC por sujeto evaluado (por defecto). avg: inferencia sobre "
                              "el funcional promediado mean_i AUC(x_i), un numero por replica.")
    parser.add_argument("-ow", "--overwrite", action="store_true", help="Recalcula aunque exista coverage.csv")
    args = parser.parse_args()

    config = {
        "input_dir": args.input_dir,
        "output_file": args.output_file,
        "num_epochs": args.num_epochs,
        "early_stop_patience": args.early_stop_patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "hidden_layers": args.hidden_layers,
        "ci_levels": tuple(float(x) for x in args.ci_level.split(",")),
        "n_points": args.n_points,
        "points_seed": args.points_seed,
        "pattern": args.pattern,
        "overwrite": args.overwrite,
        "target": args.target,
        "damping": args.damping,
        "residuals": args.residuals,
        "k_folds": args.k_folds,
    }

    main(config)
