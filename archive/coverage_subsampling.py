"""Simultaneous confidence bands for the covariate-adjusted AUC by SUBSAMPLING
(Politis & Romano), plus a check of the convergence-rate assumption the band depends on.

Why this method, given coverage_sandwich.py already gives pointwise intervals ~100x
cheaper: subsampling resamples the FINAL ESTIMATOR, not the network weights, so it needs
no Hessian, no identifiability of weights, and no differentiability anywhere -- the ReLU
kinks and the |mu1-mu0| kink that complicate the sandwich are simply irrelevant here. And
it produces a genuine UNIFORM band, which neither the sandwich nor the bootstrap-percentile
methods do (src/postprocessing/simultaneous_coverage_summary.py currently approximates one
by Bonferroni-widening pointwise intervals, which is crude and conservative). The two are
complementary: sandwich for pointwise, this for simultaneous.

Procedure. Fit the full estimator once on all n (per group) to get eta_hat_n(x). Then for
b = 1..B draw a subsample of size m < n WITHOUT replacement (independently per group, the
groups being independent samples), refit the ENTIRE procedure on it, and evaluate the same
grid:

    T_b = tau_m * sup_x | eta_hat_m^(b)(x) - eta_hat_n(x) |,   tau_k = k^beta

    band(x) = Phi( eta_hat_n(x) +/- (m/n)^beta * Q_{1-alpha}( sup_x |eta_hat_m^(b) - eta_hat_n| ) )

Everything is done on the PROBIT scale eta = Phi^-1(A) and mapped through Phi only at the
end, so the AUC band is asymmetric and never needs clipping at 0/1 -- see eta_at for why
that matters (65% of Scenario 4's points were clipped when this was done in AUC scale).

THE RATE beta IS NOT ASSUMED TO BE 1/2. The usual textbook form of this band takes
tau_k = sqrt(k), but eta(x) here is a POINTWISE functional of mu_d(x) and sigma_d(x), which a
network estimates at a nonparametric rate strictly slower than sqrt(n) -- assuming 1/2 when
the truth is slower mis-scales the band systematically. So --m_fracs takes SEVERAL
subsample sizes and beta is estimated from them: with S(m) = median_b sup_x |eta_hat_m^(b) -
eta_hat_n| ~ C * m^-beta, a regression of log S(m) on log m has slope -beta. The fitted beta,
its R^2, and the per-m spreads are all written to rate.csv so the assumption can be
inspected rather than trusted. Passing --beta overrides the estimate (use --beta 0.5 to
reproduce the textbook sqrt-n band).

Caveat this method does NOT escape: every subsample refit re-initializes the networks and
re-runs early stopping, so its spread includes the same training noise that
variance_decomposition_diagnostic.py measured at 40-60% of the bootstrap methods' variance
(there is no mitigation for that here). Also, eta_hat_m is fit on less data and so carries MORE approximation bias than eta_hat_n; in a
scenario where bias already dominates (Scenario 4) that inflates the measured spread.
"""
import argparse
import ast
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from data_simulation_reg import load_data
import true_dgp
import ground_truth_auc
from sandwich_delta_ci import fit_single


def eta_at(mu0_model, sigma0_model, mu1_model, sigma1_model, X0_t, X1_t, adaptive_direction=True):
    """eta(x) = |mu1-mu0| / sqrt(v0 + v1) on the row-paired grid (group 0's nets at X0_t,
    group 1's at X1_t), i.e. the PROBIT-scale statistic, with A(x) = Phi(eta(x)).

    Everything downstream works on this scale, not on the AUC scale, deliberately: a band
    built symmetric in AUC has to be clipped to [0,1], and at these scenarios that is not a
    rare edge case -- 65% of Scenario 4's evaluation points had their upper limit clipped
    at exactly 1.0 in the first run, which makes "covered from above" automatic and inflates
    the measured coverage. Building the band on eta and mapping through Phi gives an
    asymmetric AUC band that never needs clipping, and matches how coverage_sandwich.py
    already does it, so the two methods stay comparable.

    NOTE v0/v1 are the variance networks' RAW outputs, which are already sigma^2 (they are
    trained on SQUARED residuals) -- the denominator is sqrt(v0+v1), NOT sqrt(v0^2+v1^2).
    Getting this wrong inflates eta by ~8.5% at these scenarios' sigmas; see
    sandwich_delta_ci.
    """
    with torch.no_grad():
        mu0 = mu0_model(X0_t).view(-1)
        mu1 = mu1_model(X1_t).view(-1)
        v0 = torch.clamp(sigma0_model(X0_t).view(-1), min=0.0)
        v1 = torch.clamp(sigma1_model(X1_t).view(-1), min=0.0)
        denom = torch.sqrt(torch.clamp(v0 + v1, min=1e-12))
        diff = torch.abs(mu1 - mu0) if adaptive_direction else (mu1 - mu0)
        eta = (diff / denom).cpu().numpy()
    return eta


def fit_pair(X0, Y0, W0, X1, Y1, W1, mlp_config, device):
    """The full two-stage estimator for both groups: returns the four fitted networks."""
    mu0 = fit_single(X0, Y0, W0, mlp_config, device)
    mu1 = fit_single(X1, Y1, W1, mlp_config, device)
    r0 = ((Y0.ravel() - mu0(torch.tensor(X0, dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()) ** 2).reshape(-1, 1).astype(np.float32)
    r1 = ((Y1.ravel() - mu1(torch.tensor(X1, dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()) ** 2).reshape(-1, 1).astype(np.float32)
    s0 = fit_single(X0, r0, W0, mlp_config, device)
    s1 = fit_single(X1, r1, W1, mlp_config, device)
    return mu0, s0, mu1, s1


def subsample_deviations(X0, Y0, W0, X1, Y1, W1, mlp_config, device, m, B, eta_full,
                          X0_eval_t, X1_eval_t, rng):
    """B subsample refits at size m (without replacement, independently per group).
    Returns the B sup-deviations sup_x |eta_hat_m^(b)(x) - eta_hat_n(x)|, on the PROBIT
    scale (see eta_at for why not the AUC scale).
    """
    n0, n1 = X0.shape[0], X1.shape[0]
    sups = np.empty(B)
    for b in range(B):
        i0 = rng.choice(n0, size=m, replace=False)
        i1 = rng.choice(n1, size=m, replace=False)
        mu0_b, s0_b, mu1_b, s1_b = fit_pair(X0[i0], Y0[i0], W0[i0], X1[i1], Y1[i1], W1[i1], mlp_config, device)
        eta_b = eta_at(mu0_b, s0_b, mu1_b, s1_b, X0_eval_t, X1_eval_t)
        sups[b] = np.max(np.abs(eta_b - eta_full))
    return sups


def estimate_rate(ms, sups_by_m):
    """beta from S(m) = median_b sup-deviation ~ C * m^-beta: slope of log S on log m is
    -beta. Returns (beta, r2, medians). Needs at least two m values.
    """
    med = np.array([np.median(s) for s in sups_by_m])
    logm, logs = np.log(np.array(ms, dtype=float)), np.log(med)
    if len(ms) < 2:
        return np.nan, np.nan, med
    slope, intercept = np.polyfit(logm, logs, 1)
    pred = slope * logm + intercept
    ss_res = np.sum((logs - pred) ** 2)
    ss_tot = np.sum((logs - logs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return -slope, r2, med


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" > Device: {device}")

    mlp_config = {k: config[k] for k in
                  ["hidden_layers", "dropout", "batch_size", "learning_rate", "weight_decay",
                   "num_epochs", "early_stop_patience"]}
    target, var_to_group = "Y_generated", "mortstat"
    ci_level = config["ci_level"]

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
        selected = df.columns[df.columns.str.contains("x_D")].tolist()
        print(f"Running subsampling band for {file}")

        name = os.path.splitext(os.path.basename(f))[0]
        sceminario = name
        mt = re.match(r"^(scenario_\d+_\d+_\d+)_data$", name)
        if mt:
            sceminario = mt.group(1)
        scenario_num = int(re.match(r"^scenario_(\d+)_", f).group(1))

        # Resumable: a long sequential run (100 replicates x 9 scenarios) that dies part
        # way should pick up where it left off, not restart. --overwrite forces a redo.
        op_check = f'{config["output_file"]}/{sceminario}/coverage.csv'
        if os.path.exists(op_check) and not config["overwrite"]:
            print(f"   {op_check} ya existe -- saltando")
            continue

        gt_cols = true_dgp.covariate_columns(scenario_num)
        X0_true = df.loc[df[var_to_group] == 0, gt_cols].to_numpy(dtype=float)
        X1_true = df.loc[df[var_to_group] == 1, gt_cols].to_numpy(dtype=float)
        _, d0, d1 = load_data(file, selected, target, var_to_group)
        X0, Y0, W0 = d0["x"], d0["y"], d0["w"]
        X1, Y1, W1 = d1["x"], d1["y"], d1["w"]
        n = X0.shape[0]
        assert n == X1.shape[0], "X0/X1 must be row-paired"

        rng_gt = np.random.default_rng(ground_truth_auc.seed_from_name(sceminario))
        true_auc = ground_truth_auc.true_auc(scenario_num, X0_true, X1_true, rng=rng_gt)

        rng = np.random.default_rng(config["seed"])
        idx = np.sort(rng.choice(n, size=min(config["n_points"], n), replace=False))
        X0_eval_t = torch.tensor(X0[idx], dtype=torch.float32).to(device)
        X1_eval_t = torch.tensor(X1[idx], dtype=torch.float32).to(device)

        # --- Full-sample estimator: the band's centre. ---
        t0 = time.perf_counter()
        mu0, s0, mu1, s1 = fit_pair(X0, Y0, W0, X1, Y1, W1, mlp_config, device)
        eta_full = eta_at(mu0, s0, mu1, s1, X0_eval_t, X1_eval_t)
        auc_full = norm.cdf(eta_full)
        fit_seconds = time.perf_counter() - t0

        # --- Subsample refits at each m. ---
        ms = sorted({max(50, int(round(fr * n))) for fr in config["m_fracs"]})
        Bs = [config["bootstrap_reps"] if m == ms[-1] else config["rate_reps"] for m in ms]
        sups_by_m, t_sub0 = [], time.perf_counter()
        for m, B in zip(ms, Bs):
            tm = time.perf_counter()
            sups = subsample_deviations(X0, Y0, W0, X1, Y1, W1, mlp_config, device, m, B,
                                         eta_full, X0_eval_t, X1_eval_t, rng)
            sups_by_m.append(sups)
            print(f"   m={m:5d} (m/n={m/n:.2f}, B={B}): median sup dev={np.median(sups):.4f}  [{time.perf_counter()-tm:.0f}s]")
        sub_seconds = time.perf_counter() - t_sub0

        beta_hat, r2, medians = estimate_rate(ms, sups_by_m)
        beta = config["beta"] if config["beta"] is not None else beta_hat
        print(f"   rate: beta_hat={beta_hat:.3f} (R^2={r2:.3f}) -> using beta={beta:.3f}"
              f"{' (forced)' if config['beta'] is not None else ''};  sqrt-n convention would be 0.500")

        # --- Band from the largest m, scaled by (m/n)^beta. ---
        m_band, sups_band = ms[-1], sups_by_m[-1]
        q = np.quantile(sups_band, ci_level)
        half_width = (m_band / n) ** beta * q   # on the eta (probit) scale
        # Map through Phi: asymmetric in AUC, automatically inside [0,1], no clipping.
        lower, upper = norm.cdf(eta_full - half_width), norm.cdf(eta_full + half_width)
        covered = (lower <= true_auc[idx]) & (true_auc[idx] <= upper)
        simultaneous = bool(covered.all())
        print(f"   half-width={half_width:.4f} | cobertura puntual dentro de la banda simultanea="
              f"{covered.mean():.3f} | banda simultanea cubre TODOS los {len(idx)} puntos: {simultaneous}")

        op = f'{config["output_file"]}/{sceminario}'
        os.makedirs(op, exist_ok=True)
        pd.DataFrame({
            "Scenario": sceminario, "Method": f"FNN-Subsampling-Band{round(ci_level*100)}",
            "Row Index": idx, "True AUC": true_auc[idx], "Estimated AUC Mean": auc_full,
            "CI Lower": lower, "CI Upper": upper, "Covered": covered.astype(float),
            "Half Width Eta": half_width, "Beta": beta, "Simultaneous Covered": float(simultaneous),
        }).to_csv(f"{op}/coverage.csv", index=False)
        pd.DataFrame({"m": ms, "m_over_n": [m / n for m in ms], "B": Bs,
                      "median_sup_dev": medians,
                      "q_level": [np.quantile(s, ci_level) for s in sups_by_m],
                      "beta_hat": beta_hat, "r2": r2, "beta_used": beta}).to_csv(f"{op}/rate.csv", index=False)
        pd.DataFrame([{"Scenario": sceminario, "Method": "FNN-Subsampling",
                       "Fit Seconds": fit_seconds, "Subsample Seconds": sub_seconds,
                       "Fit Seconds Total": fit_seconds + sub_seconds,
                       "N Points": len(idx), "m_band": m_band}]).to_csv(f"{op}/timing.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simultaneous AUC band by subsampling, with a convergence-rate check.")
    p.add_argument("-i", "--input_dir", default="./data/simulation")
    p.add_argument("-o", "--output_file", default="./output")
    p.add_argument("-b", "--bootstrap_reps", type=int, default=100, help="B en el m mayor (el que construye la banda)")
    p.add_argument("-rr", "--rate_reps", type=int, default=40, help="B en los m auxiliares (solo para estimar la tasa)")
    p.add_argument("-mf", "--m_fracs", type=str, default="0.1,0.2,0.4",
                   help="Fracciones m/n, separadas por coma. Varias => se estima beta; una sola => hay que dar --beta.")
    p.add_argument("-be", "--beta", type=float, default=None,
                   help="Fuerza la tasa (0.5 = convencion sqrt-n). Por defecto se estima de los m.")
    p.add_argument("-e", "--num_epochs", type=int, default=800)
    p.add_argument("-es", "--early_stop_patience", type=int, default=40)
    p.add_argument("-bs", "--batch_size", type=int, default=32)
    p.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    p.add_argument("-wd", "--weight_decay", type=float, default=1e-4)
    p.add_argument("-dr", "--dropout", type=float, default=0.2)
    p.add_argument("-hl", "--hidden_layers", type=ast.literal_eval, default="[64, 32, 16]")
    p.add_argument("-ci", "--ci_level", type=float, default=0.95)
    p.add_argument("-np", "--n_points", type=int, default=100, help="Puntos del grid donde se evalua el sup")
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("-pt", "--pattern", type=str, default=r"_5000_.*_data\.csv$",
                   help="Regex sobre el nombre de fichero; por defecto solo n=5000.")
    p.add_argument("-ow", "--overwrite", action="store_true", help="Recalcula aunque exista coverage.csv")
    a = p.parse_args()

    main({"input_dir": a.input_dir, "output_file": a.output_file, "bootstrap_reps": a.bootstrap_reps,
          "rate_reps": a.rate_reps, "m_fracs": [float(x) for x in a.m_fracs.split(",")], "beta": a.beta,
          "num_epochs": a.num_epochs, "early_stop_patience": a.early_stop_patience, "batch_size": a.batch_size,
          "learning_rate": a.learning_rate, "weight_decay": a.weight_decay, "dropout": a.dropout,
          "hidden_layers": a.hidden_layers, "ci_level": a.ci_level, "n_points": a.n_points, "seed": a.seed, "pattern": a.pattern, "overwrite": a.overwrite})
