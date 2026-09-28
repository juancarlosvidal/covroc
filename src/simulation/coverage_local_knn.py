"""Local k-nearest-neighbour inference for the covariate-adjusted AUC: the neural network
is used ONLY to define a neighbourhood, and the confidence interval comes from held-out
outcomes inside it.

Why this exists. Every method tried before it -- bootstrap-OOB, parametric/gaussian
bootstrap, sandwich/delta (pointwise and averaged), subsampling -- fails through the same
mechanism: the network's estimation error is correlated across x in a way none of them
model correctly, so the reported uncertainty is wrong (measured coverage at nominal 95%
ranged from 0.02 to 0.99). This procedure removes that mechanism instead of modelling it:

  1. Split subjects into TRAIN and TEST.
  2. Fit the four networks (mu_d, sigma_d for d = 0, 1) on TRAIN only.
  3. Map every TEST subject to the low-dimensional summary the network predicts for it,
     (mu_hat_d(x), sigma_hat_d(x)) for that subject's own arm.
  4. For a target row, take the k TEST controls closest to it in the control summary and
     the k TEST cases closest to it in the case summary.
  5. Estimate the AUC in that frozen neighbourhood from the TEST OUTCOMES, empirically
     (Mann-Whitney), with the DeLong variance.

Conditional on the trained network, the test outcomes are fresh, so step 5 is an ordinary
two-sample AUC problem with no network uncertainty in it at all. The ReLU kinks, the
|mu1-mu0| kink, the non-symmetric two-stage Hessian, the stationarity assumption and the
convergence-rate question all become irrelevant.

Two properties worth stating explicitly.

LOCALIZING IN THE REPRESENTATION, NOT IN x, IS WHAT MAKES THIS WORK. AUC(x) depends on x
only through (mu_0, mu_1, sigma_0, sigma_1), so a neighbourhood that is narrow in the
representation is narrow in AUC even when it is wide in x. A simulation over the nine
scenarios put the smoothing bias at 0.00004-0.00034 when the neighbourhood is defined by
the TRUE representation -- essentially zero even in Scenario 7, whose true AUC varies with
sd 0.21 across the covariate range. With the representation perturbed by the estimation
error actually measured in output/sw100, the bias rises to 0.004-0.023, driven by how NOISY
the representation is rather than by how much AUC varies. It also defuses the curse of
dimensionality: scenarios 8 and 9 have four covariates but the representation stays
low-dimensional.

THE ESTIMAND CHANGES, and the interval does not cover the difference. What is estimated is
the AUC of the local population defined by the neighbourhood, not AUC(x_0) exactly. The
gap is the smoothing bias above.

--k_list sweeps k over an ORDER OF MAGNITUDE (default 5,10,15,20,50,100,300,1000) on purpose.
Checking a few nearby values (75/100/125, as one might be tempted to do) measures variance,
not bias: neighbourhoods of similar width share essentially the same smoothing bias and
would agree with each other while all being shifted. Bias only becomes visible as DRIFT in
the estimate as k grows by a large factor, so the k-sweep is the bias diagnostic, and it is
nearly free because the neighbours are sorted once and the statistics accumulated outward.
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


def delong_auc_var(y0, y1):
    """Empirical AUC = P(Y1 > Y0) with the DeLong variance, from raw outcomes.

    Uses the structural components (placement values): V10_j places each case among the
    controls, V01_i each control among the cases; AUC is the mean of either and
    Var = S10/n1 + S01/n0. Nonparametric -- no Gaussian assumption anywhere, which is why
    this survives Scenario 7 (whose healthy arm is a skew-normal/t mixture, and where every
    Phi-based estimator in this codebase is estimating the wrong quantity).
    """
    n0, n1 = len(y0), len(y1)
    if n0 < 2 or n1 < 2:
        return np.nan, np.nan
    y0s = np.sort(y0)
    # For each case: fraction of controls below it (ties count half).
    lo = np.searchsorted(y0s, y1, side="left")
    hi = np.searchsorted(y0s, y1, side="right")
    v10 = (lo + hi) / (2.0 * n0)
    y1s = np.sort(y1)
    lo1 = np.searchsorted(y1s, y0, side="left")
    hi1 = np.searchsorted(y1s, y0, side="right")
    v01 = 1.0 - (lo1 + hi1) / (2.0 * n1)
    auc = float(v10.mean())
    var = float(v10.var(ddof=1) / n1 + v01.var(ddof=1) / n0)
    return auc, max(var, 0.0)


def hanley_mcneil_neff(auc, n0, n1):
    """Effective independent sample size for a Mann-Whitney AUC: n_eff = A(1-A)/Var_HM(A),
    with Var_HM the Hanley-McNeil variance. The pairwise comparisons are dependent, so
    n_eff is neither n0*n1 (far too many) nor min(n0,n1) (too few).

    The first version of the Wilson fallback used min(n0,n1) as a prudent guess. It was too
    prudent: measured against the DeLong variance in the NON-degenerate high-AUC cells of
    output/knn100_iv, the implied n_eff is 2.1-2.6 times k, so that fallback's SE was about
    1.5x wider than it needed to be -- and indeed those cells covered 100% of the time,
    never once failing, which is the signature of an interval that is simply too wide.
    Deriving n_eff from Hanley-McNeil instead gives ~1.9-2.2 times k with no fitted
    constant, and adapts to both k and the AUC level.
    """
    a = min(max(auc, 1e-9), 1 - 1e-9)
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n0 * n1)
    if not np.isfinite(var) or var <= 0:
        return float(min(n0, n1))
    return float(np.clip(a * (1 - a) / var, 1.0, n0 * n1))


def wilson_interval(auc, n_eff, z):
    """Wilson score interval for the AUC treated as a proportion with effective sample size
    n_eff. Used ONLY as the fallback when DeLong degenerates (see auc_interval): at perfect
    separation in the neighbourhood the empirical AUC is exactly 1 and the DeLong variance
    is exactly 0, so the interval collapses to the point [1, 1] and can never cover a true
    AUC below 1 -- measured coverage in those cells is 0.000 by construction, and they were
    9.7% of Scenario 4's and 6.1% of Scenario 7's cells at k=100. Wilson stays interior at
    the boundary (e.g. AUC=1 with n_eff=100 gives roughly [0.963, 1.000]).

    n_eff = min(n0, n1) rather than n0*n1: the pairwise comparisons are strongly dependent,
    so the number of independent units is of the order of the smaller arm, not the number
    of pairs. This is deliberately conservative.
    """
    d = 1.0 + z * z / n_eff
    centre = (auc + z * z / (2 * n_eff)) / d
    half = z * np.sqrt(auc * (1 - auc) / n_eff + z * z / (4 * n_eff * n_eff)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def auc_interval(auc, var, n0, n1, z, mode="logit"):
    """DeLong normal interval where DeLong is well behaved, Wilson fallback where it is not.

    mode="delong" forces the plain interval (what the first run used), so the two can be
    compared on the same fits.
    """
    degenerate = (not np.isfinite(var)) or var <= 0.0 or auc <= 0.0 or auc >= 1.0
    if mode == "delong" or not degenerate:
        se = float(np.sqrt(max(var, 0.0)))
        return max(0.0, auc - z * se), min(1.0, auc + z * se), se, False
    # Laplace-correct the boundary estimate before reading n_eff off Hanley-McNeil:
    # at perfect separation auc is exactly 1, where the HM variance is degenerate too.
    s_max = n0 * n1
    a_tilde = (auc * s_max + 1.0) / (s_max + 2.0)
    lo, hi = wilson_interval(auc, hanley_mcneil_neff(a_tilde, n0, n1), z)
    return lo, hi, (hi - lo) / (2 * z), True


def arm_representation(mu_model, sigma_model, X_t):
    """Per-subject low-dimensional summary for one arm: (mu_hat(x), sigma_hat(x)), each
    standardized so both components weigh comparably in the distance. sigma_hat comes from
    sqrt of the variance network's output, which IS sigma^2 (trained on squared residuals).
    """
    with torch.no_grad():
        mu = mu_model(X_t).view(-1).cpu().numpy()
        var = np.clip(sigma_model(X_t).view(-1).cpu().numpy(), 0.0, None)
    sd = np.sqrt(var)
    out = np.stack([mu, sd], axis=1)
    scale = out.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return out / scale, scale


def arm_representation_at(mu_model, sigma_model, X_t, scale):
    """Same summary evaluated at arbitrary query points, reusing the scale computed on the
    TEST set. Recomputing the scale from a handful of grid points would put queries and
    neighbours on different axes and silently corrupt every distance.
    """
    with torch.no_grad():
        mu = mu_model(X_t).view(-1).cpu().numpy()
        var = np.clip(sigma_model(X_t).view(-1).cpu().numpy(), 0.0, None)
    return np.stack([mu, np.sqrt(var)], axis=1) / scale


def build_fixed_grid(df, gt_cols, n_grid, lo_q=0.025, hi_q=0.975):
    """A grid of FIXED covariate profiles, identical across replicates and scenarios.

    Why fixed x rather than test subjects as targets: with subject targets every replicate
    contributes ~100 correlated indicators (they share one fitted network), so the effective
    sample size for a coverage rate is the REPLICATE count, not the row count -- the trap
    that made earlier tables look far more precise than they were. With a fixed grid each
    point accumulates exactly one independent Bernoulli per replicate, so 100 replicates are
    100 genuine trials, and coverage can be reported AS A FUNCTION OF x. It also removes the
    worry that subject targets are covariates the network already saw during fitting.

    Only the FIRST covariate varies (over an inner quantile range of its observed values, to
    stay inside the support); the rest are held at their median, since a full grid is not
    possible for the two- and four-covariate scenarios.

    Returns (raw, normalized): raw feeds ground_truth_auc, normalized feeds the networks --
    data_simulation_reg.normalize_column is a StandardScaler fitted on the whole column, so
    the same transform is reproduced here exactly.
    """
    raw = np.tile(np.array([df[c].median() for c in gt_cols], dtype=float), (n_grid, 1))
    raw[:, 0] = np.linspace(df[gt_cols[0]].quantile(lo_q), df[gt_cols[0]].quantile(hi_q), n_grid)
    norm_grid = np.empty_like(raw)
    for j, c in enumerate(gt_cols):
        mu, sd = df[c].mean(), df[c].std(ddof=0)
        norm_grid[:, j] = (raw[:, j] - mu) / (sd if sd > 0 else 1.0)
    return raw, norm_grid.astype(np.float32)


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" > Device: {device}")
    mlp_config = {k: config[k] for k in
                  ["hidden_layers", "dropout", "batch_size", "learning_rate", "weight_decay",
                   "num_epochs", "early_stop_patience"]}
    target_col, var_to_group = "Y_generated", "mortstat"
    ci_levels, k_list = config["ci_levels"], config["k_list"]

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
        name = os.path.splitext(os.path.basename(f))[0]
        sceminario = name
        mt = re.match(r"^(scenario_\d+_\d+_\d+)_data$", name)
        if mt:
            sceminario = mt.group(1)
        scenario_num = int(re.match(r"^scenario_(\d+)_", f).group(1))

        op = f'{config["output_file"]}/{sceminario}'
        if os.path.exists(f"{op}/coverage.csv") and not config["overwrite"]:
            print(f"   {op}/coverage.csv ya existe -- saltando")
            continue
        print(f"Running local k-NN coverage for {file}")

        gt_cols = true_dgp.covariate_columns(scenario_num)
        X0_true = df.loc[df[var_to_group] == 0, gt_cols].to_numpy(dtype=float)
        X1_true = df.loc[df[var_to_group] == 1, gt_cols].to_numpy(dtype=float)
        _, d0, d1 = load_data(file, selected, target_col, var_to_group)
        X0, Y0, W0 = d0["x"], d0["y"], d0["w"]
        X1, Y1, W1 = d1["x"], d1["y"], d1["w"]
        n = X0.shape[0]
        assert n == X1.shape[0], "X0/X1 must be row-paired"

        rng_gt = np.random.default_rng(ground_truth_auc.seed_from_name(sceminario))
        true_auc = ground_truth_auc.true_auc(scenario_num, X0_true, X1_true, rng=rng_gt)

        # --- Train/test split (same row split for both arms keeps the paired convention). ---
        rng = np.random.default_rng(config["seed"])
        perm = rng.permutation(n)
        n_tr = int(round(n * config["train_frac"]))
        tr, te = perm[:n_tr], perm[n_tr:]

        t0 = time.perf_counter()
        mu0 = fit_single(X0[tr], Y0[tr], W0[tr], mlp_config, device)
        mu1 = fit_single(X1[tr], Y1[tr], W1[tr], mlp_config, device)
        r0 = ((Y0[tr].ravel() - mu0(torch.tensor(X0[tr], dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()) ** 2).reshape(-1, 1).astype(np.float32)
        r1 = ((Y1[tr].ravel() - mu1(torch.tensor(X1[tr], dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()) ** 2).reshape(-1, 1).astype(np.float32)
        s0 = fit_single(X0[tr], r0, W0[tr], mlp_config, device)
        s1 = fit_single(X1[tr], r1, W1[tr], mlp_config, device)
        fit_seconds = time.perf_counter() - t0

        X0te_t = torch.tensor(X0[te], dtype=torch.float32).to(device)
        X1te_t = torch.tensor(X1[te], dtype=torch.float32).to(device)
        rep0, rep0_scale = arm_representation(mu0, s0, X0te_t)   # (n_te, 2) controls
        rep1, rep1_scale = arm_representation(mu1, s1, X1te_t)   # (n_te, 2) cases
        # Direction, decided by the TRAIN-fitted network and never by the test outcomes.
        # ground_truth_auc.true_auc uses the adaptive convention (higher-mean arm wins, so
        # AUC >= 0.5), but a raw Mann-Whitney is fixed-direction; in Scenario 4 the arm
        # means cross at x = -0.49, so over ~75% of the covariate range the fixed-direction
        # estimate lands near 0 where the adaptive truth is near 1 (measured drift -0.68
        # before this fix). Orienting with the network keeps the estimand consistent WITHOUT
        # peeking at the outcomes -- taking max(auc, 1-auc) instead would choose the
        # direction from the same data that estimates the magnitude and bias |auc-0.5| up.
        with torch.no_grad():
            flip = (mu1(X1te_t).view(-1).cpu().numpy() < mu0(X0te_t).view(-1).cpu().numpy())
        y0te = Y0[te].ravel()
        y1te = Y1[te].ravel()
        n_te = len(te)

        if config["targets"] == "grid":
            # Fixed covariate profiles: SHARED convention, both arms evaluated at the same
            # x_0, which is the standard covariate-specific AUC. NOTE this is a DIFFERENT
            # estimand from the row-paired convention every earlier run used, so those
            # tables are not directly comparable with these.
            grid_raw, grid_norm = build_fixed_grid(df, gt_cols, config["grid_points"])
            gt_grid = ground_truth_auc.true_auc(scenario_num, grid_raw, grid_raw,
                                                 rng=np.random.default_rng(ground_truth_auc.seed_from_name(sceminario)))
            g_t = torch.tensor(grid_norm, dtype=torch.float32).to(device)
            q0 = arm_representation_at(mu0, s0, g_t, rep0_scale)
            q1 = arm_representation_at(mu1, s1, g_t, rep1_scale)
            with torch.no_grad():
                flip = (mu1(g_t).view(-1).cpu().numpy() < mu0(g_t).view(-1).cpu().numpy())
            truths = gt_grid
            row_ids = np.arange(len(grid_raw))
            xvals = grid_raw[:, 0]
        else:
            n_targets = min(config["n_points"], n_te)
            sel = np.sort(rng.choice(n_te, size=n_targets, replace=False))
            q0, q1 = rep0[sel], rep1[sel]
            truths = true_auc[te[sel]]
            row_ids = te[sel]
            xvals = np.full(len(sel), np.nan)
        n_targets = len(row_ids)

        t1 = time.perf_counter()
        rows = []
        for t in range(n_targets):
            d0_ = np.linalg.norm(rep0 - q0[t], axis=1)
            d1_ = np.linalg.norm(rep1 - q1[t], axis=1)
            o0, o1 = np.argsort(d0_), np.argsort(d1_)
            tru = truths[t]
            for k in k_list:
                if k > n_te:
                    continue
                if flip[t]:
                    auc_k, var_k = delong_auc_var(y1te[o1[:k]], y0te[o0[:k]])
                else:
                    auc_k, var_k = delong_auc_var(y0te[o0[:k]], y1te[o1[:k]])
                if not np.isfinite(auc_k) or not np.isfinite(var_k):
                    continue
                for lv in ci_levels:
                    z = norm.ppf(1 - (1 - lv) / 2)
                    lo, hi, se_k, fell_back = auc_interval(auc_k, var_k, k, k, z, config["interval"])
                    rows.append({"Scenario": sceminario, "Method": f"FNN-LocalKNN-k{k}-CI{round(lv*100)}",
                                 "k": k, "Row Index": int(row_ids[t]), "x0": float(xvals[t]), "True AUC": tru,
                                 "Estimated AUC Mean": auc_k, "CI Lower": lo, "CI Upper": hi,
                                 "SE Eta": se_k, "Covered": float(lo <= tru <= hi),
                                 "Fallback": float(fell_back),
                                 "Radius Ctrl": float(d0_[o0[k - 1]]), "Radius Case": float(d1_[o1[k - 1]])})
        ci_seconds = time.perf_counter() - t1

        coverage_df = pd.DataFrame(rows)
        os.makedirs(op, exist_ok=True)
        coverage_df.to_csv(f"{op}/coverage.csv", index=False)
        pd.DataFrame([{"Scenario": sceminario, "Method": "FNN-LocalKNN",
                       "Fit Seconds": fit_seconds, "CI Seconds Total": ci_seconds,
                       "Fit Seconds Total": fit_seconds + ci_seconds,
                       "N Points": n_targets, "N Test": n_te}]).to_csv(f"{op}/timing.csv", index=False)

        print(f"  Fit: {fit_seconds:.1f}s | local: {ci_seconds:.1f}s | n_test={n_te} | targets={n_targets}")
        for k in k_list:
            sub = coverage_df[(coverage_df["k"] == k) & coverage_df["Method"].str.endswith("CI95")]
            if len(sub):
                drift = (sub["Estimated AUC Mean"] - sub["True AUC"]).mean()
                print(f"    k={k:>4}: cobertura@95={sub['Covered'].mean():.3f}  "
                      f"semianchura={(sub['CI Upper'] - sub['CI Lower']).mean()/2:.4f}  "
                      f"deriva(est-true)={drift:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Local k-NN inference for the covariate-adjusted AUC.")
    p.add_argument("-i", "--input_dir", default="./data/simulation")
    p.add_argument("-o", "--output_file", default="./output")
    p.add_argument("-kl", "--k_list", type=str, default="5,10,15,20,50,100,300,1000",
                   help="Tamanos de vecindario. Barrer un ORDEN DE MAGNITUD es el diagnostico de sesgo "
                        "(valores cercanos solo miden varianza) -- ver docstring del modulo.")
    p.add_argument("-tf", "--train_frac", type=float, default=0.5, help="Fraccion para entrenar la red")
    p.add_argument("-np", "--n_points", type=int, default=100, help="Sujetos objetivo por replica")
    p.add_argument("-ci", "--ci_level", type=str, default="0.95,0.8,0.5")
    p.add_argument("-e", "--num_epochs", type=int, default=800)
    p.add_argument("-es", "--early_stop_patience", type=int, default=40)
    p.add_argument("-bs", "--batch_size", type=int, default=32)
    p.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    p.add_argument("-wd", "--weight_decay", type=float, default=1e-4)
    p.add_argument("-dr", "--dropout", type=float, default=0.2)
    p.add_argument("-hl", "--hidden_layers", type=ast.literal_eval, default="[64, 32, 16]")
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("-pt", "--pattern", type=str, default=r"_5000_.*_data\.csv$")
    p.add_argument("-tg", "--targets", choices=["grid", "subjects"], default="grid",
                   help="grid (por defecto): perfiles de covariables FIJOS, identicos en todas las "
                        "replicas -> cada punto acumula ensayos independientes y la cobertura se "
                        "reporta como funcion de x. subjects: sujetos de test, como en las tandas previas.")
    p.add_argument("-gp", "--grid_points", type=int, default=11, help="Puntos de la rejilla fija")
    p.add_argument("-iv", "--interval", choices=["logit", "delong"], default="logit",
                   help="logit (por defecto): DeLong donde se porta bien, Wilson donde degenera. "
                        "delong: intervalo normal siempre, como en la primera tanda.")
    p.add_argument("-ow", "--overwrite", action="store_true")
    a = p.parse_args()

    main({"input_dir": a.input_dir, "output_file": a.output_file,
          "k_list": [int(x) for x in a.k_list.split(",")], "train_frac": a.train_frac,
          "n_points": a.n_points, "ci_levels": tuple(float(x) for x in a.ci_level.split(",")),
          "num_epochs": a.num_epochs, "early_stop_patience": a.early_stop_patience,
          "batch_size": a.batch_size, "learning_rate": a.learning_rate,
          "weight_decay": a.weight_decay, "dropout": a.dropout, "hidden_layers": a.hidden_layers,
          "seed": a.seed, "pattern": a.pattern, "overwrite": a.overwrite, "interval": a.interval,
          "targets": a.targets, "grid_points": a.grid_points})
