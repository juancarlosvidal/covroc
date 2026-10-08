"""Local k-NN confidence bands for the covariate-adjusted AUC on the NHANES cohort.

This is the real-data counterpart of src/simulation/coverage_local_knn.py. That script
measures coverage against a known ground truth and so cannot be pointed at real data; the
interval construction itself is identical and is imported from it rather than reimplemented.

The network is used only to place a neighbourhood. For a target covariate profile x, each
arm is summarized by (mu_hat(x), sigma_hat(x)); the k nearest subjects of each arm in that
two-dimensional summary form the local sample; and the conditional AUC is the Mann-Whitney
statistic over those k x k pairs, with a DeLong interval and a Wilson fallback where the
DeLong variance degenerates. No distributional assumption about the biomarker enters the
interval.

TWO DELIBERATE DEPARTURES FROM THE SIMULATION
---------------------------------------------
1. Cross-fitting instead of a 50/50 split. The simulation can afford to fit on half the
   data and localize in the other half, because both arms have n = 5,000 there. Here the
   case arm has 53-234 subjects depending on sex and horizon, so a half-split would leave
   26 cases for the 3-year female cohort -- a k = 10 neighbourhood would then be 38% of the
   entire arm, neighbourhoods at different ages would overlap almost completely, and the
   band would be a marginal AUC wearing a conditional label. K-fold cross-fitting keeps the
   property that matters (a subject's coordinates come from a model that never saw its
   outcome, so neighbourhood selection cannot be correlated with the outcome being measured)
   while using every subject.

   The cost is that coordinates come from K different networks and are therefore not on
   exactly the same scale. We standardize once over the pooled representations rather than
   per fold, and the grid point -- a covariate profile with no outcome to protect -- is
   placed using the average of the K networks' predictions.

2. The arms are not row-paired. The simulation pairs x_0i with x_1i; here the arms are
   separate cohorts of different sizes, and both are evaluated at the same covariate
   profile, which is the ordinary covariate-specific AUC.

DIRECTION. Higher TAC is protective: decedents have systematically lower TAC. The
orientation is read off the fitted mean functions at the target profile, exactly as in the
simulation, so the flip fires on its own and the reported value is above 0.5 wherever the
marker discriminates at all. This reproduces the 1 - AUC convention stated in the manuscript
without a separate hard-coded correction.

Usage
-----
    python3 src/real_data/local_knn_bands_real.py -g tres -s female -o output/knn_bands_real
    python3 src/real_data/local_knn_bands_real.py --all -o output/knn_bands_real
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "simulation"))

from data_reg_real import normalize_column                      # noqa: E402
from sandwich_delta_ci import fit_single                        # noqa: E402
from coverage_local_knn import (delong_auc_var, auc_interval,          # noqa: E402
                                residualized_local_auc, gaussian_auc)

GENDER_FILES = {"female": "df_f.csv", "male": "df_m.csv"}
HORIZONS = {"tres": "3-year", "cinco": "5-year", "ocho": "8-year"}
TARGET_COL = "TAC2"
COVARIATES = ["RIDAGEYR.x", "BMI"]      # age, BMI -- the paper's stratified-by-sex model
AGE_COL = "RIDAGEYR.x"


def build_inputs(df, covariates, target_col, transform="log"):
    """Normalized covariate matrix and target, matching the notebook's preprocessing.

    `transform` applies a monotone map to the biomarker before the networks see it. This
    does not change the estimand: the AUC is invariant under any strictly increasing
    transformation of the marker, and the Mann-Whitney statistic is computed from ranks.
    It matters numerically. Raw TAC has a median of ~700 and a maximum of 45,105 -- the two
    extreme male subjects already documented in the response to Reviewer 1's Minor Concern 3
    -- so their squared residuals are of order 1e9 and dominate the variance network's loss
    entirely, which fails to converge (training loss ~2.7e11, falling by 0.001% per epoch).
    On the log scale the location-scale model is also a better description of a positive,
    right-skewed activity count.
    """
    cols = [np.expand_dims(normalize_column(df[c]), 1) for c in covariates]
    X = np.hstack(cols).astype("float32")
    y = df[target_col].values.astype("float64")
    if transform == "log":
        y = np.log(np.clip(y, 1e-6, None))
    elif transform == "standardize":
        y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
    elif transform != "none":
        raise ValueError(f"unknown transform {transform!r}")
    Y = y.astype("float32").reshape(-1, 1)
    W = (df["weights"].astype("float32").values if "weights" in df.columns
         else np.ones(len(df), dtype="float32")).reshape(-1, 1)
    return X, Y, W


def crossfit_representation(X, Y, W, n_folds, mlp_config, device, seed):
    """Out-of-fold (mu_hat, sigma_hat) for every subject, plus the fitted fold models.

    Each subject's pair comes from the fold model that did not see it, so the coordinates
    used to choose its neighbours carry no information about its own outcome. Returns the
    raw (unstandardized) representation; standardization is done once by the caller over
    the pooled array, since per-fold standardization would put each fold on its own scale.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(n) % n_folds
    rep = np.full((n, 2), np.nan, dtype=float)
    models = []
    for f in range(n_folds):
        tr = np.where(fold_of != f)[0]
        te = np.where(fold_of == f)[0]
        if len(te) == 0 or len(tr) < 10:
            continue
        mu = fit_single(X[tr], Y[tr], W[tr], mlp_config, device)
        Xtr_t = torch.tensor(X[tr], dtype=torch.float32).to(device)
        with torch.no_grad():
            resid = (Y[tr].ravel() - mu(Xtr_t).view(-1).cpu().numpy()) ** 2
        sg = fit_single(X[tr], resid.reshape(-1, 1).astype(np.float32), W[tr], mlp_config, device)
        Xte_t = torch.tensor(X[te], dtype=torch.float32).to(device)
        with torch.no_grad():
            m = mu(Xte_t).view(-1).cpu().numpy()
            v = np.clip(sg(Xte_t).view(-1).cpu().numpy(), 0.0, None)
        rep[te, 0] = m
        rep[te, 1] = np.sqrt(v)
        models.append((mu, sg))
    return rep, models


def predict_mean_sd(models, X_t):
    """Average the fold models' (mu, sigma) at given profiles.

    Used only for grid targets. A grid point is a covariate profile, not a subject: it has
    no outcome entering the Mann-Whitney statistic, so there is nothing for cross-fitting to
    protect there, and averaging the K models is simply the most stable placement available.
    """
    ms, ss = [], []
    for mu, sg in models:
        with torch.no_grad():
            ms.append(mu(X_t).view(-1).cpu().numpy())
            ss.append(np.sqrt(np.clip(sg(X_t).view(-1).cpu().numpy(), 0.0, None)))
    return np.mean(ms, axis=0), np.mean(ss, axis=0)


def age_grid(df, covariates, n_grid, lo_q, hi_q):
    """Grid of covariate profiles: age varies over an inner quantile range, BMI at median.

    The inner quantile range rather than the full observed range keeps the targets inside
    the support, where a neighbourhood is meaningful (Reviewer 2's Major Comment 7).
    """
    ages = np.linspace(df[AGE_COL].quantile(lo_q), df[AGE_COL].quantile(hi_q), n_grid)
    raw = np.zeros((n_grid, len(covariates)))
    for j, c in enumerate(covariates):
        raw[:, j] = ages if c == AGE_COL else df[c].median()
    # Normalize with the same per-column statistics the training inputs used.
    norm_cols = []
    for j, c in enumerate(covariates):
        col = df[c].to_numpy(dtype=float)
        lo, hi = np.nanmin(col), np.nanmax(col)
        rngc = hi - lo
        norm_cols.append((raw[:, j] - lo) / rngc if rngc > 0 else np.zeros(n_grid))
    return raw, np.column_stack(norm_cols).astype("float32")


def run_one(group_col, gender, args, device):
    path = os.path.join(args.input_dir, GENDER_FILES[gender])
    df = pd.read_csv(path).dropna(subset=COVARIATES + [TARGET_COL, group_col]).reset_index(drop=True)

    X, Y, W = build_inputs(df, COVARIATES, TARGET_COL, args.transform)
    i0 = np.where(df[group_col].to_numpy() == 0)[0]
    i1 = np.where(df[group_col].to_numpy() == 1)[0]
    n0, n1 = len(i0), len(i1)
    print(f"\n=== {HORIZONS[group_col]} / {gender}: N={len(df)}  controls={n0}  cases={n1} ===")
    kmax = max(args.k_values)
    if n1 < 4 * kmax:
        print(f"  !! only {n1} cases: k={kmax} is {100*kmax/n1:.0f}% of the arm. "
              f"Bands reported, but the largest k is barely local here.")

    mlp_config = {"hidden_layers": args.hidden_layers, "dropout": args.dropout,
                  "num_epochs": args.num_epochs, "batch_size": args.batch_size,
                  "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                  "early_stop_patience": args.early_stop_patience}

    t0 = time.perf_counter()
    rep0, models0 = crossfit_representation(X[i0], Y[i0], W[i0], args.n_folds, mlp_config, device, args.seed)
    rep1, models1 = crossfit_representation(X[i1], Y[i1], W[i1], args.n_folds, mlp_config, device, args.seed + 1)
    fit_seconds = time.perf_counter() - t0

    ok0 = np.isfinite(rep0).all(axis=1)
    ok1 = np.isfinite(rep1).all(axis=1)
    rep0, y0 = rep0[ok0], Y[i0].ravel()[ok0]
    rep1, y1 = rep1[ok1], Y[i1].ravel()[ok1]

    # One standardization over the pooled representations of each arm (see module docstring).
    sc0 = np.where(rep0.std(axis=0) > 0, rep0.std(axis=0), 1.0)
    sc1 = np.where(rep1.std(axis=0) > 0, rep1.std(axis=0), 1.0)
    rep0n, rep1n = rep0 / sc0, rep1 / sc1

    raw, gnorm = age_grid(df, COVARIATES, args.grid_points, args.lo_q, args.hi_q)
    g_t = torch.tensor(gnorm, dtype=torch.float32).to(device)
    gm0, gs0 = predict_mean_sd(models0, g_t)
    gm1, gs1 = predict_mean_sd(models1, g_t)
    flip = gm1 < gm0          # decedents have lower TAC -> fires, giving the protective orientation

    # Convergence check on the second stage, reported because it is what the parametric AUC
    # is most sensitive to and what fails silently: the variance network's target is of order
    # 1e4-1e5 here, and with a short budget it never leaves its initialization.
    emp_sd0 = float(Y[i0].std()); emp_sd1 = float(Y[i1].std())
    print(f"  sigma check | cases: fitted {gs1.mean():.3g} vs observed {emp_sd1:.3g} "
          f"({gs1.mean()/max(emp_sd1,1e-9):.2f}x) | controls: {gs0.mean():.3g} vs {emp_sd0:.3g} "
          f"({gs0.mean()/max(emp_sd0,1e-9):.2f}x)")
    if gs1.mean() < 0.5 * emp_sd1:
        print("  !! the variance network has NOT converged; the parametric AUC below is inflated")

    # Parametric (Gaussian location-scale) AUC, for contrast with the rank-based band.
    a_par = np.where(gm1 > gm0, (gm1 - gm0) / (gs1 + 1e-8), (gm0 - gm1) / (gs0 + 1e-8))
    b_par = np.where(gm1 > gm0, gs0 / (gs1 + 1e-8), gs1 / (gs0 + 1e-8))
    auc_par = norm.cdf(a_par / np.sqrt(1 + b_par ** 2))

    if args.localize == "covariate":
        # Neighbourhood directly in (age, BMI), each scaled to unit sd so neither dominates.
        cov0 = np.column_stack([df[c].to_numpy(float)[i0][ok0] for c in COVARIATES])
        cov1 = np.column_stack([df[c].to_numpy(float)[i1][ok1] for c in COVARIATES])
        scv = np.array([df[c].std() for c in COVARIATES], dtype=float)
        scv = np.where(scv > 0, scv, 1.0)
        rep0n, rep1n = cov0 / scv, cov1 / scv
        q0 = q1 = raw / scv
    else:
        q0 = np.column_stack([gm0, gs0]) / sc0
        q1 = np.column_stack([gm1, gs1]) / sc1

    ages0 = df[AGE_COL].to_numpy()[i0][ok0]
    ages1 = df[AGE_COL].to_numpy()[i1][ok1]

    # Para el estimador residualizado: mu y var por sujeto (fuera de pliegue) y en la rejilla.
    if args.estimator == "residualized":
        mu0s, va0s = rep0[:, 0], rep0[:, 1] ** 2
        mu1s, va1s = rep1[:, 0], rep1[:, 1] ** 2
        gq0, gs0b = predict_mean_sd(models0, g_t)
        gq1, gs1b = predict_mean_sd(models1, g_t)
        gv0, gv1 = gs0b ** 2, gs1b ** 2

    rows = []
    for k in [min(kk, len(rep0n), len(rep1n)) for kk in args.k_values]:
      for t in range(len(raw)):
        d0 = np.linalg.norm(rep0n - q0[t], axis=1)
        d1 = np.linalg.norm(rep1n - q1[t], axis=1)
        o0, o1 = np.argsort(d0)[:k], np.argsort(d1)[:k]
        # How local is this neighbourhood, in the units the figure is drawn in?
        nb_ages = np.r_[ages0[o0], ages1[o1]]
        span = float(np.ptp(nb_ages))
        iqr = float(np.subtract(*np.percentile(nb_ages, [75, 25])))
        centre_err = float(abs(np.median(nb_ages) - raw[t, 0]))
        if args.estimator == "residualized":
            if flip[t]:      # el brazo 0 hace de positivo
                iA, yA, mA, vA = o0, y0, mu0s, va0s
                iB, yB, mB, vB = o1, y1, mu1s, va1s
                a1 = gaussian_auc(gq0[t], gv0[t], gq1[t], gv1[t])
            else:
                iA, yA, mA, vA = o1, y1, mu1s, va1s
                iB, yB, mB, vB = o0, y0, mu0s, va0s
                a1 = gaussian_auc(gq1[t], gv1[t], gq0[t], gv0[t])
            auc, var, _, _ = residualized_local_auc(
                yA[iA], yB[iB], mA[iA], vA[iA], mB[iB], vB[iB], float(a1))
        elif flip[t]:
            auc, var = delong_auc_var(y1[o1], y0[o0])
        else:
            auc, var = delong_auc_var(y0[o0], y1[o1])
        if not np.isfinite(auc) or not np.isfinite(var):
            continue
        rec = {"Horizon": HORIZONS[group_col], "Sex": gender, "Age": float(raw[t, 0]),
               "AUC": float(auc), "AUC Parametric": float(auc_par[t]), "k": k, "N Controls": len(rep0n), "N Cases": len(rep1n),
               "Case Arm Fraction": float(k / len(rep1n)), "Age Span": span,
               "Age IQR": iqr, "Age Centre Error": centre_err,
               "Flipped": bool(flip[t])}
        for lv in args.ci_levels:
            z = norm.ppf(1 - (1 - lv) / 2)
            lo, hi, se, fell = auc_interval(auc, var, k, k, z, "logit")
            tag = f"{round(lv*100)}"
            rec[f"CI{tag} Lower"] = lo
            rec[f"CI{tag} Upper"] = hi
            rec[f"CI{tag} Fallback"] = float(fell)
        rec["SE"] = se
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    dest = os.path.join(args.output_dir, f"bands_{group_col}_{gender}.csv")
    out.to_csv(dest, index=False)
    print(f"  fit {fit_seconds:.1f}s | {len(out)} rows -> {dest}")
    for k, g in out.groupby("k"):
        hw = ((g["CI95 Upper"] - g["CI95 Lower"]) / 2).mean()
        print(f"    k={k:>3} ({100*k/g['N Cases'].iloc[0]:>4.1f}% of cases, "
              f"age IQR {g['Age IQR'].mean():>4.1f} yr, off-centre by "
              f"{g['Age Centre Error'].mean():>4.1f} yr): "
              f"AUC {g.AUC.min():.2f}-{g.AUC.max():.2f}  half-width {hw:.3f}  "
              f"roughness {g.AUC.diff().abs().mean():.3f}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input_dir", default="./data")
    p.add_argument("-o", "--output_dir", default="./output/knn_bands_real")
    p.add_argument("-g", "--group_col", choices=list(HORIZONS), default=None)
    p.add_argument("-s", "--sex", choices=list(GENDER_FILES), default=None)
    p.add_argument("--all", action="store_true", help="run all six horizon/sex combinations")
    p.add_argument("-k", "--k_list", type=str, default="20,30,50",
                   help="neighbours per arm; comma-separated sweeps several from ONE fit. "
                        "Small k is noisy and wide, large k stops being local -- the "
                        "Age Span column reports how many years each neighbourhood covers.")
    p.add_argument("-gp", "--grid_points", type=int, default=21)
    p.add_argument("--lo_q", type=float, default=0.025)
    p.add_argument("--hi_q", type=float, default=0.975)
    p.add_argument("-nf", "--n_folds", type=int, default=5)
    p.add_argument("-est", "--estimator", choices=["plain", "residualized"],
                   default="plain", help="ver src/simulation/coverage_local_knn.py")
    p.add_argument("-loc", "--localize", choices=["covariate", "model"], default="covariate",
                   help="where the neighbourhood is taken. 'model': in (mu_hat, sigma_hat), as "
                        "the simulation does -- necessary there because the covariate space is "
                        "up to 4-dimensional. 'covariate' (default here): directly in the "
                        "covariates. With only age and BMI the dimensionality argument for "
                        "'model' does not apply, and 'model' was measured to leave the "
                        "neighbourhood 14 years off its target age against 2 for 'covariate', "
                        "because mu_hat varies far less along the age grid than it does between "
                        "subjects in this cohort.")
    p.add_argument("-ci", "--ci_level", type=str, default="0.95,0.8,0.5")
    p.add_argument("-hl", "--hidden_layers", type=str, default="[64,32]")
    p.add_argument("-e", "--num_epochs", type=int, default=3000)
    p.add_argument("-es", "--early_stop_patience", type=int, default=200)
    p.add_argument("-bs", "--batch_size", type=int, default=200)
    p.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
    p.add_argument("-wd", "--weight_decay", type=float, default=1e-4)
    p.add_argument("-dr", "--dropout", type=float, default=0.2)
    p.add_argument("--transform", choices=["log", "standardize", "none"], default="log",
                   help="monotone map applied to the biomarker before fitting; does not "
                        "change the AUC, which is rank-based (default: log)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    a.ci_levels = tuple(float(x) for x in a.ci_level.split(","))
    a.hidden_layers = [int(x) for x in a.hidden_layers.strip("[]").split(",")]
    a.k_values = [int(x) for x in str(a.k_list).split(",")]

    if not a.all and (a.group_col is None or a.sex is None):
        p.error("give both -g and -s, or --all")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device} | k={a.k_values} | folds={a.n_folds} | "
          f"grid={a.grid_points} | transform={a.transform}")

    combos = ([(g, s) for g in HORIZONS for s in GENDER_FILES] if a.all
              else [(a.group_col, a.sex)])
    parts = [run_one(g, s, a, device) for g, s in combos]
    if a.all:
        allout = pd.concat([p_ for p_ in parts if len(p_)], ignore_index=True)
        dest = os.path.join(a.output_dir, "bands_all.csv")
        allout.to_csv(dest, index=False)
        print(f"\ncombined -> {dest}  ({len(allout)} rows)")


if __name__ == "__main__":
    main()
