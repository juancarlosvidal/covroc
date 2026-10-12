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
                                residualized_local_auc, gaussian_auc,
                                weighted_delong_auc_var, kish_neff)

GENDER_FILES = {"female": "df_f{sfx}.csv", "male": "df_m{sfx}.csv"}
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
    elif transform == "logstd":
        # log, then centred and scaled on the POOLED sample (both arms share one
        # centring, so this is an affine transform of the marker and leaves the AUC
        # unchanged). It is a numerical fix, not a modelling choice: Adam's
        # weight_decay penalises every parameter including the output layer's bias,
        # so an uncentred target of ~6.5 is held up against a constant pull toward
        # zero, and the arm with fewer subjects -- always the cases -- has the weaker
        # data term to resist it. Measured on the uncentred fit, mu_hat for the case
        # arm sits 0.07 to 0.67 local SDs below the neighbourhood's own case mean
        # while the control arm is unbiased, which inflates the estimated separation
        # between arms by a median factor of 1.67.
        y = np.log(np.clip(y, 1e-6, None))
        y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
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
        # A seed of its own per network, derived from this call's seed and the fold, so
        # the fit depends on its arguments and not on what ran before it in the process.
        mu = fit_single(X[tr], Y[tr], W[tr], mlp_config, device,
                        seed=1_000_003 * seed + 17 * f + 1)
        Xtr_t = torch.tensor(X[tr], dtype=torch.float32).to(device)
        with torch.no_grad():
            resid = (Y[tr].ravel() - mu(Xtr_t).view(-1).cpu().numpy()) ** 2
        sg = fit_single(X[tr], resid.reshape(-1, 1).astype(np.float32), W[tr], mlp_config,
                        device, seed=1_000_003 * seed + 17 * f + 2)
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


def age_grid(df, covariates, n_grid, lo_q, hi_q, age_min=None, age_max=None):
    """Grid of covariate profiles: age varies over an inner quantile range, BMI at median.

    The inner quantile range rather than the full observed range keeps the targets inside
    the support, where a neighbourhood is meaningful (Reviewer 2's Major Comment 7).
    age_min/age_max override that with explicit endpoints, for reporting over a region
    chosen on the data's support rather than on its quantiles.
    """
    lo = df[AGE_COL].quantile(lo_q) if age_min is None else float(age_min)
    hi = df[AGE_COL].quantile(hi_q) if age_max is None else float(age_max)
    ages = np.linspace(lo, hi, n_grid)
    raw = np.zeros((n_grid, len(covariates)))
    for j, c in enumerate(covariates):
        raw[:, j] = ages if c == AGE_COL else df[c].median()
    # Normalize with the same per-column statistics the training inputs used, which is
    # the z-score StandardScaler applies inside build_inputs -> normalize_column.
    #
    # This used to be a min-max scaling to [0, 1], which is NOT what the networks saw:
    # age spans [-1.65, 1.86] as a z-score against [0, 1] under min-max, and BMI spans
    # [-2.03, 6.97] because of its upper tail. Every grid point was therefore handed to
    # the network as a covariate value other than the one the axis labelled it with, so
    # the plug-in curve described the wrong profiles. Anything model-based computed
    # before this was fixed has to be recomputed; the rank-based band does not, since it
    # never passes through here.
    norm_cols = []
    for j, c in enumerate(covariates):
        col = df[c].to_numpy(dtype=float)
        mu, sd = np.nanmean(col), np.nanstd(col)   # ddof=0, as StandardScaler uses
        norm_cols.append((raw[:, j] - mu) / sd if sd > 0 else np.zeros(n_grid))
    return raw, np.column_stack(norm_cols).astype("float32")


def run_one(group_col, gender, args, device):
    path = os.path.join(args.input_dir,
                        GENDER_FILES[gender].format(sfx=args.file_suffix))
    df = pd.read_csv(path).dropna(subset=COVARIATES + [TARGET_COL, group_col]).reset_index(drop=True)

    X, Y, W = build_inputs(df, COVARIATES, TARGET_COL, args.transform)
    i0 = np.where(df[group_col].to_numpy() == 0)[0]
    i1 = np.where(df[group_col].to_numpy() == 1)[0]
    n0, n1 = len(i0), len(i1)
    print(f"\n=== {HORIZONS[group_col]} / {gender}: N={len(df)}  controls={n0}  cases={n1} ===")
    kmax = max(k1 for _, k1 in args.k_values)
    if n1 < 4 * kmax:
        print(f"  !! only {n1} cases: k={kmax} is {100*kmax/n1:.0f}% of the arm. "
              f"Bands reported, but the largest k is barely local here.")

    mlp_config = {"hidden_layers": args.hidden_layers, "dropout": args.dropout,
                  "num_epochs": args.num_epochs, "batch_size": args.batch_size,
                  "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                  "early_stop_patience": args.early_stop_patience,
                  "decay_biases": not args.no_bias_decay}

    # Optional ensemble over independent restarts. Re-fitting the identical model with a
    # different seed moves the plug-in AUC on this cohort by 0.044 on average (max 0.087),
    # which is pure optimisation variability: same data, same architecture, same
    # hyperparameters. Averaging the fold models over several seeds is the direct way to
    # damp it. It does NOT touch the larger, separate effect of the marker's scale
    # (0.107 between --transform log and logstd), which is a regularisation bias rather
    # than variance and does not average away.
    #
    # predict_mean_sd already averages over whatever model list it is handed, so the
    # ensemble is formed simply by concatenating the per-seed fold models; the per-subject
    # out-of-fold representations are averaged across seeds in the same way.
    def fit_arm(Xa, Ya, Wa, base_seed):
        reps, mods = [], []
        for s in range(args.n_seeds):
            r, m = crossfit_representation(Xa, Ya, Wa, args.n_folds, mlp_config,
                                           device, base_seed + 1000 * s)
            reps.append(r)
            mods += m
        return np.mean(reps, axis=0), mods

    t0 = time.perf_counter()
    rep0, models0 = fit_arm(X[i0], Y[i0], W[i0], args.seed)
    rep1, models1 = fit_arm(X[i1], Y[i1], W[i1], args.seed + 1)
    fit_seconds = time.perf_counter() - t0
    if args.n_seeds > 1:
        print(f"  ensemble of {args.n_seeds} restarts per arm "
              f"({len(models0)} fold models each)")

    ok0 = np.isfinite(rep0).all(axis=1)
    ok1 = np.isfinite(rep1).all(axis=1)
    rep0, y0 = rep0[ok0], Y[i0].ravel()[ok0]
    rep1, y1 = rep1[ok1], Y[i1].ravel()[ok1]
    # The survey weights, which the networks already carry through train_model's loss.
    # Whether the BAND uses them as well is a separate decision: --band_weights.
    wt0 = W[i0].ravel()[ok0].astype(float)
    wt1 = W[i1].ravel()[ok1].astype(float)
    if not args.band_weights:
        wt0, wt1 = np.ones_like(wt0), np.ones_like(wt1)
    elif np.allclose(wt0, wt0[0]) and np.allclose(wt1, wt1[0]):
        print("  !! --band_weights given but every subject carries the same weight; "
              "the band is unchanged. Did you mean --file_suffix _full?")
    print(f"  band weights: {'on' if args.band_weights else 'off'} | "
          f"weight range {min(wt0.min(), wt1.min()):.2f}-{max(wt0.max(), wt1.max()):.2f}")

    # One standardization over the pooled representations of each arm (see module docstring).
    sc0 = np.where(rep0.std(axis=0) > 0, rep0.std(axis=0), 1.0)
    sc1 = np.where(rep1.std(axis=0) > 0, rep1.std(axis=0), 1.0)
    rep0n, rep1n = rep0 / sc0, rep1 / sc1

    raw, gnorm = age_grid(df, COVARIATES, args.grid_points, args.lo_q, args.hi_q,
                          args.age_min, args.age_max)
    g_t = torch.tensor(gnorm, dtype=torch.float32).to(device)
    gm0, gs0 = predict_mean_sd(models0, g_t)
    gm1, gs1 = predict_mean_sd(models1, g_t)
    # Orientation: which arm plays the "higher" role at each grid point.
    #
    # "auto" reads it from the fitted means, as the simulation does. That is safe when the
    # arms are well separated, but on this cohort they are not: re-fitting the identical
    # model on an affine rescaling of the marker -- a transform the AUC cannot see --
    # moves the sign at 20% of the male grid points, and each sign change maps the
    # reported AUC from A to 1-A, a jump of 0.288 on average. The sign is read from the
    # model, so it carries the model's instability into an estimate that is otherwise
    # model-free.
    #
    # Here the direction is known a priori and need not be estimated at all: TAC is
    # protective, so controls sit above cases by construction of the cohort. Fixing it
    # removes that failure mode outright. The simulation keeps "auto" because there the
    # truth is unknown to the estimator and the arms are well separated.
    if args.orientation == "auto":
        flip = gm1 < gm0
    else:
        flip = np.full(len(gm0), args.orientation == "controls-higher", dtype=bool)

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
    # Driven by the same `flip` as the local statistic, so that the two are always
    # reported in the same direction and remain comparable point by point.
    a_par = np.where(flip, (gm0 - gm1) / (gs0 + 1e-8), (gm1 - gm0) / (gs1 + 1e-8))
    b_par = np.where(flip, gs1 / (gs0 + 1e-8), gs0 / (gs1 + 1e-8))
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

    # Optional floor on who may be a neighbour. The networks are still fitted on the whole
    # sample -- this governs the BAND, which is the part that reaches for subjects wherever
    # it can find them. Without it, a neighbourhood centred at 50 is free to pull in
    # twenty-year-olds from the region the cohort cannot support, and the comparison then
    # carries an age difference between the arms rather than holding age fixed. The floor
    # sits below the first grid point on purpose: it bounds how far the neighbourhood may
    # reach down, without emptying the youngest target of its own neighbours.
    if args.neighbour_age_min is not None:
        keep0 = ages0 >= args.neighbour_age_min
        keep1 = ages1 >= args.neighbour_age_min
        print(f"  neighbour pool restricted to age >= {args.neighbour_age_min:g}: "
              f"controls {len(ages0)} -> {int(keep0.sum())}, "
              f"cases {len(ages1)} -> {int(keep1.sum())}")
        rep0, rep0n, y0, wt0, ages0 = (rep0[keep0], rep0n[keep0], y0[keep0],
                                       wt0[keep0], ages0[keep0])
        rep1, rep1n, y1, wt1, ages1 = (rep1[keep1], rep1n[keep1], y1[keep1],
                                       wt1[keep1], ages1[keep1])

    # Para el estimador residualizado: mu y var por sujeto (fuera de pliegue) y en la rejilla.
    if args.estimator == "residualized":
        mu0s, va0s = rep0[:, 0], rep0[:, 1] ** 2
        mu1s, va1s = rep1[:, 0], rep1[:, 1] ** 2
        gq0, gs0b = predict_mean_sd(models0, g_t)
        gq1, gs1b = predict_mean_sd(models1, g_t)
        gv0, gv1 = gs0b ** 2, gs1b ** 2

    # Each arm is clipped to its own size, so a request larger than an arm takes that
    # whole arm. Several requests can then land on the same pair of neighbourhoods --
    # every case-arm k at or above the case count collapses to the whole case arm, where
    # the "local" estimate is the arm's global AUC. Keep each distinct pair once, in the
    # order asked for, and say which requests merged.
    n0a, n1a = len(rep0n), len(rep1n)
    k_eff, merged = [], []
    for kk0, kk1 in args.k_values:
        pair = (min(kk0, n0a), min(kk1, n1a))
        if pair in k_eff:
            merged.append(f"{kk0}:{kk1}")
        else:
            k_eff.append(pair)
    if merged:
        print(f"  k={merged} exceed an arm ({n0a} controls, {n1a} cases) and repeat a "
              f"pair already asked for; dropped. Reporting "
              f"{[f'{a}:{b}' for a, b in k_eff]}")

    rows = []
    for k0, k1 in k_eff:
      for t in range(len(raw)):
        # The case arm is chosen first, because it is the one that cannot be moved: with
        # 53 to 234 cases over a sixty-year age range, the k1 "nearest" cases to a young
        # target are not near it at all.
        d1 = np.linalg.norm(rep1n - q1[t], axis=1)
        o1 = np.argsort(d1)[:k1]

        # Where the controls are taken from. The default follows the grid point, which is
        # what makes the estimate nominally an estimate AT that covariate value. But when
        # the case neighbourhood sits far from the grid point, the two arms are centred at
        # different ages -- 45 against 24 at a target of 21 -- and the comparison then
        # mixes the age difference between the arms into a statistic whose whole purpose
        # is to hold age fixed. --match_controls anchors the controls on the case
        # neighbourhood's own centroid instead, which removes that confounding by
        # construction. What it cannot do is put the estimate back at the grid point: the
        # result describes the age the cases actually occupy, which is reported as
        # "Case Median Age" and is what such a run should be plotted against.
        anchor = rep1n[o1].mean(axis=0) if args.match_controls else q0[t]
        d0 = np.linalg.norm(rep0n - anchor, axis=1)
        o0 = np.argsort(d0)[:k0]
        # How local is this neighbourhood, in the units the figure is drawn in? The
        # pooled figures are kept for continuity with the symmetric runs, but once the
        # arms are sized differently the pooled ones are dominated by whichever arm is
        # larger, so the case arm -- always the binding one here -- is also reported
        # on its own.
        nb_ages = np.r_[ages0[o0], ages1[o1]]
        span = float(np.ptp(nb_ages))
        iqr = float(np.subtract(*np.percentile(nb_ages, [75, 25])))
        centre_err = float(abs(np.median(nb_ages) - raw[t, 0]))
        case_iqr = float(np.subtract(*np.percentile(ages1[o1], [75, 25])))
        case_centre_err = float(abs(np.median(ages1[o1]) - raw[t, 0]))
        case_med_age = float(np.median(ages1[o1]))
        ctrl_med_age = float(np.median(ages0[o0]))
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
            auc, var = weighted_delong_auc_var(y1[o1], y0[o0], wt1[o1], wt0[o0])
        else:
            auc, var = weighted_delong_auc_var(y0[o0], y1[o1], wt0[o0], wt1[o1])
        if not np.isfinite(auc) or not np.isfinite(var):
            continue
        # With unequal weights the k subjects of an arm no longer carry k observations'
        # worth of information, and it is that reduced count, not k, that the saturation
        # fallback has to be told about.
        neff0, neff1 = kish_neff(wt0[o0]), kish_neff(wt1[o1])
        rec = {"Horizon": HORIZONS[group_col], "Sex": gender, "Age": float(raw[t, 0]),
               "AUC": float(auc), "AUC Parametric": float(auc_par[t]),
               # "k" stays the CASE-arm size: it is the binding one, and it keeps the
               # symmetric runs and everything that selects on it unchanged.
               "k": k1, "k Controls": k0, "k Cases": k1,
               "N Controls": len(rep0n), "N Cases": len(rep1n),
               "Case Arm Fraction": float(k1 / len(rep1n)),
               "Control Arm Fraction": float(k0 / len(rep0n)), "Age Span": span,
               "Age IQR": iqr, "Age Centre Error": centre_err,
               "Case Age IQR": case_iqr, "Case Age Centre Error": case_centre_err,
               "Case Median Age": case_med_age, "Control Median Age": ctrl_med_age,
               "Age Mismatch": case_med_age - ctrl_med_age,
               "Flipped": bool(flip[t]),
               "N0 Eff": neff0, "N1 Eff": neff1}

        # Diagnostic: the two inputs the parametric AUC is built from, each paired with
        # the neighbourhood's own non-parametric counterpart. When "AUC Parametric"
        # departs from "AUC", these say whether the discrepancy sits in the numerator
        # (the mean gap) or the denominator (the scale), which the AUC alone cannot
        # distinguish. Nothing downstream uses them.
        rec.update({
            "Mu0 Net": float(gm0[t]), "Mu1 Net": float(gm1[t]),
            "Sigma0 Net": float(gs0[t]), "Sigma1 Net": float(gs1[t]),
            "Mu0 Local": float(np.mean(y0[o0])), "Mu1 Local": float(np.mean(y1[o1])),
            "Sigma0 Local": float(np.std(y0[o0], ddof=1)),
            "Sigma1 Local": float(np.std(y1[o1], ddof=1)),
        })
        for lv in args.ci_levels:
            z = norm.ppf(1 - (1 - lv) / 2)
            # neff0/neff1 are k0/k1 exactly when the band is unweighted.
            lo, hi, se, fell = auc_interval(auc, var, neff0, neff1, z, "logit")
            tag = f"{round(lv*100)}"
            rec[f"CI{tag} Lower"] = lo
            rec[f"CI{tag} Upper"] = hi
            rec[f"CI{tag} Fallback"] = float(fell)
        rec["SE"] = se
        rows.append(rec)

    # The plug-in at every subject's OWN covariates, rather than on the grid at the
    # median BMI. It needs no extra fitting and no renormalisation: X already holds each
    # subject's covariates on the scale the networks were trained on, so the two arms'
    # models are simply evaluated there. The result is one AUC per subject, which is
    # what a scatter against age shows -- and the vertical spread at a given age is the
    # BMI variation the grid curve hides by fixing BMI at its median.
    if args.subject_auc:
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        sm0, ss0 = predict_mean_sd(models0, Xt)
        sm1, ss1 = predict_mean_sd(models1, Xt)
        if args.orientation == "auto":
            sflip = sm1 < sm0
        else:
            sflip = np.full(len(sm0), args.orientation == "controls-higher", dtype=bool)
        sa = np.where(sflip, (sm0 - sm1) / (ss0 + 1e-8), (sm1 - sm0) / (ss1 + 1e-8))
        sb = np.where(sflip, ss1 / (ss0 + 1e-8), ss0 / (ss1 + 1e-8))
        subj = pd.DataFrame({
            "Horizon": HORIZONS[group_col], "Sex": gender,
            AGE_COL: df[AGE_COL].to_numpy(float), "BMI": df["BMI"].to_numpy(float),
            "Arm": df[group_col].to_numpy(int),
            "AUC Parametric": norm.cdf(sa / np.sqrt(1 + sb ** 2)),
            "Mu0 Net": sm0, "Mu1 Net": sm1, "Sigma0 Net": ss0, "Sigma1 Net": ss1,
        })
        os.makedirs(args.output_dir, exist_ok=True)
        sdest = os.path.join(args.output_dir, f"subjects_{group_col}_{gender}.csv")
        subj.to_csv(sdest, index=False)
        print(f"  {len(subj)} per-subject AUCs -> {sdest}")

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
    p.add_argument("-fs", "--file_suffix", default="",
                   help="suffix on the cohort files: \"\" reads df_f.csv/df_m.csv, "
                        "\"_full\" reads df_f_full.csv/df_m_full.csv, which carry the "
                        "survey weights, strata and PSUs from merge_survey_design.py.")
    p.add_argument("--band_weights", action="store_true",
                   help="let the survey weights into the local statistic as well, which "
                        "makes it estimate the population AUC rather than the cohort's. "
                        "The network fit uses them whenever the column is present; this "
                        "flag governs only the band, so the two can be separated.")
    p.add_argument("-o", "--output_dir", default="./output/knn_bands_real")
    p.add_argument("-g", "--group_col", choices=list(HORIZONS), default=None)
    p.add_argument("-s", "--sex", choices=list(GENDER_FILES), default=None)
    p.add_argument("--all", action="store_true", help="run all six horizon/sex combinations")
    p.add_argument("-k", "--k_list", type=str, default="20,30,50",
                   help="neighbours per arm; comma-separated sweeps several from ONE fit. "
                        "Small k is noisy and wide, large k stops being local -- the "
                        "Age Span column reports how many years each neighbourhood "
                        "covers. An entry may be a single number, which sizes both arms "
                        "alike, or CONTROLS:CASES, which sizes them apart -- e.g. "
                        "\"300:30\". The arms here are 2,300 controls against 53-234 "
                        "cases, so the case arm sets both the locality and the variance "
                        "floor; enlarging the control arm costs almost no locality and "
                        "removes the control term S0^2/k0 from the variance, which can "
                        "shrink the interval by at most a factor 1/sqrt(2).")
    p.add_argument("-gp", "--grid_points", type=int, default=21)
    p.add_argument("--lo_q", type=float, default=0.025)
    p.add_argument("--hi_q", type=float, default=0.975)
    p.add_argument("--age_min", type=float, default=None,
                   help="first grid age, overriding --lo_q")
    p.add_argument("--age_max", type=float, default=None,
                   help="last grid age, overriding --hi_q")
    p.add_argument("--neighbour_age_min", type=float, default=None,
                   help="exclude subjects below this age from the k-NN pool of both "
                        "arms. Bounds how far down a neighbourhood may reach; the "
                        "networks are still fitted on the whole sample.")
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
    p.add_argument("--no_bias_decay", action="store_true",
                   help="exempt the biases from the L2 penalty. Off by default so that "
                        "earlier runs reproduce; on, it removes about two fifths of the "
                        "downward bias in the fitted case mean, and combines with "
                        "--transform logstd, which removes rather more of it.")
    p.add_argument("--subject_auc", action="store_true",
                   help="also write subjects_<horizon>_<sex>.csv: the plug-in AUC at "
                        "every subject's own covariates, for a scatter against age.")
    p.add_argument("--match_controls", action="store_true",
                   help="take the controls nearest the CASE neighbourhood's centroid "
                        "rather than nearest the grid point, so the two arms are centred "
                        "on the same covariate value. Removes the age mismatch between "
                        "the arms; in exchange the estimate describes \"Case Median Age\" "
                        "rather than the grid age, and should be plotted against it.")
    p.add_argument("-ci", "--ci_level", type=str, default="0.95,0.8,0.5")
    p.add_argument("-hl", "--hidden_layers", type=str, default="[64,32]")
    p.add_argument("-e", "--num_epochs", type=int, default=3000)
    p.add_argument("-es", "--early_stop_patience", type=int, default=200)
    p.add_argument("-bs", "--batch_size", type=int, default=200)
    p.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
    p.add_argument("-wd", "--weight_decay", type=float, default=1e-4)
    p.add_argument("-dr", "--dropout", type=float, default=0.2)
    p.add_argument("-ns", "--n_seeds", type=int, default=1,
                   help="independent restarts to average over, per arm. 1 (default) "
                        "reproduces a single fit. Cost scales linearly.")
    p.add_argument("--orientation",
                   choices=["auto", "controls-higher", "cases-higher"],
                   default="auto",
                   help="which arm is taken to sit higher on the marker. 'auto' reads "
                        "it per grid point from the fitted means, as the simulation "
                        "does; the two fixed settings impose it a priori. The default "
                        "is 'controls-higher' because TAC is protective, so the "
                        "direction is known and does not need estimating -- see the "
                        "comment at the flip assignment for why estimating it here is "
                        "unstable.")
    p.add_argument("--transform",
                   choices=["log", "logstd", "standardize", "none"], default="log",
                   help="monotone map applied to the biomarker before fitting; does not "
                        "change the AUC, which is rank-based (default: log)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="'auto' takes the best available. Pin it to 'cpu' when the run "
                        "has to be reproducible across machines: the same seed on CPU "
                        "and on GPU does not give the same weights.")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads; 0 leaves torch's own default. Under SLURM, "
                        "pass $SLURM_CPUS_PER_TASK.")
    a = p.parse_args()

    a.ci_levels = tuple(float(x) for x in a.ci_level.split(","))
    a.hidden_layers = [int(x) for x in a.hidden_layers.strip("[]").split(",")]
    # "k" sizes both arms alike; "k0:k1" sizes the control and case arms apart.
    a.k_values = []
    for item in str(a.k_list).split(","):
        item = item.strip()
        if ":" in item:
            s0, s1 = item.split(":", 1)
            a.k_values.append((int(s0), int(s1)))
        else:
            a.k_values.append((int(item), int(item)))

    if not a.all and (a.group_col is None or a.sex is None):
        p.error("give both -g and -s, or --all")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.device != "auto":
        device = torch.device(a.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Small networks over a few thousand rows: more threads than cores is slower, not
    # faster, and under SLURM it also oversubscribes the allocation.
    if device.type == "cpu" and a.threads:
        torch.set_num_threads(a.threads)
    print(f"device: {device} | k(controls:cases)="
          f"{[f'{x}:{y}' for x, y in a.k_values]} | folds={a.n_folds} | "
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
