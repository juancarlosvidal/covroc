"""Bootstrap + K-fold cross-fitting + out-of-bag (OOB) confidence intervals for the
covariate-adjusted AUC, built on the same two-stage estimator (separate mean model, then
a second model on squared residuals) used everywhere else in this repository for the FNN
results -- mlp_reg_data_simulation_multi.py's MLP/train_model/compute_mean/compute_std/
compute_residues, reused here rather than a separately-trained joint-likelihood
heteroscedastic network, so the confidence bands quantify uncertainty in the *same*
estimator whose point estimates are reported everywhere else (Table 1, Figures 2-18).

This also fixes the in-sample residual-variance bias Reviewer 2's Major Comment 3 raises
("discuss possible bias in the residual variance estimate"): the point-estimate scripts
compute residuals on the same data the mean model was trained on; _fit_group_bootstrap
below computes them only on each cross-fitting fold's held-out validation rows.

Two entry points sharing the inner per-group-per-replicate fit:
  bootstrap_crossfit_oob_shared -- real-data style (notebooks/nhanes_hetero_residuos.ipynb):
    one dataset split into two groups, evaluated at a single shared covariate profile x
    (aROC(p|x) per the paper's formalism). X_eval must be row-aligned with X/group (i.e.
    typically X_eval is X itself).
  bootstrap_crossfit_oob_paired -- simulation style (coverage_bootstrap_crossfit.py):
    two independently-drawn covariate pools of equal length, evaluated row-paired
    (X0[i] with X1[i]), matching the convention ground_truth_auc.py and the FNN/RF
    point-estimate scripts already use.
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

from mlp_reg_data_simulation_multi import MLP, train_model, compute_mean, compute_std, compute_residues


def _make_loader(X, y, w, batch_size, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _fit_group_bootstrap(X_pool, Y_pool, W_pool, X_eval, mlp_config, K, device, inner_val_frac=0.15):
    """One bootstrap replicate's cross-fitted two-stage estimator for one group.

    X_pool/Y_pool/W_pool: this replicate's resampled data for the group. X_eval:
    covariate profiles to evaluate the fold-averaged mu(x)/sigma(x) at.

    Returns (mu_eval, sigma_eval, std_residuals): mu_eval/sigma_eval are arrays aligned
    to X_eval, averaged across the K folds' models; std_residuals is the pooled array of
    *out-of-fold* standardized residuals (y_val - mu_val) / sigma_val, genuinely
    out-of-sample since X_val is never touched during that fold's training.
    """
    n = X_pool.shape[0]
    kf = KFold(n_splits=K, shuffle=True)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32).to(device)
    mu_accum = np.zeros(X_eval.shape[0])
    sigma_accum = np.zeros(X_eval.shape[0])
    std_resid_pool = []

    for tr_idx, val_idx in kf.split(np.arange(n)):
        X_tr, Y_tr, W_tr = X_pool[tr_idx], Y_pool[tr_idx], W_pool[tr_idx]
        X_val, Y_val = X_pool[val_idx], Y_pool[val_idx]

        # Inner train/val carve-out of X_tr, purely for train_model's early stopping.
        # X_val (the outer cross-fitting fold) is never used for training either model.
        n_tr = len(X_tr)
        inner_val_n = max(1, int(round(n_tr * inner_val_frac)))
        perm = np.random.permutation(n_tr)
        inner_val_idx, inner_train_idx = perm[:inner_val_n], perm[inner_val_n:]

        # Stage 1: conditional mean.
        mean_model = MLP(X_pool.shape[1], mlp_config["hidden_layers"], mlp_config["dropout"]).to(device)
        mean_model = train_model(
            mean_model,
            _make_loader(X_tr[inner_train_idx], Y_tr[inner_train_idx], W_tr[inner_train_idx], mlp_config["batch_size"]),
            _make_loader(X_tr[inner_val_idx], Y_tr[inner_val_idx], W_tr[inner_val_idx], mlp_config["batch_size"], shuffle=False),
            device, mlp_config["learning_rate"], mlp_config["weight_decay"],
            mlp_config["num_epochs"], mlp_config["early_stop_patience"],
        )

        # Stage 2: conditional variance, trained on Stage 1's squared residuals
        # (computed on the fold's train partition only, not X_val).
        resid_tr = compute_residues({'x': X_tr, 'y': Y_tr}, mean_model)
        var_model = MLP(X_pool.shape[1], mlp_config["hidden_layers"], mlp_config["dropout"]).to(device)
        var_model = train_model(
            var_model,
            _make_loader(X_tr[inner_train_idx], resid_tr[inner_train_idx], W_tr[inner_train_idx], mlp_config["batch_size"]),
            _make_loader(X_tr[inner_val_idx], resid_tr[inner_val_idx], W_tr[inner_val_idx], mlp_config["batch_size"], shuffle=False),
            device, mlp_config["learning_rate"], mlp_config["weight_decay"],
            mlp_config["num_epochs"], mlp_config["early_stop_patience"],
        )

        # Out-of-fold standardized residuals from X_val.
        mu_val = compute_mean({'x': X_val}, mean_model).ravel()
        sigma_val = compute_std({'x': X_val}, var_model).ravel()
        eps = (Y_val.ravel() - mu_val) / np.clip(sigma_val, 1e-6, None)
        std_resid_pool.append(eps)

        # Fold-averaged predictions at the evaluation covariates.
        mean_model.eval()
        var_model.eval()
        with torch.no_grad():
            mu_accum += mean_model(X_eval_t).cpu().numpy().ravel() / K
            sigma_accum += np.sqrt(np.clip(var_model(X_eval_t).cpu().numpy().ravel(), 0, None)) / K

    return mu_accum, sigma_accum, np.concatenate(std_resid_pool)


def compute_empirical_auc(mu0, mu1, s0, s1, eps0, eps1, n_mc=500, adaptive_direction=False):
    """P(T_control > T_case | x) via Monte Carlo over pooled empirical residuals --
    same construction as the paper's Methods (Semi-Parametric Monte Carlo Estimation of
    the Conditional AUC), applied here to the cross-fitted mu/sigma/residuals above.

    adaptive_direction=False (default, real-data/bootstrap_crossfit_oob_shared) always
    reports P(T0 > T1), matching the original notebook's fixed "P(Control > Case)"
    convention for that dataset.

    adaptive_direction=True (bootstrap_crossfit_oob_paired) instead reports, per row,
    P(higher-mean group's draw > lower-mean group's draw) -- i.e. flips to P(T1 > T0)
    wherever mu1 > mu0. This is required to match ground_truth_auc.true_auc's own
    direction convention (via true_dgp/_ab_from_means_stds' `higher_is_1` branching,
    the same adaptive logic mlp_reg_data_simulation_multi.py's own a_predicted/
    b_predicted already use for the point estimates), since which group has the higher
    mean is not fixed across the 9 scenarios -- e.g. Scenario IV's healthy/diseased
    means actually cross over within the covariate range. Without this, comparing a
    fixed-direction P(T0 > T1) against the ground truth's adaptive AUC amounts to
    comparing against 1 - true_AUC whenever mu1 > mu0, which silently collapses
    pointwise coverage towards 0% for any scenario with real separation.
    """
    e0_samp = np.random.choice(eps0, size=n_mc, replace=True)
    e1_samp = np.random.choice(eps1, size=n_mc, replace=True)
    T0_sim = mu0[:, None] + s0[:, None] * e0_samp[None, :]
    T1_sim = mu1[:, None] + s1[:, None] * e1_samp[None, :]
    if adaptive_direction:
        higher_is_1 = (mu1 > mu0)[:, None]
        wins = np.where(higher_is_1, T1_sim > T0_sim, T0_sim > T1_sim).astype(np.float32)
    else:
        wins = (T0_sim > T1_sim).astype(np.float32)
    return wins.mean(axis=1)


def _aggregate_oob(patient_aucs, min_samples=6, ci_level=0.95):
    n = len(patient_aucs)
    auc_mean = np.full(n, np.nan)
    auc_lower = np.full(n, np.nan)
    auc_upper = np.full(n, np.nan)
    n_oob = np.zeros(n, dtype=int)
    alpha = (1 - ci_level) / 2
    for i, samples in enumerate(patient_aucs):
        arr = np.asarray(samples)
        n_oob[i] = len(arr)
        if len(arr) >= min_samples:
            auc_mean[i] = arr.mean()
            auc_lower[i] = np.percentile(arr, 100 * alpha)
            auc_upper[i] = np.percentile(arr, 100 * (1 - alpha))
    return auc_mean, auc_lower, auc_upper, n_oob


def bootstrap_crossfit_oob_shared(X, Y, group, W, X_eval, mlp_config, B, K, device, n_mc=500, ci_level=0.95):
    """Real-data style: X/Y/group/W describe ALL subjects (split into two groups by
    `group`, 0/1); X_eval is evaluated with both groups' models at the same covariate
    profile. X_eval must be row-aligned with X/group (typically X_eval is X itself).

    Returns per-row (aligned to X_eval) auc_mean, auc_ci_lower, auc_ci_upper, n_oob.
    """
    idx0 = np.where(group == 0)[0]
    idx1 = np.where(group == 1)[0]
    X0, Y0, W0 = X[idx0], Y[idx0], W[idx0]
    X1, Y1, W1 = X[idx1], Y[idx1], W[idx1]
    N0, N1 = len(idx0), len(idx1)
    N_eval = X_eval.shape[0]

    patient_aucs = [[] for _ in range(N_eval)]

    for _ in tqdm(range(B), desc="Bootstrap"):
        sel0 = np.random.choice(N0, size=N0, replace=True)
        sel1 = np.random.choice(N1, size=N1, replace=True)

        mu0, sigma0, eps0 = _fit_group_bootstrap(X0[sel0], Y0[sel0], W0[sel0], X_eval, mlp_config, K, device)
        mu1, sigma1, eps1 = _fit_group_bootstrap(X1[sel1], Y1[sel1], W1[sel1], X_eval, mlp_config, K, device)

        auc_all = compute_empirical_auc(mu0, mu1, sigma0, sigma1, eps0, eps1, n_mc=n_mc)

        used0 = set(idx0[sel0].tolist())
        used1 = set(idx1[sel1].tolist())
        for i in range(N_eval):
            is_oob = (i not in used0) if group[i] == 0 else (i not in used1)
            if is_oob:
                patient_aucs[i].append(auc_all[i])

    return _aggregate_oob(patient_aucs, ci_level=ci_level)


def bootstrap_crossfit_oob_paired(X0, Y0, W0, X1, Y1, W1, mlp_config, B, K, device, n_mc=500, ci_level=0.95):
    """Simulation style: X0/Y0/W0 and X1/Y1/W1 are two independently-drawn covariate
    pools of equal length N (e.g. one scenario replicate's healthy/diseased arms),
    evaluated row-paired -- row i's AUC uses the group-0 model at X0[i] and the group-1
    model at X1[i]. Row i is only OOB-eligible for a replicate if *both* its group-0 and
    group-1 source rows were excluded from that replicate's resample.

    Returns per-row (length N, aligned to X0/X1) auc_mean, auc_ci_lower, auc_ci_upper, n_oob.
    """
    N0, N1 = X0.shape[0], X1.shape[0]
    assert N0 == N1, "X0/X1 must be row-paired (same length)"
    N = N0
    patient_aucs = [[] for _ in range(N)]

    for _ in tqdm(range(B), desc="Bootstrap"):
        sel0 = np.random.choice(N0, size=N0, replace=True)
        sel1 = np.random.choice(N1, size=N1, replace=True)

        mu0, sigma0, eps0 = _fit_group_bootstrap(X0[sel0], Y0[sel0], W0[sel0], X0, mlp_config, K, device)
        mu1, sigma1, eps1 = _fit_group_bootstrap(X1[sel1], Y1[sel1], W1[sel1], X1, mlp_config, K, device)

        # adaptive_direction=True: match ground_truth_auc.true_auc's per-row "higher-mean
        # group vs. the other" convention (see compute_empirical_auc's docstring) -- fixes
        # pointwise coverage silently collapsing towards 0% for scenarios with real
        # separation between groups.
        auc_all = compute_empirical_auc(mu0, mu1, sigma0, sigma1, eps0, eps1, n_mc=n_mc, adaptive_direction=True)

        used0 = set(sel0.tolist())
        used1 = set(sel1.tolist())
        for i in range(N):
            if i not in used0 and i not in used1:
                patient_aucs[i].append(auc_all[i])

    return _aggregate_oob(patient_aucs, ci_level=ci_level)
