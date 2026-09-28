"""Parametric bootstrap + cross-fitting confidence bands for the covariate-adjusted AUC,
as an alternative to bootstrap_crossfit_oob.py's nonparametric bootstrap-OOB approach.
Both share the same two-stage estimator (_fit_group_bootstrap: mean model, then a second
model on squared residuals) and the same Monte Carlo AUC estimator (compute_empirical_auc),
reused here unmodified.

The key difference from bootstrap_crossfit_oob.py: instead of resampling *subjects* with
replacement, each replicate keeps every subject's covariates X_i fixed and redraws only
the response,

    Y_i*^(b) = mu_hat(X_i) + sigma_hat(X_i) * Z_i*^(b),   Z_i*^(b) ~ N(0, 1) iid,

from the location-scale model with mu_hat/sigma_hat fitted ONCE on the original (real)
data before the bootstrap loop starts. Each replicate then reruns cross-fitting from
scratch on (X_i, Y_i*^(b)) to get mu_hat*^(b)/sigma_hat*^(b), so the bootstrap captures
estimation uncertainty in mu/sigma, not just resampling variability. Because X_i never
leaves the design across replicates, there is no "out-of-bag" subset -- every subject
gets exactly B replicate draws, and the percentile CI just uses all of them.

The reported point estimate, auc_point, is the AUC from that ORIGINAL (non-bootstrapped)
fit, NOT the mean of the B replicate AUCs (see _aggregate_bootstrap) -- the bootstrap is
used only to build the percentile CI around auc_point, the same textbook bootstrap-CI
convention bootstrap_crossfit_oob.py's OOB percentiles also follow around their own
per-subject OOB mean. The replicate mean is still returned (as auc_boot_mean) purely as a
bootstrap-bias diagnostic (auc_boot_mean - auc_point), never as the reported curve.

Two ways to turn a replicate's mu/sigma into a replicate AUC -- note BOTH use the same
Gaussian parametric bootstrap above to generate Y*; the distinction is only in how AUC is
computed from a replicate's refit mu/sigma, not in whether the bootstrap itself is
parametric:
  bootstrap_crossfit_parametric_* -- semiparametric: draws simulated T0/T1 Monte Carlo
    samples from the pooled empirical out-of-fold residuals of that replicate's refit
    (compute_empirical_auc, same estimator bootstrap_crossfit_oob.py uses), so the AUC
    formula itself does not assume the residuals are Gaussian.
  bootstrap_crossfit_gaussian_*   -- fully parametric: assumes the location-scale
    residuals are genuinely Gaussian, so AUC(x) = P(Y1 > Y0 | x) has the closed form
    Phi((mu1-mu0)/sqrt(sigma0^2+sigma1^2)), replacing the Monte Carlo step entirely. NOTE
    this is not the same as reporting AUC_hat +/- 1.96*SE: Phi is only used to map
    (mu, sigma) -> AUC within each bootstrap replicate; the sampling distribution of
    AUC_hat(x) across replicates is still whatever it is, and the percentile CI over the B
    replicate AUCs makes no further normality assumption about it.

Two entry points per method, mirroring bootstrap_crossfit_oob.py's shared/paired split:
  *_shared -- real-data style (notebooks/nhanes_hetero_residuos.ipynb): one dataset split
    into two groups, evaluated at a single shared covariate profile x (aROC(p|x)).
  *_paired -- simulation style (coverage_bootstrap_crossfit.py): two independently-drawn
    covariate pools of equal length, evaluated row-paired (X0[i] with X1[i]).

Known limitation inherited from bootstrap_crossfit_oob._fit_group_bootstrap, shared by
every function in this file: its second-stage (variance) model is trained on residuals
computed by evaluating the fold's mean model on that SAME fold's training partition
(compute_residues({'x': X_tr, ...}, mean_model)) rather than on a further nested
out-of-fold split, so those training targets are somewhat in-sample w.r.t. the mean model
-- a real, but pre-existing and unmodified-here, source of possible sigma underestimation.
"""
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from bootstrap_crossfit_oob import _fit_group_bootstrap, compute_empirical_auc


def _gaussian_auc(mu0, mu1, s0, s1, adaptive_direction=False):
    """Closed-form AUC(x) = P(Y1 > Y0 | x) under Y0|x ~ N(mu0,s0^2), Y1|x ~ N(mu1,s1^2)
    independent given x, i.e. D = Y1 - Y0 | x ~ N(mu1-mu0, s0^2+s1^2) and AUC = P(D>0).

    adaptive_direction=False matches compute_empirical_auc's default convention (fixed
    P(T0 > T1)); adaptive_direction=True matches its adaptive convention (P(higher-mean
    group's draw > lower-mean group's draw), flipping per row wherever mu1 > mu0).
    """
    denom = np.sqrt(np.clip(s0 ** 2 + s1 ** 2, 1e-12, None))
    if adaptive_direction:
        z = np.abs(mu1 - mu0) / denom
    else:
        z = (mu0 - mu1) / denom
    return norm.cdf(z)


def _aggregate_bootstrap(auc_point, auc_reps, ci_levels=(0.95,)):
    """auc_point: (N_eval,) AUC from the ORIGINAL (non-bootstrapped) cross-fitted fit --
    this, not the replicates' mean, is the reported point estimate (module docstring).

    auc_reps: (B, N_eval) array, one row per bootstrap replicate, used only for the
    percentile CIs and the bootstrap-bias diagnostic below. Unlike
    bootstrap_crossfit_oob._aggregate_oob, every column has exactly B draws (no OOB
    filtering), so this is a plain percentile interval.

    ci_levels: iterable of confidence levels (e.g. (0.95, 0.8)) to compute percentile CIs
    for, all from this SAME auc_reps array -- i.e. at no extra bootstrap-replicate cost,
    since the B replicates only need to be fit once regardless of how many levels are
    requested afterward. Useful for calibration checks (does empirical coverage at 80%
    track its nominal level as well as at 95%?) without re-running the expensive fit.

    Returns per-row auc_point, ci_bounds, n_reps, auc_boot_mean -- ci_bounds is a dict
    {level: (auc_ci_lower, auc_ci_upper)} with one entry per requested level; n_reps is
    just B, repeated (kept in the tuple so callers can reuse the same coverage.csv layout
    bootstrap_crossfit_oob.py's n_oob column uses); auc_boot_mean is E*[AUC*(x)], for
    diagnosing bootstrap bias via (auc_boot_mean - auc_point), not a substitute for
    auc_point.
    """
    arr = np.asarray(auc_reps)
    ci_bounds = {}
    for level in ci_levels:
        alpha = (1 - level) / 2
        ci_bounds[level] = (np.percentile(arr, 100 * alpha, axis=0), np.percentile(arr, 100 * (1 - alpha), axis=0))
    n_reps = np.full(arr.shape[1], arr.shape[0], dtype=int)
    auc_boot_mean = arr.mean(axis=0)
    return np.asarray(auc_point), ci_bounds, n_reps, auc_boot_mean


def _replicate_responses(mu_hat, sigma_hat):
    """mu_hat/sigma_hat come back from _fit_group_bootstrap already .ravel()'d to 1D, but
    the rest of the pipeline (data_simulation_reg.create_dict, _fit_group_bootstrap's own
    Y_pool[tr_idx] indexing alongside a (M,1) model output in compute_residues) treats Y
    as a (N,1) column -- reshape here so the regenerated response matches that convention.
    """
    return (mu_hat + sigma_hat * np.random.standard_normal(mu_hat.shape[0])).reshape(-1, 1)


def bootstrap_crossfit_parametric_shared(X, Y, group, W, X_eval, mlp_config, B, K, device, n_mc=500, ci_levels=(0.95,)):
    """Real-data style, Monte Carlo AUC. See module docstring; X_eval must be row-aligned
    with X/group (typically X_eval is X itself). Returns per-row (aligned to X_eval)
    auc_point, ci_bounds, n_reps, auc_boot_mean, auc_reps -- see _aggregate_bootstrap for
    the ci_bounds dict shape; auc_reps is the raw (B, N_eval) bootstrap-replicate AUC
    array, e.g. for a PIT/calibration check ((auc_reps <= true_auc[None,:]).mean(axis=0)
    should be ~Uniform(0,1) across subjects if the bootstrap is well-calibrated).
    """
    idx0 = np.where(group == 0)[0]
    idx1 = np.where(group == 1)[0]
    X0, Y0, W0 = X[idx0], Y[idx0], W[idx0]
    X1, Y1, W1 = X[idx1], Y[idx1], W[idx1]

    # Point-estimate fit on the real data, evaluated at X_eval: this is the reported AUC
    # curve, not the bootstrap replicates' mean below (see module docstring).
    mu0_pt, sigma0_pt, eps0_pt = _fit_group_bootstrap(X0, Y0, W0, X_eval, mlp_config, K, device)
    mu1_pt, sigma1_pt, eps1_pt = _fit_group_bootstrap(X1, Y1, W1, X_eval, mlp_config, K, device)
    auc_point = compute_empirical_auc(mu0_pt, mu1_pt, sigma0_pt, sigma1_pt, eps0_pt, eps1_pt, n_mc=n_mc)

    # Separate fit evaluated at each group's OWN covariates, purely to get mu_hat(X_i)/
    # sigma_hat(X_i) for the Y* generation below -- a small fixed overhead (2 extra
    # cross-fitted refits) relative to the B bootstrap replicates that follow.
    mu0_own, sigma0_own, _ = _fit_group_bootstrap(X0, Y0, W0, X0, mlp_config, K, device)
    mu1_own, sigma1_own, _ = _fit_group_bootstrap(X1, Y1, W1, X1, mlp_config, K, device)

    N_eval = X_eval.shape[0]
    auc_reps = np.empty((B, N_eval))
    for b in tqdm(range(B), desc="Parametric bootstrap"):
        Y0_star = _replicate_responses(mu0_own, sigma0_own)
        Y1_star = _replicate_responses(mu1_own, sigma1_own)

        mu0_b, sigma0_b, eps0_b = _fit_group_bootstrap(X0, Y0_star, W0, X_eval, mlp_config, K, device)
        mu1_b, sigma1_b, eps1_b = _fit_group_bootstrap(X1, Y1_star, W1, X_eval, mlp_config, K, device)

        auc_reps[b] = compute_empirical_auc(mu0_b, mu1_b, sigma0_b, sigma1_b, eps0_b, eps1_b, n_mc=n_mc)

    return (*_aggregate_bootstrap(auc_point, auc_reps, ci_levels=ci_levels), auc_reps)


def bootstrap_crossfit_parametric_paired(X0, Y0, W0, X1, Y1, W1, mlp_config, B, K, device, n_mc=500, ci_levels=(0.95,)):
    """Simulation style, Monte Carlo AUC. X0/Y0/W0 and X1/Y1/W1 are two independently-drawn
    covariate pools of equal length N, evaluated row-paired (X0[i] with X1[i]). Returns
    per-row (length N, aligned to X0/X1) auc_point, ci_bounds, n_reps, auc_boot_mean,
    auc_reps -- see bootstrap_crossfit_parametric_shared's docstring for what auc_reps is
    for (PIT/calibration checks) and _aggregate_bootstrap for the ci_bounds dict shape.
    """
    N0, N1 = X0.shape[0], X1.shape[0]
    assert N0 == N1, "X0/X1 must be row-paired (same length)"

    # This single fit, evaluated at each group's own covariates, does double duty: in the
    # paired convention X_eval is already X0/X1, so it's both the point estimate AND the
    # mu_hat(X_i)/sigma_hat(X_i) used to generate Y* below -- no extra fit needed.
    mu0_hat, sigma0_hat, eps0_hat = _fit_group_bootstrap(X0, Y0, W0, X0, mlp_config, K, device)
    mu1_hat, sigma1_hat, eps1_hat = _fit_group_bootstrap(X1, Y1, W1, X1, mlp_config, K, device)
    auc_point = compute_empirical_auc(mu0_hat, mu1_hat, sigma0_hat, sigma1_hat, eps0_hat, eps1_hat, n_mc=n_mc, adaptive_direction=True)

    auc_reps = np.empty((B, N0))
    for b in tqdm(range(B), desc="Parametric bootstrap"):
        Y0_star = _replicate_responses(mu0_hat, sigma0_hat)
        Y1_star = _replicate_responses(mu1_hat, sigma1_hat)

        mu0_b, sigma0_b, eps0_b = _fit_group_bootstrap(X0, Y0_star, W0, X0, mlp_config, K, device)
        mu1_b, sigma1_b, eps1_b = _fit_group_bootstrap(X1, Y1_star, W1, X1, mlp_config, K, device)

        # adaptive_direction=True: match ground_truth_auc.true_auc's per-row "higher-mean
        # group vs. the other" convention (see bootstrap_crossfit_oob.compute_empirical_auc's
        # docstring for why this matters for pointwise coverage).
        auc_reps[b] = compute_empirical_auc(mu0_b, mu1_b, sigma0_b, sigma1_b, eps0_b, eps1_b, n_mc=n_mc, adaptive_direction=True)

    return (*_aggregate_bootstrap(auc_point, auc_reps, ci_levels=ci_levels), auc_reps)


def bootstrap_crossfit_gaussian_shared(X, Y, group, W, X_eval, mlp_config, B, K, device, ci_levels=(0.95,)):
    """Real-data style, closed-form Gaussian AUC (no Monte Carlo). See module docstring;
    X_eval must be row-aligned with X/group. Returns per-row (aligned to X_eval) auc_point,
    ci_bounds, n_reps, auc_boot_mean, auc_reps -- see bootstrap_crossfit_parametric_shared's
    docstring for what auc_reps is for (PIT/calibration checks) and _aggregate_bootstrap
    for the ci_bounds dict shape.
    """
    idx0 = np.where(group == 0)[0]
    idx1 = np.where(group == 1)[0]
    X0, Y0, W0 = X[idx0], Y[idx0], W[idx0]
    X1, Y1, W1 = X[idx1], Y[idx1], W[idx1]

    mu0_pt, sigma0_pt, _ = _fit_group_bootstrap(X0, Y0, W0, X_eval, mlp_config, K, device)
    mu1_pt, sigma1_pt, _ = _fit_group_bootstrap(X1, Y1, W1, X_eval, mlp_config, K, device)
    auc_point = _gaussian_auc(mu0_pt, mu1_pt, sigma0_pt, sigma1_pt)

    mu0_own, sigma0_own, _ = _fit_group_bootstrap(X0, Y0, W0, X0, mlp_config, K, device)
    mu1_own, sigma1_own, _ = _fit_group_bootstrap(X1, Y1, W1, X1, mlp_config, K, device)

    N_eval = X_eval.shape[0]
    auc_reps = np.empty((B, N_eval))
    for b in tqdm(range(B), desc="Gaussian parametric bootstrap"):
        Y0_star = _replicate_responses(mu0_own, sigma0_own)
        Y1_star = _replicate_responses(mu1_own, sigma1_own)

        mu0_b, sigma0_b, _ = _fit_group_bootstrap(X0, Y0_star, W0, X_eval, mlp_config, K, device)
        mu1_b, sigma1_b, _ = _fit_group_bootstrap(X1, Y1_star, W1, X_eval, mlp_config, K, device)

        auc_reps[b] = _gaussian_auc(mu0_b, mu1_b, sigma0_b, sigma1_b)

    return (*_aggregate_bootstrap(auc_point, auc_reps, ci_levels=ci_levels), auc_reps)


def bootstrap_crossfit_gaussian_paired(X0, Y0, W0, X1, Y1, W1, mlp_config, B, K, device, ci_levels=(0.95,)):
    """Simulation style, closed-form Gaussian AUC (no Monte Carlo). X0/Y0/W0 and X1/Y1/W1
    are two independently-drawn covariate pools of equal length N, evaluated row-paired.
    Returns per-row (length N, aligned to X0/X1) auc_point, ci_bounds, n_reps, auc_boot_mean,
    auc_reps -- see bootstrap_crossfit_parametric_shared's docstring for what auc_reps is
    for (PIT/calibration checks) and _aggregate_bootstrap for the ci_bounds dict shape.
    """
    N0, N1 = X0.shape[0], X1.shape[0]
    assert N0 == N1, "X0/X1 must be row-paired (same length)"

    mu0_hat, sigma0_hat, _ = _fit_group_bootstrap(X0, Y0, W0, X0, mlp_config, K, device)
    mu1_hat, sigma1_hat, _ = _fit_group_bootstrap(X1, Y1, W1, X1, mlp_config, K, device)
    auc_point = _gaussian_auc(mu0_hat, mu1_hat, sigma0_hat, sigma1_hat, adaptive_direction=True)

    auc_reps = np.empty((B, N0))
    for b in tqdm(range(B), desc="Gaussian parametric bootstrap"):
        Y0_star = _replicate_responses(mu0_hat, sigma0_hat)
        Y1_star = _replicate_responses(mu1_hat, sigma1_hat)

        mu0_b, sigma0_b, _ = _fit_group_bootstrap(X0, Y0_star, W0, X0, mlp_config, K, device)
        mu1_b, sigma1_b, _ = _fit_group_bootstrap(X1, Y1_star, W1, X1, mlp_config, K, device)

        # adaptive_direction=True: same rationale as bootstrap_crossfit_parametric_paired.
        auc_reps[b] = _gaussian_auc(mu0_b, mu1_b, sigma0_b, sigma1_b, adaptive_direction=True)

    return (*_aggregate_bootstrap(auc_point, auc_reps, ci_levels=ci_levels), auc_reps)
