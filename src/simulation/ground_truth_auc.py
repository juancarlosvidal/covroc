"""Ground-truth (population-level) covariate-adjusted ROC/AUC for the nine
simulation scenarios, built from the exact data-generating process in true_dgp.py.
Used by the *_reg_data_simulation_multi* scripts to evaluate each estimator (FNN,
Random Forest, ...) against the true aROC(p|x) instead of only the raw-biomarker MSE
(Reviewer 1, Major Concern 1).

For the eight Gaussian location-scale scenarios, aROC(p|x) has the closed form used
throughout the paper: 1 - Phi(b(x)*Phi^-1(1-p) - a(x)) (true_dgp.true_std is constant
for all of these except Scenario III, whose std is a designed function of x -- see
true_dgp.py). Scenario VII's healthy (D=0) arm is a skew-normal/Student-t mixture with
no closed form, so its ROC curve is instead estimated from Monte Carlo draws of the true
DGP via the same empirical-CDF construction the roc() estimator in the
*_reg_data_simulation* scripts uses on residuals -- applied here to raw Y draws from the
known population distribution.
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF

import true_dgp

DEFAULT_N_MC = 20000
DEFAULT_P_GRID = np.linspace(0.001, 0.999, 100)


def _roc_from_ab(a, b, p):
    return 1 - norm.cdf(norm.ppf(1 - p) * b - a)


def roc_from_samples(y0, y1, p):
    """Empirical-CDF ROC curve estimate from raw samples of Y in each group (no
    covariate adjustment). Public since it's also reused, unmodified, by
    naive_roc_baseline.py's pooled/marginal ROC estimator."""
    y0 = np.sort(np.asarray(y0))
    ecdf0 = ECDF(y0)
    inv_ecdf0 = interp1d(ecdf0(y0), y0, bounds_error=False, fill_value=(y0[0], y0[-1]))
    ecdf1 = ECDF(y1)
    return 1 - ecdf1(inv_ecdf0(1 - p))


def _roc_from_samples_batch(y0, y1, p):
    """Row-batched equivalent of calling roc_from_samples(y0[i], y1[i], p) once per row
    -- numerically identical (verified to within float64 rounding, ~1e-16) to that loop,
    but avoids constructing an ECDF + interp1d Python object per row. Used by
    true_roc_curve's Scenario VII branch, where that loop was the dominant cost (one call
    per subject, each over n_mc=20000 Monte Carlo draws).

    y0, y1: (n, n_mc) arrays of MC draws, one row per subject. p: (len_p,) grid.
    Returns (n, len_p).
    """
    n, n_mc = y0.shape
    y0_sorted = np.sort(y0, axis=1)
    y1_sorted = np.sort(y1, axis=1)

    # ecdf0(y0_sorted) evaluated at its own (sorted, tie-free for continuous MC draws)
    # data points is just rank / n_mc -- identical across rows, so the interpolation
    # x-grid (and therefore the bracket indices/weights below) is shared by every row;
    # only the y0_sorted *values* at those brackets differ row to row.
    f_vals = np.arange(1, n_mc + 1) / n_mc
    q = 1 - p
    idx = np.clip(np.searchsorted(f_vals, q, side="left"), 1, n_mc - 1)
    lo, hi = idx - 1, idx
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.clip((q - f_vals[lo]) / (f_vals[hi] - f_vals[lo]), 0.0, 1.0)
    quantiles = y0_sorted[:, lo] + weight[None, :] * (y0_sorted[:, hi] - y0_sorted[:, lo])
    # bounds_error=False, fill_value=(y0[0], y0[-1]) equivalent: clip queries outside
    # the achievable probability range to the min/max sample instead of extrapolating.
    quantiles[:, q < f_vals[0]] = y0_sorted[:, [0]]
    quantiles[:, q > f_vals[-1]] = y0_sorted[:, [-1]]

    # ecdf1 evaluated at each row's own quantiles still needs a per-row pass (both the
    # reference array and the query points vary by row), but it's a single cheap
    # searchsorted call per row now instead of building two ECDF objects and an
    # interp1d -- negligible next to the sorts above.
    ecdf1_vals = np.empty((n, len(p)))
    for i in range(n):
        ecdf1_vals[i] = np.searchsorted(y1_sorted[i], quantiles[i], side="right") / n_mc
    return 1 - ecdf1_vals


def _ab_from_means_stds(mean0, mean1, std0, std1):
    epsilon = 1e-8
    higher_is_1 = mean1 > mean0
    a = np.where(higher_is_1, (mean1 - mean0) / (std1 + epsilon), (mean0 - mean1) / (std0 + epsilon))
    b = np.where(higher_is_1, std0 / (std1 + epsilon), std1 / (std0 + epsilon))
    return a, b


def true_roc_curve(scenario, X0, X1, p=DEFAULT_P_GRID, n_mc=DEFAULT_N_MC, rng=None):
    """True aROC(p|x) curve(s), one per row of X0/X1 (paired row-wise -- the same
    pairing the predicted a(x)/b(x) arrays use in the *_reg_data_simulation* scripts).
    X0/X1 are (n, k) covariate arrays in true_dgp.covariate_columns(scenario) order.
    Returns an (n, len(p)) array.
    """
    X0 = np.atleast_2d(X0)
    X1 = np.atleast_2d(X1)
    n = X0.shape[0]

    if scenario == 7:
        # Chunked over rows: materializing all n subjects' (n_mc,) MC draws at once (e.g.
        # 20000 x 20000 float64 = ~3.2GB per array, several such arrays alive at once
        # between sample_conditional's output and _roc_from_samples_batch's sorted
        # copies) was observed to OOM-kill an n=20000, --mem=16G job. A few thousand
        # subjects at a time keeps peak memory to a few hundred MB while keeping almost
        # all of the vectorized speedup (only the outer loop over chunks is Python-level,
        # not one iteration per subject).
        chunk_size = 2000
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            y0_chunk = true_dgp.sample_conditional(scenario, 0, X0[start:end], n_mc=n_mc, rng=rng)
            y1_chunk = true_dgp.sample_conditional(scenario, 1, X1[start:end], n_mc=n_mc, rng=rng)
            chunks.append(_roc_from_samples_batch(y0_chunk, y1_chunk, p))
        return np.concatenate(chunks, axis=0)

    mean0 = true_dgp.true_mean(scenario, 0, X0)
    mean1 = true_dgp.true_mean(scenario, 1, X1)
    std0 = true_dgp.true_std(scenario, 0, X0)
    std1 = true_dgp.true_std(scenario, 1, X1)
    a, b = _ab_from_means_stds(mean0, mean1, std0, std1)
    return np.array([_roc_from_ab(a[i], b[i], p) for i in range(n)])


def true_auc(scenario, X0, X1, p=DEFAULT_P_GRID, n_mc=DEFAULT_N_MC, rng=None):
    """True covariate-adjusted AUC(x), one per row of X0/X1 (integrates true_roc_curve)."""
    curves = true_roc_curve(scenario, X0, X1, p=p, n_mc=n_mc, rng=rng)
    return np.array([simpson(curve, p) for curve in curves])
