"""Single source of truth for the nine simulation scenarios' data-generating process
(DGP). data_generation.py calls into this module to generate data; the ground-truth
AUC/ROC comparison in ground_truth_auc.py calls into it to evaluate the true
conditional mean/variance/distribution the estimators (FNN, RF, ...) are being
compared against. Keeping both uses on the same source avoids the two ever drifting
apart (Reviewer 1, Major Concern 1).

Every scenario models Y | (X=x, D=d) as Normal(true_mean(x), true_std) EXCEPT
Scenario VII's healthy (D=0) arm, which is a skew-normal / Student-t mixture with no
closed-form mean or variance -- true_std raises for that one case, and callers must
use sample_conditional (Monte Carlo) instead.

Covariate columns consumed by each scenario, in the order true_mean/sample_conditional
expect them (matches the column names data_generation.py writes to the scenario CSVs):
    1, 2, 3, 4, 7: ['x_D_1']
    5:             ['x_D_1', 'x_D_2']
    6:             ['x_D_1', 'x_D_3']
    8, 9:          ['x_D_1', 'x_D_6', 'x_D_7', 'x_D_8']
"""
import numpy as np
from scipy.stats import skewnorm, t

_COVARIATE_COLUMNS = {
    1: ['x_D_1'],
    2: ['x_D_1'],
    3: ['x_D_1'],
    4: ['x_D_1'],
    5: ['x_D_1', 'x_D_2'],
    6: ['x_D_1', 'x_D_3'],
    7: ['x_D_1'],
    8: ['x_D_1', 'x_D_6', 'x_D_7', 'x_D_8'],
    9: ['x_D_1', 'x_D_6', 'x_D_7', 'x_D_8'],
}


def covariate_columns(scenario):
    """Raw dataframe column names needed for `scenario`, in positional order."""
    return list(_COVARIATE_COLUMNS[scenario])


def true_mean(scenario, group, X):
    """Deterministic conditional mean mu_d(x). X is (n, k) with the k columns
    returned by covariate_columns(scenario). Raises for Scenario VII group 0,
    which has no closed-form mean (use sample_conditional instead).
    """
    X = np.atleast_2d(X)
    healthy = group == 0

    if scenario == 1:
        return np.full(X.shape[0], 0.5 if healthy else 1.0)

    if scenario == 2:
        x1 = X[:, 0]
        return (0.5 if healthy else 1.0) + (2 * x1 - 10) / 23

    if scenario == 3:
        x1 = X[:, 0]
        if healthy:
            return 0.25 + 0.5 * (2 * x1 - 10) / 23
        return 0.75 + (2 * x1 - 10) / 23

    if scenario == 4:
        x1 = X[:, 0]
        if healthy:
            return (5 + 3 * ((x1 + 8) / 23) ** 2
                    - 25 * (((x1 + 8) / 23) - 0.2) ** 3
                    + 250 * (((x1 + 8) / 23) - 0.65) ** 3)
        return -3 - 0.6 * ((x1 + 8) / 23)

    if scenario == 5:
        x1, x2 = X[:, 0], X[:, 1]
        if healthy:
            return 0.5 * np.exp((2 * x1 - 10) / 10) - 2 * ((2 * x2 ** 2 - 10) / 10)
        return 0.5 * np.sin(np.pi * ((2 * x1 - 10) / 10 + 1)) + 0.5 * np.exp((2 * x1 - 10) / 10)

    if scenario == 6:
        x1, x3 = X[:, 0], X[:, 1]
        if healthy:
            return (-np.sin(0.7 * np.pi * ((2 * x1 - 10) / 10 + 30)) * x3
                    + ((2 * x1 - 10) / 10) ** 2 * (1 - x3))
        return 0.5 + ((2 * x1 - 10) / 10) ** 2

    if scenario == 7:
        x1 = X[:, 0]
        if healthy:
            raise ValueError(
                "Scenario 7 group 0 (healthy) is a skew-normal/t mixture with no "
                "closed-form mean; use sample_conditional() instead."
            )
        return np.sin(2 * np.pi * x1) + 1.5

    if scenario == 8:
        return np.full(X.shape[0], 0.5 if healthy else 1.0)

    if scenario == 9:
        x1, x6, x7, x8 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        base = 0.5 * np.exp(2 * x1) - x6 ** 2 + 0.5 * x7 ** 2 + x8
        return base if healthy else 0.5 + base

    raise ValueError(f"Unknown scenario {scenario}")


def true_std(scenario, group):
    """Constant conditional std sigma_d(x). Returns None for Scenario VII group 0,
    which has no closed-form std (use sample_conditional instead). Scenario VII group 1
    is Normal(mean, 0.5), unlike every other scenario's group 1, which is std 1.0."""
    if scenario == 7:
        return None if group == 0 else 0.5
    return 0.5 if group == 0 else 1.0


def _draw_normal(rng, loc, scale, size):
    if rng is None:
        return np.random.normal(loc, scale, size)
    return rng.normal(loc, scale, size)


def _draw_uniform(rng, size):
    if rng is None:
        return np.random.uniform(size=size)
    return rng.uniform(size=size)


def sample_conditional(scenario, group, X, n_mc=1, rng=None):
    """Monte Carlo draws of Y | X=x, D=group from the *exact* generating
    distribution (matches generate_scenario_* in data_generation.py).

    Returns an (n, n_mc) array: one row per covariate profile in X, n_mc i.i.d.
    draws each. `rng` is either None (uses the global numpy random state, matching
    data_generation.py's own unseeded calls) or a numpy Generator (e.g.
    np.random.default_rng(seed)) for reproducible ground-truth Monte Carlo draws.
    """
    X = np.atleast_2d(X)
    n = X.shape[0]

    if scenario == 7 and group == 0:
        x1 = np.repeat(X[:, 0], n_mc)
        branch = _draw_uniform(rng, n * n_mc) < np.exp(-2 * x1)
        skew = skewnorm.rvs(a=2, loc=x1 ** 2, scale=0.25, size=n * n_mc, random_state=rng)
        tdist = t.rvs(df=5, loc=np.sin(np.pi * x1), scale=0.25, size=n * n_mc, random_state=rng)
        draws = np.where(branch, skew, tdist)
        return draws.reshape(n, n_mc)

    mean = true_mean(scenario, group, X)
    std = true_std(scenario, group)
    return _draw_normal(rng, np.repeat(mean, n_mc), std, n * n_mc).reshape(n, n_mc)
