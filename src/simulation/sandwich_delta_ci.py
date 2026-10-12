"""Sandwich (robust M-estimation) + delta-method confidence bands for the covariate-
adjusted AUC -- a fast, non-bootstrap alternative to bootstrap_crossfit_oob.py/
bootstrap_crossfit_parametric.py. Computed from a SINGLE fit (no retraining, no
resampling), so it measures genuine parameter-estimation uncertainty directly from the
loss's curvature and per-observation score dispersion, rather than bootstrap replicate-to-
replicate spread -- sidestepping entirely the training-noise contamination the variance-
decomposition diagnostic found in the bootstrap methods (see
variance_decomposition_diagnostic.py and project memory).

Unlike bootstrap_crossfit_oob.py/bootstrap_crossfit_parametric.py, this does NOT use
K-fold cross-fitting: theta_hat here must be a single, cleanly-defined M-estimator
(argmin of an empirical risk over ALL of a group's data) for the sandwich/delta theory to
apply -- a fold-averaged ensemble doesn't have an equally standard asymptotic variance
theory. This means the point estimate mu_hat/sigma_hat used by this module can differ
slightly from the cross-fitted point estimate reported elsewhere in the paper; this
module is a complementary fast diagnostic, not a drop-in replacement for the main
point-estimate pipeline. It also reintroduces the in-sample residual-variance bias
bootstrap_crossfit_oob.py's cross-fitting was built to avoid (Reviewer 2 Major Comment 3)
since sigma^2 is now trained on the mean-network's own in-sample residuals -- a known,
accepted trade-off of using a single non-cross-fitted fit here.

Math (Murphy & Topel 1985 / Newey 1984 two-step M-estimation correction, applied to two
sequentially-trained neural networks instead of two closed-form parametric models):

  eta(x) = (mu1(x) - mu0(x)) / sqrt(sigma1(x)^2 + sigma0(x)^2),   AUC(x) = Phi(eta(x))

For one group d, stack theta = (theta_mu, theta_sigma) (stage 1's mean-network params,
stage 2's variance-network params). The stacked per-observation estimating equation is

  psi_i(theta) = ( grad_theta_mu ell_mu_i(theta_mu),
                    grad_theta_sigma ell_sigma_i(theta_sigma; theta_mu) )

ell_sigma_i depends on theta_mu only through the (in-sample) residual r_i(theta_mu) =
y_i - mu(x_i; theta_mu) feeding its target r_i^2. Because stage 1 is estimated first,
ignorant of stage 2, but stage 2 depends on stage 1, the Jacobian of psi is block LOWER
triangular (not symmetric -- standard in two-step M-estimation, nothing to do with
bootstrap_crossfit_parametric.py's H_sigma_mu being some kind of error):

  H = [[H_mumu,        0     ],
       [H_sigmamu,   H_sigmasigma]]

with H_mumu = mean_i hessian_theta_mu ell_mu_i, H_sigmasigma = mean_i
hessian_theta_sigma ell_sigma_i (both TRUE, symmetric Hessians of their own stage's loss),
and H_sigmamu = d/d(theta_mu) [grad_theta_sigma ell_sigma_avg] (a genuine cross term
capturing how stage 1's fit shifts stage 2's fitted optimum -- the "generated regressor"
propagation a naive per-stage-separate sandwich would miss entirely).

H^{-1}a is never formed via an explicit matrix inverse (infeasible for a p ~ 10^3-10^4
parameter network): solved via conjugate gradient using only Hessian-vector products
(autodiff, Pearlmutter's trick), exploiting the block-triangular structure:
  1. CG-solve H_mumu @ v_mu = a_mu
  2. Compute H_sigmamu @ v_mu (a Jacobian-vector product -- forward-over-reverse autodiff,
     torch.func.jvp composed with torch.func.grad)
  3. CG-solve H_sigmasigma @ v_sigma = a_sigma - H_sigmamu @ v_mu
Both CG solves use a damped operator (H + lambda*I) for numerical stability -- note the
network's own weight_decay (L2 penalty) already contributes exactly a lambda*I term to H
by construction (see _damped_hvp), so this isn't purely an ad-hoc numerical trick here.

J = mean_i psi_i psi_i^T uses ONLY the data-dependent part of each score (the L2 penalty
term is a deterministic function of theta, contributing to H's curvature but not to the
across-observations score VARIANCE J is meant to capture -- standard treatment for
penalized/ridge M-estimation robust standard errors).

Finally Var{eta_hat(x)} ~= (1/n) * v_x^T J v_x where v_x = H^{-1} a_x, a_x =
grad_theta eta(x) -- computed group-by-group and SUMMED (group 0 and group 1 are
independent samples, so the joint 4-network H/J is block-diagonal across groups: no cross
term between group 0's and group 1's networks, so their variance contributions to eta(x)
are additive by the delta method).

IMPORTANT, not yet checked here: the sandwich/delta approximation is a first-order Taylor
expansion around a genuine STATIONARY point of the training loss (mean score ~= 0). The
rest of this pipeline's early stopping fires very early (epoch ~15-60 out of an 800
budget, per variance_decomposition_diagnostic.py's real-scale runs) based on a
*validation*-loss plateau, not a *training*-loss gradient norm -- so the returned model
may not be close to a stationary point of the training loss this module differentiates.
sandwich_delta_ci's diagnostic() function reports the mean training score norm at the
fitted theta_hat precisely so this can be checked before trusting a given CI, rather than
assumed.
"""
import numpy as np
import torch
from scipy.stats import norm
from sklearn.model_selection import KFold
from torch.func import functional_call, vmap, grad as func_grad, jvp

from mlp_reg_data_simulation_multi import MLP, train_model


def _make_loader(X, y, w, batch_size, shuffle=True):
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fit_single(X, Y, W, mlp_config, device, val_frac=0.15, seed=None):
    """Fits ONE MLP on ALL of (X, Y, W) -- no cross-fitting/folds. An internal train/val
    split is still used purely for early stopping (same convention as elsewhere in the
    pipeline), but the RETURNED model is a single, well-defined M-estimator fit on the
    full sample, which the sandwich/delta theory requires.

    seed=None (the default) draws the split, the initialisation and the dropout masks from
    the PROCESS-WIDE generators. That makes a fit depend on how many draws preceded it, so
    the same data with the same --seed gives different parameters according to what else
    ran first in the same process: in a --all pass over six NHANES combinations, the fifth
    one is fitted from whatever state the previous four left behind. Measured on the
    cohort, that moved the plug-in AUC by up to 0.062 between an --all run and the same
    combination run alone.

    Passing a seed makes the fit depend on nothing but its arguments. The default is left
    as it was so that every existing caller -- the whole simulation -- reproduces exactly;
    only callers that opt in change, and they change from irreproducible to reproducible.
    """
    n = X.shape[0]
    if seed is None:
        perm = np.random.permutation(n)
    else:
        # Both generators, because the split comes from numpy while the initialisation
        # and the dropout masks come from torch; seeding one and not the other leaves the
        # fit order-dependent through the other.
        perm = np.random.default_rng(seed).permutation(n)
        torch.manual_seed(seed)
    n_val = max(1, int(round(n * val_frac)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    model = MLP(X.shape[1], mlp_config["hidden_layers"], mlp_config["dropout"]).to(device)
    model = train_model(
        model,
        _make_loader(X[train_idx], Y[train_idx], W[train_idx], mlp_config["batch_size"]),
        _make_loader(X[val_idx], Y[val_idx], W[val_idx], mlp_config["batch_size"], shuffle=False),
        device, mlp_config["learning_rate"], mlp_config["weight_decay"],
        mlp_config["num_epochs"], mlp_config["early_stop_patience"],
        # Read from the config rather than taken as an argument, so a caller that does
        # not set it -- every simulation one -- keeps the original behaviour untouched.
        decay_biases=mlp_config.get("decay_biases", True),
    )
    model.eval()
    return model


def oof_squared_residuals(X, Y, W, mlp_config, K, device):
    """Genuinely out-of-fold squared residuals for a group: K auxiliary mean-network fits,
    each predicting only the fold it never saw. Returns an (n, 1) float32 array suitable as
    the variance network's training target.

    Why this exists: training the variance network on the full-data mean network's own
    IN-SAMPLE residuals biases sigma_hat DOWNWARD (the mean net partly fits the noise), so
    eta = |mu1-mu0| / sqrt(sigma0^2+sigma1^2) comes out too large and the AUC too extreme.
    Measured on the real-scale sandwich runs: eta_hat - eta_true was positive in all six
    replicates (+0.037 to +0.297), between 1.0 and 2.4 times the reported standard error,
    which is what was destroying coverage -- the variance estimate itself was accurate
    wherever the stationarity assumption held (see project memory).

    Cost: K extra mean-network fits per group (the variance net is still fit once). At the
    measured ~10-30s per network fit at n=5000 that is roughly +100-300s per replicate on
    top of the ~1-3 min the in-sample variant takes.

    NOTE the theoretical consequence, handled by GroupSandwichContext(cross_term=False):
    the stage-2 loss no longer depends on theta_mu at all (its targets now come from the
    auxiliary fold networks, not from the full-data theta_mu), so the H_sigma_mu cross term
    is ZERO by construction and H becomes block DIAGONAL and symmetric. That makes v == w
    in the sandwich (only 2 CG solves instead of 4), but it also means the auxiliary
    networks' own estimation uncertainty is NOT propagated -- treating the OOF residuals as
    fixed data is an approximation, and one that understates the variance somewhat.
    """
    n = X.shape[0]
    kf = KFold(n_splits=K, shuffle=True)
    oof_pred = np.zeros(n, dtype=np.float64)
    for tr_idx, val_idx in kf.split(np.arange(n)):
        fold_model = fit_single(X[tr_idx], Y[tr_idx], W[tr_idx], mlp_config, device)
        with torch.no_grad():
            X_val_t = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
            oof_pred[val_idx] = fold_model(X_val_t).cpu().numpy().ravel()
    return ((Y.ravel() - oof_pred) ** 2).reshape(-1, 1).astype(np.float32)


def _flatten(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def _params_list(model):
    return list(model.parameters())


def per_sample_losses(model, X_t, Y_t, W_t):
    """Weighted per-sample MSE -- matches train_model's loss exactly (no averaging)."""
    pred = model(X_t).view(-1)
    return W_t.view(-1) * (pred - Y_t.view(-1)) ** 2


def mean_data_loss(model, X_t, Y_t, W_t):
    return per_sample_losses(model, X_t, Y_t, W_t).mean()


def hvp(loss_scalar, params, vector_flat, retain_graph=True):
    """(Hessian of loss_scalar w.r.t. params) @ vector_flat, via double-backward
    (Pearlmutter's trick) -- never forms the Hessian matrix itself.
    """
    grads = torch.autograd.grad(loss_scalar, params, create_graph=True)
    flat_grad = _flatten(grads)
    gv = (flat_grad * vector_flat).sum()
    hvp_parts = torch.autograd.grad(gv, params, retain_graph=retain_graph)
    return _flatten(hvp_parts).detach()


def conjugate_gradient(matvec, b, max_iter=250, tol=1e-8):
    """Solves A @ x = b for x, given only a callable matvec(v) = A @ v (A implicit,
    symmetric positive [semi]definite -- never formed or inverted explicitly).
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = (r * r).sum()
    if torch.sqrt(rs_old) < tol:
        return x
    for _ in range(max_iter):
        Ap = matvec(p)
        denom = (p * Ap).sum()
        alpha = rs_old / denom.clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def estimate_min_eigenvalue(matvec, p, device):
    """Smallest algebraic eigenvalue of the (implicit, symmetric) operator `matvec`, via
    scipy's ARPACK-based eigsh on a matrix-free LinearOperator -- needed because a
    network's true Hessian is generally INDEFINITE away from a genuine local minimum
    (confirmed empirically: an untrained/early-stopped network's H can have sizable
    negative eigenvalues), and conjugate_gradient requires a positive [semi]definite
    operator -- silently diverges (blows up to inf/nan) otherwise, since CG has no
    indefiniteness safeguard of its own. Used to pick a damping large enough to guarantee
    H + damping*I is PD, rather than trusting a fixed, possibly-too-small damping value.
    """
    from scipy.sparse.linalg import eigsh, LinearOperator

    def np_matvec(v_np):
        v_t = torch.tensor(v_np, dtype=torch.float32, device=device)
        return matvec(v_t).cpu().numpy().astype(np.float64)

    op = LinearOperator((p, p), matvec=np_matvec, dtype=np.float64)
    try:
        eigvals = eigsh(op, k=1, which="SA", return_eigenvectors=False, maxiter=5000)
        return float(eigvals[0])
    except Exception:
        # ARPACK can fail to converge on a near-degenerate/tiny operator -- fall back to a
        # conservative (i.e. more damping than strictly needed) estimate via power
        # iteration on (shift*I - H), which converges to shift - min_eigenvalue.
        shift = 10.0
        v = torch.randn(p, device=device)
        v = v / v.norm()
        for _ in range(50):
            v = shift * v - matvec(v)
            v = v / v.norm().clamp_min(1e-12)
        Hv = matvec(v)
        rayleigh = (v * Hv).sum().item()
        return rayleigh


def damped_hvp_factory(model, X_t, Y_t, W_t, damping, adaptive=True):
    """matvec for CG: (H_data + effective_damping*I) @ v, where H_data is the Hessian of
    the mean DATA loss only (no L2 term -- see module docstring for why weight_decay's L2
    curvature is folded into damping instead of recomputed here, and why J excludes it
    too). If adaptive=True (default), effective_damping is raised above `damping` as
    needed so that H_data + effective_damping*I is positive definite (see
    estimate_min_eigenvalue) -- a fixed, too-small damping otherwise makes
    conjugate_gradient diverge silently to inf/nan on an indefinite H_data.
    """
    params = _params_list(model)
    p = sum(pp.numel() for pp in params)

    def undamped_matvec(v):
        loss = mean_data_loss(model, X_t, Y_t, W_t)
        return hvp(loss, params, v)

    effective_damping = damping
    if adaptive:
        min_eig = estimate_min_eigenvalue(undamped_matvec, p, X_t.device)
        if min_eig < damping:
            effective_damping = -min_eig + max(damping, 1e-3)

    def matvec(v):
        return undamped_matvec(v) + effective_damping * v

    return matvec


def per_sample_scores(model, X_t, Y_t, W_t):
    """(n, p) matrix of per-sample DATA-only score vectors s_i = grad_theta ell_i(theta),
    via vmap+grad (torch.func) -- avoids an explicit n-iteration Python loop.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    param_names = list(params.keys())

    def compute_loss(p, x, y, w):
        pred = functional_call(model, (p, buffers), (x.unsqueeze(0),))
        return w * (pred.squeeze() - y) ** 2

    grad_fn = func_grad(compute_loss)
    per_sample_grads = vmap(grad_fn, in_dims=(None, 0, 0, 0))(params, X_t, Y_t, W_t)
    n = X_t.shape[0]
    return torch.cat([per_sample_grads[name].reshape(n, -1) for name in param_names], dim=1)


def mixed_jvp_sigma_wrt_mu(model_mu, model_sigma, X_t, Y_t, W_t, v_mu):
    """H_{sigma,mu} @ v_mu = d/d(theta_mu) [grad_theta_sigma ell_sigma_avg(theta_sigma;
    theta_mu)] applied in direction v_mu -- a Jacobian-vector product (forward-mode over
    the inner reverse-mode gradient), computed via torch.func.jvp(torch.func.grad(...))
    so no explicit p_sigma x p_mu matrix is ever formed. Both sub-networks are called via
    functional_call throughout (not the plain nn.Module.__call__) -- composing
    torch.func's forward-mode jvp with an INNER reverse-mode grad requires the whole
    computation to go through torch.func's own functional primitives; mixing in a plain
    autograd.grad call here breaks the dual-tensor tracking jvp relies on.
    """
    mu_params = {k: v.detach() for k, v in model_mu.named_parameters()}
    mu_buffers = {k: v.detach() for k, v in model_mu.named_buffers()}
    sigma_params = {k: v.detach() for k, v in model_sigma.named_parameters()}
    sigma_buffers = {k: v.detach() for k, v in model_sigma.named_buffers()}
    mu_param_names = list(mu_params.keys())
    v_mu_dict = {}
    offset = 0
    for name in mu_param_names:
        numel = mu_params[name].numel()
        v_mu_dict[name] = v_mu[offset:offset + numel].view_as(mu_params[name])
        offset += numel

    def loss_sigma_avg(mu_params_dict, sigma_params_dict):
        pred_mu = functional_call(model_mu, (mu_params_dict, mu_buffers), (X_t,)).view(-1)
        resid2 = (Y_t.view(-1) - pred_mu) ** 2
        pred_var = functional_call(model_sigma, (sigma_params_dict, sigma_buffers), (X_t,)).view(-1)
        return (W_t.view(-1) * (pred_var - resid2) ** 2).mean()

    grad_sigma_fn = func_grad(loss_sigma_avg, argnums=1)  # d(loss)/d(sigma_params), as fn of (mu_params, sigma_params)

    def grad_sigma_of_mu(mu_params_dict):
        return grad_sigma_fn(mu_params_dict, sigma_params)

    _, jvp_out_dict = jvp(grad_sigma_of_mu, (mu_params,), (v_mu_dict,))
    sigma_param_names = list(sigma_params.keys())
    return torch.cat([jvp_out_dict[name].reshape(-1) for name in sigma_param_names]).detach()


def mixed_vjp_mu_wrt_sigma(model_mu, model_sigma, X_t, Y_t, W_t, w_sigma):
    """H_{sigma,mu}^T @ w_sigma = grad_theta_mu [ w_sigma . grad_theta_sigma
    ell_sigma_avg(theta_sigma; theta_mu) ] -- the TRANSPOSE-direction mixed term, needed
    because H (block lower triangular, not symmetric) means a^T H^-1 != (H^-1 a)^T in
    general: computing a^T H^-1 J H^-1 a correctly requires solving BOTH H v = a and
    H^T w = a separately (see group_variance_contribution), not just v = H^-1 a alone.
    Unlike the forward JVP above (H_{sigma,mu} @ v_mu), this is a plain reverse-mode
    double-backward (a VJP, the direction reverse-mode autodiff gives natively) -- no
    torch.func/functional_call needed, same pattern as `hvp`.
    """
    mu_params = _params_list(model_mu)
    sigma_params = _params_list(model_sigma)
    pred_mu = model_mu(X_t).view(-1)
    resid2 = (Y_t.view(-1) - pred_mu) ** 2
    pred_var = model_sigma(X_t).view(-1)
    loss_sigma = (W_t.view(-1) * (pred_var - resid2) ** 2).mean()
    grad_sigma = torch.autograd.grad(loss_sigma, sigma_params, create_graph=True)
    flat_grad_sigma = _flatten(grad_sigma)
    scalar = (flat_grad_sigma * w_sigma).sum()
    grad_mu = torch.autograd.grad(scalar, mu_params)
    return _flatten(grad_mu).detach()


def eta_grad(mu0_model, sigma0_model, mu1_model, sigma1_model, x0_row, x1_row, adaptive_direction=True):
    """grad_theta eta(x) for all four networks' parameters. Group 0's networks are
    evaluated at x0_row and group 1's at x1_row -- NOT necessarily the same covariate
    profile, matching the row-paired convention the rest of this codebase uses
    (bootstrap_crossfit_oob_paired/bootstrap_crossfit_parametric_*_paired,
    ground_truth_auc's row-paired true_auc): row i pairs an independently-drawn group-0
    subject's covariates with an independently-drawn group-1 subject's. Pass the same
    tensor for both if you actually want a single shared covariate profile (the "_shared"/
    real-data convention instead).

    adaptive_direction=True (the DEFAULT, and what the paired convention requires) uses
    eta(x) = |mu1(x1)-mu0(x0)| / sqrt(sigma0(x0)^2 + sigma1(x1)^2), matching
    bootstrap_crossfit_parametric._gaussian_auc(..., adaptive_direction=True) and hence
    ground_truth_auc.true_auc's own per-row "higher-mean group vs. the other" convention.
    This is NOT optional for the simulation scenarios: which group has the higher mean is
    not fixed across scenarios (Scenario IV's healthy/diseased means cross over within the
    covariate range), so a fixed-direction eta silently estimates 1 - true_AUC on the rows
    where the ordering flips -- observed in practice as estimated AUC ~1e-5 against a true
    AUC ~0.998, collapsing coverage toward 0 (see bootstrap_crossfit_oob's
    compute_empirical_auc docstring, which flags exactly this trap).

    adaptive_direction=False gives the fixed P(T1>T0) convention instead, for the
    real-data/_shared style where the direction is deliberately held fixed.
    """
    x0_t = x0_row if x0_row.dim() == 2 else x0_row.unsqueeze(0)
    x1_t = x1_row if x1_row.dim() == 2 else x1_row.unsqueeze(0)
    mu0 = mu0_model(x0_t).view(())
    mu1 = mu1_model(x1_t).view(())
    # The variance networks are trained on SQUARED residuals, so their output already IS
    # sigma^2, not sigma (same convention as mlp_reg_data_simulation_multi.compute_std,
    # which sqrt()s the raw output). The denominator is therefore sqrt(v0 + v1), NOT
    # sqrt(v0^2 + v1^2) -- the latter computes sqrt(sigma0^4 + sigma1^4) and inflates eta
    # by ~8.5% at the simulation scenarios' sigma0=0.5/sigma1=1.0, which is most of the
    # systematic positive eta bias measured in the first real-scale run (see project
    # memory). Each variance is floored at 0 before summing, as compute_std does.
    v0 = torch.clamp(sigma0_model(x0_t).view(()), min=0.0)
    v1 = torch.clamp(sigma1_model(x1_t).view(()), min=0.0)
    denom = torch.sqrt(torch.clamp(v0 + v1, min=1e-12))
    diff = torch.abs(mu1 - mu0) if adaptive_direction else (mu1 - mu0)
    eta = diff / denom

    params0mu, params1mu = _params_list(mu0_model), _params_list(mu1_model)
    params0s, params1s = _params_list(sigma0_model), _params_list(sigma1_model)
    # Each network's gradient is independent of the other three (eta(x) for one group's
    # block only depends on that group's own two networks), so four separate grad() calls
    # -- retain_graph=True on all but the last, since they all differentiate the same eta.
    g0mu = torch.autograd.grad(eta, params0mu, retain_graph=True)
    g0s = torch.autograd.grad(eta, params0s, retain_graph=True)
    g1mu = torch.autograd.grad(eta, params1mu, retain_graph=True)
    g1s = torch.autograd.grad(eta, params1s, retain_graph=False)

    return (_flatten(g0mu), _flatten(g0s), _flatten(g1mu), _flatten(g1s), eta.detach())


class GroupSandwichContext:
    """Everything needed for ONE group's sandwich variance contribution that does NOT
    depend on the query point x -- computed ONCE per group and reused across every
    subject's CI, instead of recomputed per query (per_sample_scores and the adaptive
    eigenvalue/damping estimate in particular are both expensive and x-independent;
    recomputing them per subject was the dominant, unnecessary cost in early benchmarking
    -- see project memory).
    """

    def __init__(self, mu_model, sigma_model, X_t, Y_t, W_t, damping, resid2_t=None, cross_term=True):
        self.mu_model = mu_model
        self.sigma_model = sigma_model
        # resid2_t: the variance network's actual training target. Defaults to the full-data
        # mean net's own IN-SAMPLE squared residuals; pass out-of-fold ones (see
        # oof_squared_residuals) to remove the downward sigma bias, in which case
        # cross_term must be False -- the stage-2 loss then doesn't depend on theta_mu.
        self.resid2 = resid2_t if resid2_t is not None else (Y_t.view(-1) - mu_model(X_t).view(-1).detach()) ** 2
        self.cross_term = cross_term
        self.mu_matvec = damped_hvp_factory(mu_model, X_t, Y_t, W_t, damping)
        self.sigma_matvec = damped_hvp_factory(sigma_model, X_t, self.resid2, W_t, damping)
        self.s_mu = per_sample_scores(mu_model, X_t, Y_t, W_t)
        self.s_sigma = per_sample_scores(sigma_model, X_t, self.resid2, W_t)
        self.X_t, self.Y_t, self.W_t = X_t, Y_t, W_t
        self.mean_score_norm = self.s_mu.mean(dim=0).norm().item()


def group_variance_contribution(ctx, a_mu, a_sigma):
    """Var contribution to eta(x) from ONE group's two-stage (mu, sigma) sandwich, given
    a_mu = grad_theta_mu eta(x), a_sigma = grad_theta_sigma eta(x) for that group and a
    precomputed GroupSandwichContext (x-independent quantities, shared across subjects).

    H (stacked over both networks) is block LOWER triangular, not symmetric (stage 2
    depends on stage 1's fit, not vice versa) -- so a^T H^-1 J H^-1 a is NOT simply
    v^T J v for v = H^-1 a (that shortcut needs H symmetric, true for a single network's
    own Hessian but NOT for this joint two-stage H; confirmed by a dense-matrix numerical
    check during development, see project memory). The correct computation needs BOTH
        v = H^-1 a       (solves H v = a, block lower-triangular forward substitution)
        w = H^-T a       (solves H^T w = a, block UPPER-triangular backward substitution)
    and then a^T H^-1 J H^-1 a = w^T J v.

    When ctx.cross_term is False (the out-of-fold-residual variant: stage 2's targets come
    from auxiliary fold networks, not from theta_mu, so H_sigma_mu is zero by construction)
    H is block DIAGONAL and therefore symmetric, v == w, and only 2 CG solves are needed.

    Returns a scalar tensor: (1/n) * w^T J v.
    """
    n = ctx.X_t.shape[0]
    mu_model, sigma_model = ctx.mu_model, ctx.sigma_model
    X_t, Y_t, W_t = ctx.X_t, ctx.Y_t, ctx.W_t

    if not ctx.cross_term:
        v_mu = conjugate_gradient(ctx.mu_matvec, a_mu)
        v_sigma = conjugate_gradient(ctx.sigma_matvec, a_sigma)
        proj = ctx.s_mu @ v_mu + ctx.s_sigma @ v_sigma
        return (proj ** 2).mean() / n

    # Forward solve: H v = a. H_mumu block first, then H_sigmasigma with the adjusted RHS.
    v_mu = conjugate_gradient(ctx.mu_matvec, a_mu)
    h_sigma_mu_vmu = mixed_jvp_sigma_wrt_mu(mu_model, sigma_model, X_t, Y_t, W_t, v_mu)
    v_sigma = conjugate_gradient(ctx.sigma_matvec, a_sigma - h_sigma_mu_vmu)

    # Transpose solve: H^T w = a. H_sigmasigma block first (H^T is upper triangular), then
    # H_mumu with the adjusted RHS via the transpose-direction mixed term (VJP, not JVP).
    w_sigma = conjugate_gradient(ctx.sigma_matvec, a_sigma)
    h_sigma_mu_T_wsigma = mixed_vjp_mu_wrt_sigma(mu_model, sigma_model, X_t, Y_t, W_t, w_sigma)
    w_mu = conjugate_gradient(ctx.mu_matvec, a_mu - h_sigma_mu_T_wsigma)

    proj_v = ctx.s_mu @ v_mu + ctx.s_sigma @ v_sigma            # (n,)
    proj_w = ctx.s_mu @ w_mu + ctx.s_sigma @ w_sigma            # (n,)
    return (proj_w * proj_v).mean() / n


def sandwich_auc_ci(ctx0, ctx1, x0_row, x1_row, ci_levels=(0.95,), adaptive_direction=True):
    """Full sandwich+delta CI for AUC(x) = Phi(eta(x)) at one (x0_row, x1_row) row-paired
    covariate pair, for every level in ci_levels -- se_eta is computed ONCE and reused for
    all levels (only the z-multiplier changes), same no-extra-cost-per-level principle as
    bootstrap_crossfit_parametric.py's ci_levels.

    ctx0/ctx1: GroupSandwichContext for group 0/1 (build once per replicate, reuse across
    every subject's query -- see that class's docstring for why).

    Returns (auc_point, ci_bounds, se_eta, score_norm0, score_norm1) -- ci_bounds is a
    dict {level: (lower, upper)}; score_norm0/1 are the stationarity diagnostic (module
    docstring): should be small relative to the score's own scale if the M-estimation
    approximation is trustworthy at this theta_hat.
    """
    a0mu, a0s, a1mu, a1s, eta = eta_grad(
        ctx0.mu_model, ctx0.sigma_model, ctx1.mu_model, ctx1.sigma_model,
        x0_row, x1_row, adaptive_direction=adaptive_direction,
    )

    var0 = group_variance_contribution(ctx0, a0mu, a0s)
    var1 = group_variance_contribution(ctx1, a1mu, a1s)
    var_eta = (var0 + var1).item()
    se_eta = np.sqrt(max(var_eta, 0.0))
    eta_val = eta.item()

    ci_bounds = {}
    for level in ci_levels:
        z = norm.ppf(1 - (1 - level) / 2)
        ci_bounds[level] = (norm.cdf(eta_val - z * se_eta), norm.cdf(eta_val + z * se_eta))

    auc_point = norm.cdf(eta_val)
    return auc_point, ci_bounds, se_eta, ctx0.mean_score_norm, ctx1.mean_score_norm


def _solve_group(ctx, a_mu, a_sigma):
    """v = H^-1 a and w = H^-T a for one group (see group_variance_contribution for why
    both are needed when the two-stage H is non-symmetric). With ctx.cross_term False,
    H is block diagonal and symmetric, so v == w and one pair of solves suffices.
    """
    if not ctx.cross_term:
        v_mu = conjugate_gradient(ctx.mu_matvec, a_mu)
        v_sigma = conjugate_gradient(ctx.sigma_matvec, a_sigma)
        return (v_mu, v_sigma), (v_mu, v_sigma)
    mu_model, sigma_model = ctx.mu_model, ctx.sigma_model
    X_t, Y_t, W_t = ctx.X_t, ctx.Y_t, ctx.W_t
    v_mu = conjugate_gradient(ctx.mu_matvec, a_mu)
    v_sigma = conjugate_gradient(ctx.sigma_matvec,
                                  a_sigma - mixed_jvp_sigma_wrt_mu(mu_model, sigma_model, X_t, Y_t, W_t, v_mu))
    w_sigma = conjugate_gradient(ctx.sigma_matvec, a_sigma)
    w_mu = conjugate_gradient(ctx.mu_matvec,
                               a_mu - mixed_vjp_mu_wrt_sigma(mu_model, sigma_model, X_t, Y_t, W_t, w_sigma))
    return (v_mu, v_sigma), (w_mu, w_sigma)


def averaged_auc_ci(ctx0, ctx1, X0_t, X1_t, ci_levels=(0.95,), adaptive_direction=True):
    """Inference for the AVERAGED functional  Abar = mean_i AUC(x0_i, x1_i)  instead of the
    pointwise AUC(x).

    Why this target: the pointwise conditional AUC at a continuous x is a nonparametric-rate
    functional (the subsampling rate check measured beta ~ 0.21-0.59, never 0.5), whereas an
    average over the covariate distribution is a smooth functional that is normally
    sqrt(n)-estimable. Empirically the pointwise error decomposes into a replicate-level
    shift, which the sandwich SE sizes essentially perfectly (0.99 in Scenario 1), plus an
    across-x within-fit deviation (0.67 in SE units) that it misses -- and the second
    component CANCELS on averaging. See project memory.

    TWO variance terms, not one:
      1. theta estimation:  gbar = grad_theta Abar, then v = H^-1 gbar per group (ONE
         backward pass on the scalar mean, not n separate gradients).
      2. covariate sampling: the x_i are a sample, so estimating E[AUC(X)] by an empirical
         mean carries Var_X(AUC(X))/n on top. Omitting this underestimates the SE -- same
         structure as the ATE influence function in causal inference. Measured size on
         these scenarios: nil where the true AUC is constant (1, 8), up to ~40% of the
         pointwise SE where it varies a lot (4, 6, 7).

    Per row i (the paired convention makes row i an i.i.d. draw of a group-0 subject paired
    with a group-1 subject) the influence contribution is

        psi_i = v0 . s0_i  +  v1 . s1_i  +  (AUC_i - Abar)

    and SE^2 = (1/n^2) sum_i psi_i^2, which picks up the cross-covariance between the two
    terms automatically. NOTE when cross_term is True the theta part is not a plain square
    (H non-symmetric, the exact form is w^T J v); that exact value is used for its own
    variance and the symmetrized influence is used only for the cross-covariance, which is
    an approximation flagged here rather than hidden.

    Returns (auc_mean, ci_bounds, se, parts) with parts = (var_theta, var_cov, 2*cov).
    """
    n = X0_t.shape[0]
    normal = torch.distributions.Normal(0.0, 1.0)
    mu0 = ctx0.mu_model(X0_t).view(-1)
    mu1 = ctx1.mu_model(X1_t).view(-1)
    v0 = torch.clamp(ctx0.sigma_model(X0_t).view(-1), min=0.0)
    v1 = torch.clamp(ctx1.sigma_model(X1_t).view(-1), min=0.0)
    denom = torch.sqrt(torch.clamp(v0 + v1, min=1e-12))
    diff = torch.abs(mu1 - mu0) if adaptive_direction else (mu1 - mu0)
    auc_i = normal.cdf(diff / denom)
    abar = auc_i.mean()

    p0mu, p0s = _params_list(ctx0.mu_model), _params_list(ctx0.sigma_model)
    p1mu, p1s = _params_list(ctx1.mu_model), _params_list(ctx1.sigma_model)
    g0mu = _flatten(torch.autograd.grad(abar, p0mu, retain_graph=True))
    g0s = _flatten(torch.autograd.grad(abar, p0s, retain_graph=True))
    g1mu = _flatten(torch.autograd.grad(abar, p1mu, retain_graph=True))
    g1s = _flatten(torch.autograd.grad(abar, p1s, retain_graph=False))

    (v0mu, v0s), (w0mu, w0s) = _solve_group(ctx0, g0mu, g0s)
    (v1mu, v1s), (w1mu, w1s) = _solve_group(ctx1, g1mu, g1s)

    psi_v = (ctx0.s_mu @ v0mu + ctx0.s_sigma @ v0s) + (ctx1.s_mu @ v1mu + ctx1.s_sigma @ v1s)
    psi_w = (ctx0.s_mu @ w0mu + ctx0.s_sigma @ w0s) + (ctx1.s_mu @ w1mu + ctx1.s_sigma @ w1s)
    auc_np = auc_i.detach()
    psi_cov = auc_np - auc_np.mean()

    var_theta = ((psi_w * psi_v).mean() / n).item()
    var_cov = ((psi_cov ** 2).mean() / n).item()
    cross = 2.0 * ((0.5 * (psi_v + psi_w) * psi_cov).mean() / n).item()
    var_total = max(var_theta + var_cov + cross, 0.0)
    se = float(np.sqrt(var_total))

    abar_val = abar.item()
    ci_bounds = {lv: (max(0.0, abar_val - norm.ppf(1 - (1 - lv) / 2) * se),
                      min(1.0, abar_val + norm.ppf(1 - (1 - lv) / 2) * se)) for lv in ci_levels}
    return abar_val, ci_bounds, se, (var_theta, var_cov, cross)
