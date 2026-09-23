"""Paired bootstrap confidence intervals for comparing Monte Carlo estimators.

Setup: R independent batches; every method is evaluated on the SAME R batches, so
method m has per-batch errors e_m[r] = estimate_m[r] - exact, r = 1..R, and
    RMSE_m = sqrt( (1/R) sum_r e_m[r]^2 ).
Errors of different methods on the same batch are correlated (common random
numbers), so the methods must be resampled TOGETHER: each bootstrap replicate draws
R batch indices with replacement, i* ~ Uniform{1..R}^R, and recomputes every
method's RMSE on those same indices. Ratios RMSE_a / RMSE_b computed within a
replicate then carry the positive correlation, which makes their intervals much
tighter than comparing two separate intervals. This is the nonparametric
(Efron) bootstrap applied to the batch means, the unit that is i.i.d. here
(Glasserman 2004, App. A on batching; Efron & Tibshirani 1993, ch. 13 for the
percentile interval).

Percentile interval: the 2.5% and 97.5% quantiles of the B replicate values.

Claims: "A beats B" only if the interval for RMSE_A / RMSE_B lies entirely
below 1; if it contains 1, A and B are "not distinguishable". A method whose
error is identically zero-variance (e.g. naive autodiff gamma, always 0) should be
reported by its bias, not put in a ratio.

Resampling uses numpy's Generator(PCG64) with a fixed integer seed: deterministic,
and independent of the JAX PRNG streams that produced the paths.
"""
import numpy as np
from scipy import stats


def paired_bootstrap(errors_by_method, n_boot=2000, seed=0, reference=None, level=0.95):
    """RMSE, percentile CI, and RMSE ratio vs `reference` with CI, for every method.

    errors_by_method: {name: array of R per-batch errors}, all on the same batches.
    reference: method name for the ratios; default = the method with the lowest RMSE.
    Returns {"reference": name, "n_batches": R, "methods": {name: {rmse, ci, ratio,
    ratio_ci, beats_reference, beaten_by_reference}}}.
    """
    names = list(errors_by_method)
    E = np.stack([np.asarray(errors_by_method[m], dtype=np.float64) for m in names])
    k, R = E.shape
    rmse = np.sqrt(np.mean(E**2, axis=1))
    if reference is None:
        reference = names[int(np.argmin(rmse))]
    j = names.index(reference)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, R, size=(n_boot, R))
    sq = E**2
    rmse_b = np.sqrt(np.stack([sq[i][idx].mean(axis=1) for i in range(k)]))   # (k, n_boot)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    ci = np.percentile(rmse_b, [lo, hi], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_b = rmse_b / rmse_b[j]
    ratio_ci = np.percentile(ratio_b, [lo, hi], axis=1)

    out = {}
    for i, m in enumerate(names):
        out[m] = {"rmse": float(rmse[i]), "ci": (float(ci[0, i]), float(ci[1, i])),
                  "ratio": float(rmse[i] / rmse[j]),
                  "ratio_ci": (float(ratio_ci[0, i]), float(ratio_ci[1, i])),
                  "beats_reference": bool(ratio_ci[1, i] < 1.0),
                  "beaten_by_reference": bool(ratio_ci[0, i] > 1.0)}
    return {"reference": reference, "n_batches": R, "methods": out}


def rmse(errors):
    return float(np.sqrt(np.mean(np.asarray(errors, dtype=np.float64) ** 2)))


def tune(grid, est_cal, est_eval, exact_cal, exact_eval=None):
    """Out-of-sample choice of a tuning parameter (bump size h, smoothing width eps).

    est_cal, est_eval: arrays (len(grid), R) of per-batch estimates on the
    calibration batches and on the evaluation batches (independent random numbers).
    The parameter is chosen by RMSE on the CALIBRATION batches and then evaluated on
    the evaluation batches. Choosing it by RMSE on the evaluation batches themselves
    (in-sample) picks the value whose noise happened to be favourable there, which
    biases that RMSE downward: selection bias. Both are returned so the gap is visible.
    """
    exact_eval = exact_cal if exact_eval is None else exact_eval
    cal = np.sqrt(np.mean((np.asarray(est_cal) - exact_cal) ** 2, axis=1))
    ev = np.sqrt(np.mean((np.asarray(est_eval) - exact_eval) ** 2, axis=1))
    i_out, i_in = int(np.argmin(cal)), int(np.argmin(ev))
    return {"value": float(grid[i_out]), "index": i_out, "rmse_out": float(ev[i_out]),
            "value_in": float(grid[i_in]), "index_in": i_in, "rmse_in": float(ev[i_in]),
            "rmse_cal": float(cal[i_out]), "errors": np.asarray(est_eval[i_out]) - exact_eval}


def verdict(result, method):
    """Plain-language comparison of `method` with the result's reference."""
    ref = result["reference"]
    r = result["methods"][method]
    if method == ref:
        return "reference"
    if r["beats_reference"]:
        return f"beats {ref}"
    if r["beaten_by_reference"]:
        return f"worse than {ref}"
    return f"not distinguishable from {ref}"


# ---- families of correlated comparisons ---------------------------------------------
# A "family" is P estimates computed from the same B independent batches (e.g. all
# 2d + d(d-1)/2 basket sensitivities). G[b, k] is the batch-b estimate of quantity k;
# the estimate is the batch mean, SE_k = sd_b(G[:, k]) / sqrt(B), and
# z_k = (mean_k - exact_k) / SE_k ~ t_{B-1} for each k. The z_k are CORRELATED (the same
# paths drive all of them), which matters for any statement about the whole family.

def k_familywise(P, alpha, dof=None):
    """Per-comparison threshold k with P(any |z_k| >= k) <= alpha for P comparisons:
    Sidak, 1 - (1 - alpha)^(1/P) per comparison. Exact for independent z_k and
    conservative for positively dependent ones (Sidak 1967), so it controls the
    family-wise false-alarm rate here."""
    p = 1.0 - (1.0 - alpha) ** (1.0 / P)
    return float(stats.norm.isf(p / 2) if dof is None else stats.t.isf(p / 2, dof))


def family_zscores(G, exact):
    """z-scores of a family and two aggregate statistics with correct (dependent) SEs.

    G: (B, P) batch estimates; exact: (P,).
    Returns z (P,), and
      mean_z, se_mean_z: mean_k z_k = mean_b u_b with u_b = (1/P) sum_k (G[b,k] - exact_k)/SE_k,
          a fixed linear combination of the estimates (weights 1/(P SE_k)), so its SE
          is sd_b(u_b)/sqrt(B): this accounts for every correlation between the z_k.
          Under no bias, mean_z / se_mean_z ~ t_{B-1}. Detects a bias with a common sign.
      mean_z2, expected_mean_z2, sd_mean_z2: mean_k z_k^2 has expectation nu/(nu - 2),
          nu = B - 1 (t_nu), and, for jointly Gaussian estimates with correlations rho_ij,
              Var(mean z^2) = (1/P^2) [ P v + 2 sum_{i != j} rho_ij^2 ],
          v = Var(t_nu^2) = 2 nu^2 (nu - 1) / ((nu - 2)^2 (nu - 4)) (Isserlis: Cov(z_i^2,
          z_j^2) = 2 rho_ij^2). rho_ij^2 is estimated from the batches, bias-corrected with
          E[r^2] ~ rho^2 + (1 - rho^2)^2/(B - 1) ~ rho^2 + 1/(B - 1) for small rho:
          rho^2_hat = ((B - 1) r^2 - 1)/(B - 2). sum_{i,j} r_ij^2 = ||Xs Xs^T||_F^2/(B - 1)^2
          with Xs the standardised (B, P) batch matrix: a B x B computation, not P x P.
          Detects SEs that are systematically wrong, and biases of mixed sign that
          cancel in mean_z.
      chi2_scale a, chi2_dof f, mean_z2_score: mean z^2 is positive and right-skewed
          (with correlated z_k its effective degrees of freedom are few: f ~ 2.3 at
          d = 2, ~39 at d = 50 for the E9 baskets), so a normal approximation
          understates its upper tail. It is matched instead to a scaled chi-square
          a chi^2_f with the same mean and variance (Satterthwaite 1946; Box 1954):
          a f = E, 2 a^2 f = Var  ->  f = 2 E^2 / Var, a = Var / (2 E).
          mean_z2_score = Phi^{-1}(F_{chi^2_f}(mean_z2 / a)) is ~N(0, 1) if the
          approximation holds (validated over 200 seeds in E9 (c)).
    """
    G = np.asarray(G, dtype=float)
    ex = np.asarray(exact, dtype=float)
    B, P = G.shape
    est = G.mean(0)
    se = G.std(0, ddof=1) / np.sqrt(B)
    z = (est - ex) / se
    u = ((G - ex) / se).mean(1)                       # (B,): mean over k, per batch
    nu = B - 1
    Xs = (G - est) / G.std(0, ddof=1)
    gram = Xs @ Xs.T / (B - 1)                        # (B, B)
    sum_r2_off = float(np.sum(gram**2)) - P           # sum_{i != j} r_ij^2
    n_off = P * (P - 1)
    sum_rho2_off = max(((B - 1) * sum_r2_off - n_off) / (B - 2), 0.0)
    v = 2 * nu**2 * (nu - 1) / ((nu - 2) ** 2 * (nu - 4))
    E, var = nu / (nu - 2), (P * v + 2 * sum_rho2_off) / P**2
    f, a = 2 * E**2 / var, var / (2 * E)
    mean_z2 = float(np.mean(z**2))
    return {"z": z, "mean_z": float(u.mean()), "se_mean_z": float(u.std(ddof=1) / np.sqrt(B)),
            "mean_z2": mean_z2, "expected_mean_z2": E, "sd_mean_z2": float(np.sqrt(var)),
            "chi2_dof": float(f), "chi2_scale": float(a),
            "mean_z2_score": float(stats.norm.ppf(stats.chi2.cdf(mean_z2 / a, f))),
            "dof": nu, "P": P}