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
