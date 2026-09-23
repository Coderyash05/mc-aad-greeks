"""Shared assertions for Monte Carlo tests. The policy is in docs/TESTING.md.

Every statistical comparison goes through `assert_within_se` (or `record` for a
hand-built test), which also logs its per-comparison false-alarm probability. The
pytest summary (tests/conftest.py) then prints how many comparisons ran and the
family-wise false-alarm probability of the whole run.
"""
import math

import numpy as np
from scipy import integrate, special, stats

from mcgreeks.stats import family_zscores

K_SE = 4.0
P_FALSE_ALARM_4SE = 2 * stats.norm.sf(K_SE)      # 6.3e-5 per comparison
RARE_THRESHOLD = 100                             # nonzero-payoff paths needed for an SE test

LEDGER = []          # (test node id, n comparisons, false-alarm prob per comparison)
RARE_LOG = []        # (test node id, description): rare-event branches taken
CURRENT = {"test": None}


def record(n, p, test=None):
    """Log n comparisons that each fail a correct estimator with probability p."""
    LEDGER.append((test or CURRENT["test"], int(n), float(p)))


def p_false_alarm(k=K_SE, dof=None):
    """Two-sided P(|z| >= k) for z ~ N(0, 1), or ~ t_dof for a batch-means SE."""
    return 2 * (stats.norm.sf(k) if dof is None else stats.t.sf(k, dof))


def assert_within_se(estimate, se, exact, k=K_SE, reason="", dof=None):
    """|estimate - exact| < k * se, elementwise.

    `exact` must be the expectation of the ESTIMATOR: the true Greek for an unbiased
    estimator; for a biased one (finite differences, smoothing) its own target,
    computed exactly (see smoothed_*_expectation and the FD tests). No absolute floor:
    an SE of 0 is an error here, and rare-event cases (few nonzero payoffs, where the
    SE itself is unreliable) must use assert_rare_event instead.
    """
    est, se, ex = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (estimate, se, exact)))
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        raise AssertionError(f"{reason}: SE must be finite and > 0 (got min {se.min():.3g}); "
                             "for rare events use assert_rare_event")
    z = (est - ex) / se
    record(z.size, p_false_alarm(k, dof))
    i = int(np.argmax(np.abs(z)))
    assert np.all(np.abs(z) < k), (
        f"{reason}: |z| = {abs(z.flat[i]):.2f} >= {k} at index {i} "
        f"(estimate {est.flat[i]:.6g}, exact {ex.flat[i]:.6g}, SE {se.flat[i]:.3g})")


def assert_family(G, exact, k=K_SE, reason=""):
    """A family of P correlated estimates from the same B batches (G: (B, P) batch
    estimates, exact: (P,)); see mcgreeks.stats.family_zscores and docs/TESTING.md.

    1. every |z_k| < k (pass k = k_familywise(P, ...) for family-wise control);
    2. aggregate: mean z within 4 of ITS SE (from the batches; t_{B-1});
    3. aggregate: mean z^2 inside the two-sided band of a moment-matched scaled
       chi-square a chi^2_f (mean nu/(nu - 2), variance from the estimated
       correlations) with the false-alarm probability of one 4-SE test. A normal
       approximation understates the right tail (E9 (c): 4.55 sd seen in 200 seeds).
    """
    fz = family_zscores(G, exact)
    nu = fz["dof"]
    assert_within_se(fz["z"], 1.0, 0.0, k=k, reason=f"{reason}: individual z", dof=nu)
    assert_within_se(fz["mean_z"], fz["se_mean_z"], 0.0, reason=f"{reason}: mean z", dof=nu)
    a, f = fz["chi2_scale"], fz["chi2_dof"]
    lo, hi = a * stats.chi2.ppf([P_FALSE_ALARM_4SE / 2, 1 - P_FALSE_ALARM_4SE / 2], f)
    record(1, P_FALSE_ALARM_4SE)
    assert lo < fz["mean_z2"] < hi, (f"{reason}: mean z^2 = {fz['mean_z2']:.3f} outside "
                                     f"({lo:.3f}, {hi:.3f}), scaled chi^2 with f = {f:.1f}")
    return fz


def assert_rare_event(estimate, se, n_nonzero, n, p_nonzero, reason=""):
    """Structural checks when fewer than RARE_THRESHOLD paths have a nonzero payoff.

    With so few nonzero terms the sample SE is itself unreliable (it is estimated from
    a handful of values and is usually too small), so an SE test is not meaningful.
    Instead:
      - the count of nonzero payoffs is consistent with Binomial(n, p_nonzero), with
        p_nonzero the exact risk-neutral exercise probability (two-sided binomial test
        at the same false-alarm level as a 4-SE test). This checks the tail of the
        path generator, which is what the price depends on here;
      - no nonzero payoff -> estimate and SE exactly 0; otherwise estimate > 0.
    Sign facts for Greeks are asserted by the caller.
    """
    assert_count_consistent(n_nonzero, n, p_nonzero, reason)
    if n_nonzero == 0:
        assert float(estimate) == 0.0 and float(se) == 0.0, reason
    else:
        assert float(estimate) > 0.0, reason


def assert_count_consistent(n_observed, n, p, reason=""):
    """Two-sided exact binomial test of n_observed ~ Binomial(n, p), at the false-alarm
    level of a 4-SE test (conservative: the binomial is discrete)."""
    pval = stats.binomtest(int(n_observed), int(n), float(p)).pvalue
    record(1, P_FALSE_ALARM_4SE)
    RARE_LOG.append((CURRENT["test"], f"{reason}: count {n_observed} of {n} "
                                      f"(expected {n * p:.1f})"))
    assert pval >= P_FALSE_ALARM_4SE, (
        f"{reason}: count {n_observed}, expected {n * p:.1f} (binomial p-value {pval:.2g})")


def assert_chi2(sum_sq, dof, reason=""):
    """sum of dof squared standard normals within the two-sided chi^2_dof band that has
    the same false-alarm probability as a 4-SE test."""
    lo, hi = stats.chi2.ppf([P_FALSE_ALARM_4SE / 2, 1 - P_FALSE_ALARM_4SE / 2], dof)
    record(1, P_FALSE_ALARM_4SE)
    assert lo < sum_sq < hi, f"{reason}: sum z^2 = {sum_sq:.1f}, band ({lo:.1f}, {hi:.1f})"


# ---- exact expectations of the smoothed estimators (their biased targets) --------------
# Under GBM, S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) z), z ~ N(0, 1), and S_T is
# linear in S0 (dS_T/dS0 = S_T/S0, d^2 S_T/dS0^2 = 0). For a smoothed payoff f_eps:
#   E[smoothed delta] = e^{-rT} E[f_eps'(S_T) S_T/S0]
#   E[smoothed gamma] = e^{-rT} E[f_eps''(S_T) (S_T/S0)^2]
# Sigmoid digital: f' = s(1 - s)/eps with s = sigmoid((S - K)/eps). Softplus call
# eps*softplus((S - K)/eps): f'' = s(1 - s)/eps. Both integrals are one-dimensional in
# z; the integrand has width ~eps / (S0 sigma sqrt(T)) around the strike, so the
# range is split there for adaptive quadrature (scipy.integrate.quad).

def _gauss_expect(g, z_star, width):
    edges = [-12.0, z_star - 20 * width, z_star + 20 * width, 12.0]
    edges = sorted(min(max(e, -12.0), 12.0) for e in edges)
    return sum(integrate.quad(lambda z: g(z) * stats.norm.pdf(z), a, b, limit=500,
                              epsabs=1e-15, epsrel=1e-12)[0]
               for a, b in zip(edges[:-1], edges[1:]) if b > a)


def _smoothed_expectation(S0, sigma, r, T, K, eps, power):
    vol = sigma * math.sqrt(T)

    def g(z):
        ST = S0 * math.exp((r - 0.5 * sigma**2) * T + vol * z)
        s = special.expit((ST - K) / eps)
        return math.exp(-r * T) * s * (1 - s) / eps * (ST / S0) ** power

    z_star = (math.log(K / S0) - (r - 0.5 * sigma**2) * T) / vol
    return _gauss_expect(g, z_star, eps / (K * vol))


def smoothed_digital_delta_expectation(S0, sigma, r, T, K, eps):
    return _smoothed_expectation(S0, sigma, r, T, K, eps, power=1)


def smoothed_call_gamma_expectation(S0, sigma, r, T, K, eps):
    return _smoothed_expectation(S0, sigma, r, T, K, eps, power=2)


def exercise_probability(S0, sigma, r, T, K, kind):
    """Risk-neutral P(S_T > K) for a call, P(S_T < K) for a put: N(+-d2)."""
    d2 = (math.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(stats.norm.cdf(d2 if kind == "call" else -d2))
