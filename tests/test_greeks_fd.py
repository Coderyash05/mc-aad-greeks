"""Finite-difference Greeks behave as theory predicts."""
import jax
import jax.numpy as jnp
import numpy as np
from helpers import assert_within_se

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_greeks, bs_price
from mcgreeks.greeks_ad import ad_greeks
from mcgreeks.greeks_fd import (fd_delta, fd_first, fd_gamma, fd_greeks_loop,
                                fd_greeks_vmap)
from mcgreeks.models import gbm_terminal, normals
from mcgreeks.pricer import discounted_payoffs

N = 200_000
ARGS = (100.0, 0.2, 0.05, 1.0, 100.0)  # S0, sigma, r, T, K


def test_crn_delta_converges_to_pathwise_delta():
    """With common random numbers, FD delta -> the autodiff (pathwise) delta as h -> 0.

    S_T is linear in S0 (a = S_T/S0 = dS_T/dS0), so on each path the central
    difference [max(S_T + h a - K, 0) - max(S_T - h a - K, 0)] / (2h) equals the
    pathwise derivative a 1{S_T > K} unless |S_T - K| < h a, and then differs from it
    by at most a/2. Hence, exactly on these paths,
        |FD - AD| <= e^{-rT} mean(a/2 * 1{|S_T - K| < h a}),
    a bound that shrinks like O(h) (the fraction of paths near the kink).
    """
    S0, sigma, r, T, K = ARGS
    Z = normals(jax.random.PRNGKey(10), N)
    ad = ad_greeks(*ARGS, Z)["delta"]
    ST = gbm_terminal(S0, sigma, r, T, Z)
    a = ST / S0
    for h in (1.0, 0.1, 0.01):
        bound = jnp.exp(-r * T) * jnp.mean(0.5 * a * (jnp.abs(ST - K) < h * a))
        gap = abs(fd_delta(*ARGS, h, Z, Z) - ad)
        # exact inequality; the slack is floating-point rounding only (1/(2h) amplifies it)
        assert gap <= bound * (1 + 1e-9) + 1e-15, f"h={h}: gap {gap:.3g} > bound {bound:.3g}"


def test_crn_vega_and_rho_match_autodiff():
    """Same paths and h = 1e-4: the same estimator up to the O(h^2) Taylor term and the
    few paths within ~h |dS_T/dtheta| of the strike (~1e-5 relative), not a statistical
    comparison."""
    Z = normals(jax.random.PRNGKey(11), N)
    ad = ad_greeks(*ARGS, Z)
    np.testing.assert_allclose(fd_first("sigma", *ARGS, 1e-4, Z, Z), ad["vega"], rtol=1e-4)
    np.testing.assert_allclose(fd_first("r", *ARGS, 1e-4, Z, Z), ad["rho"], rtol=1e-4)


def test_crn_gamma_matches_its_expectation():
    """With CRN, the second difference has expectation [C(S0+h) - 2C(S0) + C(S0-h)]/h^2
    with C the exact Black-Scholes price: the FD of the exact prices. The estimate must
    match that within 4 SE (SE from the per-path second differences); its O(h^2) gap to
    the true gamma is deterministic (checked: halving h quarters it)."""
    S0, sigma, r, T, K = ARGS
    Z = normals(jax.random.PRNGKey(12), 1_000_000)
    h = 1.0
    g = fd_gamma(*ARGS, h, Z, Z, Z)
    x = (discounted_payoffs(S0 + h, sigma, r, T, K, Z) - 2 * discounted_payoffs(S0, sigma, r, T, K, Z)
         + discounted_payoffs(S0 - h, sigma, r, T, K, Z)) / h**2
    np.testing.assert_allclose(g, jnp.mean(x), rtol=1e-10)

    def target(h):
        return float((bs_price(S0 + h, K, r, sigma, T) - 2 * bs_price(S0, K, r, sigma, T)
                      + bs_price(S0 - h, K, r, sigma, T)) / h**2)

    assert_within_se(g, jnp.std(x, ddof=1) / np.sqrt(x.shape[0]), target(h), reason="CRN FD gamma")
    exact = float(bs_greeks(S0, K, r, sigma, T)["gamma"])
    ratio = (target(h) - exact) / (target(h / 2) - exact)
    assert abs(ratio - 4.0) < 0.05, ratio


def test_vectorised_bumping_equals_loop():
    """Same 7 bumped pricings on the same Z; only the dispatch differs."""
    Z = normals(jax.random.PRNGKey(14), N)
    for h in (1e-2, 1e-4):
        loop = fd_greeks_loop(*ARGS, h, Z)
        vec = fd_greeks_vmap(*ARGS, h, Z)
        for g in ("price", "delta", "vega", "rho"):
            np.testing.assert_allclose(vec[g], loop[g], rtol=0, atol=1e-10,
                                       err_msg=f"{g}, h={h}")


def test_independent_seeds_small_h_is_useless():
    """Without CRN, a tiny bump divides pure noise by 2h. With h = 1e-3 the estimator's
    standard error, sqrt(var(up) + var(down)) / (2h sqrt(N)), is ~20, against a true
    delta of 0.64; with CRN it is below 0.01. Checked on the SEs, which are
    estimated from 200,000 paths each and so are accurate to ~1%, rather than on one
    noisy estimate (|estimate - delta| > 1 would itself fail ~4% of the time)."""
    S0, sigma, r, T, K = ARGS
    k1, k2 = jax.random.split(jax.random.PRNGKey(13))
    Zu, Zd = normals(k1, N), normals(k2, N)
    h = 1e-3
    up, dn = discounted_payoffs(S0 + h, sigma, r, T, K, Zu), discounted_payoffs(S0 - h, sigma, r, T, K, Zd)
    se_indep = jnp.sqrt(up.var(ddof=1) + dn.var(ddof=1)) / (2 * h * np.sqrt(N))
    crn = (discounted_payoffs(S0 + h, sigma, r, T, K, Zu) - discounted_payoffs(S0 - h, sigma, r, T, K, Zu)) / (2 * h)
    se_crn = crn.std(ddof=1) / np.sqrt(N)
    delta = bs_greeks(S0, K, r, sigma, T)["delta"]
    assert se_indep > 10 * delta and se_crn < 0.01 * delta, (se_indep, se_crn)
    np.testing.assert_allclose(fd_delta(*ARGS, h, Zu, Zd), (up.mean() - dn.mean()) / (2 * h),
                               rtol=1e-10)
