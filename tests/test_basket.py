"""Basket option: no closed form, so correctness is checked three other ways."""
import jax
import jax.numpy as jnp
import numpy as np

import mcgreeks  # noqa: F401
from mcgreeks.basket import (basket_dK, basket_greeks, basket_price, corr_from_offdiag,
                             offdiag_from_corr)
from mcgreeks.black_scholes import bs_greeks, bs_price

R, T, K = 0.05, 1.0, 100.0


def setup(d, n=200_000, seed=40, corr=0.5):
    S0 = jnp.full(d, 100.0)
    sigma = jnp.linspace(0.15, 0.35, d) if d > 1 else jnp.array([0.2])
    rho = jnp.full(d * (d - 1) // 2, corr)
    w = jnp.full(d, 1.0 / d)
    Z = jax.random.normal(jax.random.PRNGKey(seed), (n, d), dtype=jnp.float64)
    return S0, sigma, rho, w, Z


def test_corr_roundtrip():
    C = corr_from_offdiag(jnp.array([0.1, 0.2, 0.3]), 3)
    np.testing.assert_allclose(C, C.T)
    np.testing.assert_allclose(jnp.diag(C), 1.0)
    np.testing.assert_allclose(offdiag_from_corr(C), [0.1, 0.2, 0.3])


def test_one_asset_basket_is_black_scholes():
    """With d = 1 the basket is a vanilla call: price, delta, vega must match."""
    S0, sigma, rho, w, Z = setup(1, n=1_000_000)
    g = basket_greeks(S0, sigma, rho, R, T, K, w, Z)
    ex = bs_greeks(100.0, K, R, 0.2, T)
    np.testing.assert_allclose(g["price"], bs_price(100.0, K, R, 0.2, T), rtol=5e-3)
    np.testing.assert_allclose(g["delta"][0], ex["delta"], rtol=5e-3)
    np.testing.assert_allclose(g["vega"][0], ex["vega"], rtol=5e-3)


def test_autodiff_matches_crn_bumping():
    """Same random numbers + tiny bumps = the same estimator, to ~1e-5 relative."""
    d = 4
    S0, sigma, rho, w, Z = setup(d)
    g = basket_greeks(S0, sigma, rho, R, T, K, w, Z)
    h = 1e-5
    for name, x, idx in (("delta", S0, 0), ("vega", sigma, 1), ("corr", rho, 2)):
        for i in range(x.shape[0]):
            e = jnp.zeros_like(x).at[i].set(h)
            args_up = [S0, sigma, rho]
            args_dn = [S0, sigma, rho]
            args_up[idx] = x + e
            args_dn[idx] = x - e
            fd = (basket_price(*args_up, R, T, K, w, Z)
                  - basket_price(*args_dn, R, T, K, w, Z)) / (2 * h)
            np.testing.assert_allclose(g[name][i], fd, rtol=1e-4, atol=1e-6,
                                       err_msg=f"{name}[{i}]")


def test_euler_homogeneity():
    """Price is homogeneous of degree 1 in (S0, K): V = sum_i S0_i delta_i + K dV/dK.

    This holds path by path, so it holds exactly for the estimator, a strong
    model-free check on the deltas.
    """
    S0, sigma, rho, w, Z = setup(5)
    g = basket_greeks(S0, sigma, rho, R, T, K, w, Z)
    dK = basket_dK(S0, sigma, rho, R, T, K, w, Z)
    np.testing.assert_allclose(jnp.sum(S0 * g["delta"]) + K * dK, g["price"], rtol=1e-10)


def test_higher_correlation_raises_price():
    """More correlation = less diversification = more basket variance = higher call price."""
    S0, sigma, rho, w, Z = setup(5)
    g = basket_greeks(S0, sigma, rho, R, T, K, w, Z)
    assert jnp.all(g["corr"] > 0)
