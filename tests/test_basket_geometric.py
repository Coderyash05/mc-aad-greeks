"""Geometric basket: a closed form, so the Monte Carlo basket Greeks can be checked exactly.

The Monte Carlo side is the same code as the arithmetic basket (basket_price_L with
geometric=True changes only the payoff), so passing here validates the Cholesky
correlation plumbing, the vegas and the d(d-1)/2 correlation sensitivities.
"""
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mcgreeks  # noqa: F401
from mcgreeks.basket import (basket_greeks_se, basket_price_L, corr_from_offdiag,
                             example_basket, geometric_basket_greeks,
                             geometric_basket_price)
from mcgreeks.black_scholes import bs_greeks, bs_price

R, T, K = 0.05, 1.0, 100.0
N, N_BATCHES = 200_000, 200


def seed(tag):
    return zlib.crc32(tag.encode())


def test_closed_form_one_asset_is_black_scholes():
    S0, sigma, rho, w = example_basket(1, 0)
    g = geometric_basket_greeks(S0, sigma, rho, R, T, K, w)
    ex = bs_greeks(100.0, K, R, 0.2, T)
    np.testing.assert_allclose(g["price"], bs_price(100.0, K, R, 0.2, T), rtol=1e-12)
    np.testing.assert_allclose(g["delta"][0], ex["delta"], rtol=1e-12)
    np.testing.assert_allclose(g["vega"][0], ex["vega"], rtol=1e-12)


def test_closed_form_perfect_correlation_is_black_scholes():
    """C = all ones, equal sigma and S0: G_T = S_T, so price = BS, delta_i = w_i BS delta
    (dG/dS0_i = w_i G / S0_i) and the vegas sum to the BS vega (all sigmas move together)."""
    d = 6
    w = jnp.asarray(np.random.default_rng(seed("geo-rho1")).uniform(0.5, 1.5, d))
    w = w / w.sum()
    S0, sigma, rho = jnp.full(d, 100.0), jnp.full(d, 0.2), jnp.ones(d * (d - 1) // 2)
    g = geometric_basket_greeks(S0, sigma, rho, R, T, K, w)
    ex = bs_greeks(100.0, K, R, 0.2, T)
    np.testing.assert_allclose(g["price"], bs_price(100.0, K, R, 0.2, T), rtol=1e-12)
    np.testing.assert_allclose(g["delta"], w * ex["delta"], rtol=1e-12)
    np.testing.assert_allclose(g["vega"].sum(), ex["vega"], rtol=1e-12)


def test_closed_form_delta_matches_explicit_formula():
    """dV/dS0_i = e^{-rT} (w_i / S0_i) e^{mu + s^2/2} N(d1): checks the AD of the closed form."""
    from jax.scipy.stats import norm
    S0, sigma, rho, w = example_basket(5, seed("geo-explicit"))
    C = corr_from_offdiag(rho, 5)
    mu = jnp.sum(w * (jnp.log(S0) + (R - 0.5 * sigma**2) * T))
    s2 = T * (w * sigma) @ C @ (w * sigma)
    d1 = (mu - jnp.log(K) + s2) / jnp.sqrt(s2)
    explicit = jnp.exp(-R * T) * w / S0 * jnp.exp(mu + 0.5 * s2) * norm.cdf(d1)
    g = geometric_basket_greeks(S0, sigma, rho, R, T, K, w)
    np.testing.assert_allclose(g["delta"], explicit, rtol=1e-12)


def test_closed_form_corr_sensitivity_matches_bumping():
    """Correlation convention: rho[k] moves BOTH symmetric entries, in the closed form too.
    Central differences of a smooth closed form: error O(h^2) ~ 1e-12, rounding ~1e-16/h."""
    S0, sigma, rho, w = example_basket(4, seed("geo-corr"))
    g = geometric_basket_greeks(S0, sigma, rho, R, T, K, w)
    h = 1e-5
    fd = [(geometric_basket_price(S0, sigma, rho.at[k].add(h), R, T, K, w)
           - geometric_basket_price(S0, sigma, rho.at[k].add(-h), R, T, K, w)) / (2 * h)
          for k in range(rho.shape[0])]
    np.testing.assert_allclose(g["corr"], fd, rtol=1e-7)


@pytest.mark.parametrize("d", [2, 10, 50])
def test_mc_greeks_match_closed_form(d):
    """Every Monte Carlo (pathwise AD) sensitivity within 4 batch-means SE of the exact
    value: price, d deltas, d vegas, d(d-1)/2 correlation sensitivities (1,326 at d = 50).
    See README (E9) for the family-wise false-alarm rate of this many comparisons."""
    S0, sigma, rho, w = example_basket(d, seed(f"geo-basket-{d}"))
    Z = jax.random.normal(jax.random.PRNGKey(seed(f"geo-basket-Z-{d}")), (N, d),
                          dtype=jnp.float64)
    mc = basket_greeks_se(S0, sigma, rho, R, T, K, w, Z, n_batches=N_BATCHES, geometric=True)
    ex = geometric_basket_greeks(S0, sigma, rho, R, T, K, w)
    for k in ("price", "delta", "vega", "corr"):
        est, se = mc[k]
        z = np.abs(np.asarray((est - ex[k]) / se))
        assert np.all(z < 4), f"{k}: max |z| = {z.max():.2f} at index {np.argmax(z)}"


@pytest.mark.parametrize("geometric", [False, True])
def test_mc_perfect_correlation_is_black_scholes(geometric):
    """Correlation 1 and equal vols and spots: arithmetic and geometric basket both equal
    S_T, so price, sum of deltas and sum of vegas must match Black-Scholes within 4 SE.
    The correlation matrix is singular, so the Cholesky factor is given directly:
    L = a column of ones (C = L L^T = all ones)."""
    d = 5
    S0, sigma = jnp.full(d, 100.0), jnp.full(d, 0.2)
    w = jnp.full(d, 1.0 / d)
    L = jnp.zeros((d, d)).at[:, 0].set(1.0)
    Z = jax.random.normal(jax.random.PRNGKey(seed(f"geo-rho1-mc-{geometric}")), (N, d),
                          dtype=jnp.float64)

    def batch(z):
        v, (dS, dsig) = jax.value_and_grad(basket_price_L, argnums=(0, 1))(
            S0, sigma, L, R, T, K, w, z, geometric)
        return jnp.stack([v, dS.sum(), dsig.sum()])

    b = jax.lax.map(batch, Z.reshape(N_BATCHES, -1, d))
    est, se = b.mean(0), b.std(0, ddof=1) / np.sqrt(N_BATCHES)
    ex = bs_greeks(100.0, K, R, 0.2, T)
    exact = np.array([bs_price(100.0, K, R, 0.2, T), ex["delta"], ex["vega"]])
    z = np.abs((np.asarray(est) - exact) / np.asarray(se))
    assert np.all(z < 4), f"|z| (price, delta, vega) = {z}"
