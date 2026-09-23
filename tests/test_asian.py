"""Asian call: geometric average vs its closed form; arithmetic AD vs CRN bumping."""
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_within_se

import mcgreeks  # noqa: F401
from mcgreeks.asian import asian_greeks_se, asian_payoffs, asian_price, geometric_asian_price
from mcgreeks.black_scholes import bs_price

R, T = 0.05, 1.0
N = 200_000


def _Z(tag, M, n=N):
    key = jax.random.PRNGKey(zlib.crc32(tag.encode()))
    return jax.random.normal(key, (M, n), dtype=jnp.float64)


def test_closed_form_one_date_is_black_scholes():
    for K in (80.0, 100.0, 120.0):
        np.testing.assert_allclose(geometric_asian_price(100.0, K, R, 0.2, T, 1),
                                   bs_price(100.0, K, R, 0.2, T), rtol=1e-12)


def test_closed_form_below_european():
    """Averaging lowers volatility, so the Asian is cheaper than the European call."""
    assert geometric_asian_price(100.0, 100.0, R, 0.2, T, 12) < bs_price(100.0, 100.0, R, 0.2, T)


@pytest.mark.parametrize("M", [12, 52])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_geometric_price_matches_closed_form(M, K):
    x = asian_payoffs(100.0, 0.2, R, T, K, _Z(f"geo-price-{M}-{K}", M), geometric=True)
    assert_within_se(jnp.mean(x), jnp.std(x, ddof=1) / jnp.sqrt(N),
                     geometric_asian_price(100.0, K, R, 0.2, T, M), reason=f"M={M} K={K}")


@pytest.mark.parametrize("M", [12, 52])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_geometric_greeks_match_closed_form(M, K):
    """Pathwise AD delta and vega vs jax.grad of the closed form, within 4 SE."""
    est = asian_greeks_se(100.0, 0.2, R, T, K, _Z(f"geo-greeks-{M}-{K}", M), geometric=True)
    exact = jax.grad(geometric_asian_price, argnums=(0, 3))(100.0, K, R, 0.2, T, M)
    for (name, (m, se)), ex in zip(est.items(), exact):
        assert_within_se(m, se, ex, reason=f"{name} M={M} K={K}")


@pytest.mark.parametrize("M", [12, 52])
def test_arithmetic_ad_matches_crn_bumping(M):
    """Same paths, central differences with h = 1e-5: the same estimator, so they
    agree to ~1e-5 relative (kink paths within h of the strike, and rounding / 2h)."""
    Z = _Z(f"arith-fd-{M}", M, 100_000)
    h = 1e-5
    ad = jax.grad(asian_price, argnums=(0, 1))(100.0, 0.2, R, T, 100.0, Z)
    fd_delta = (asian_price(100.0 + h, 0.2, R, T, 100.0, Z)
                - asian_price(100.0 - h, 0.2, R, T, 100.0, Z)) / (2 * h)
    fd_vega = (asian_price(100.0, 0.2 + h, R, T, 100.0, Z)
               - asian_price(100.0, 0.2 - h, R, T, 100.0, Z)) / (2 * h)
    np.testing.assert_allclose(ad[0], fd_delta, rtol=1e-5)
    np.testing.assert_allclose(ad[1], fd_vega, rtol=1e-5)


def test_arithmetic_above_geometric():
    """AM >= GM path by path, so the arithmetic Asian is worth more on the same paths."""
    Z = _Z("am-gm", 12, 50_000)
    assert asian_price(100.0, 0.2, R, T, 100.0, Z) > asian_price(100.0, 0.2, R, T, 100.0, Z,
                                                                 geometric=True)


def test_unrolled_scan_is_the_same_program():
    """unroll=True (used only for cost_analysis) must not change the numbers."""
    Z = _Z("unroll", 12, 20_000)
    np.testing.assert_allclose(asian_price(100.0, 0.2, R, T, 100.0, Z, unroll=True),
                               asian_price(100.0, 0.2, R, T, 100.0, Z), rtol=1e-14)
