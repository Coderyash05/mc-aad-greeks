"""Deliverable 2: autodiff delta, vega, rho match closed-form Black-Scholes."""
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_greeks
from mcgreeks.greeks_ad import ad_greeks, ad_greeks_se
from mcgreeks.models import normals

N = 200_000
R = 0.05


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("T", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.4])
def test_greeks_match_black_scholes(kind, K, T, sigma):
    seed = zlib.crc32(f"greeks-{kind}-{K}-{T}-{sigma}".encode())
    Z = normals(jax.random.PRNGKey(seed), N)
    est = ad_greeks_se(100.0, sigma, R, T, K, Z, kind=kind)
    exact = bs_greeks(100.0, K, R, sigma, T, kind=kind)
    for g in ("delta", "vega", "rho"):
        mean, se = est[g]
        # 4 SE for the same multiple-testing reason as the pricer tests.
        # The 1e-3 floor covers deep OTM cases where no path pays off, so the
        # estimate is exactly 0 while the true Greek is tiny (e.g. vega ~3e-4).
        assert abs(mean - exact[g]) < 4 * se + 1e-3, f"{g}: {kind} K={K} T={T} sigma={sigma}"


def test_grad_of_mean_equals_mean_of_grads():
    """jax.grad on the full pricer and the per-path vmap give identical numbers."""
    Z = normals(jax.random.PRNGKey(3), N)
    a = ad_greeks(100.0, 0.2, R, 1.0, 100.0, Z)
    b = ad_greeks_se(100.0, 0.2, R, 1.0, 100.0, Z)
    for g in ("delta", "vega", "rho"):
        np.testing.assert_allclose(a[g], b[g][0], rtol=1e-10)


def test_delta_parity():
    """Pathwise delta_call - delta_put = e^{-rT} mean(S_T) / S0, which is ~1."""
    Z = normals(jax.random.PRNGKey(4), N)
    c = ad_greeks(100.0, 0.2, R, 1.0, 100.0, Z, kind="call")["delta"]
    p = ad_greeks(100.0, 0.2, R, 1.0, 100.0, Z, kind="put")["delta"]
    ST_mean_disc = jnp.mean(jnp.exp(-0.5 * 0.2**2 + 0.2 * Z))  # e^{-rT} S_T / S0
    np.testing.assert_allclose(c - p, ST_mean_disc, rtol=1e-10)
    assert abs(c - p - 1.0) < 0.01


def test_pathwise_delta_is_bounded():
    """A call's pathwise delta per path is 0 or S_T/S0 > 0, so the estimate is in (0, 1+)."""
    Z = normals(jax.random.PRNGKey(5), N)
    d = ad_greeks(100.0, 0.2, R, 1.0, 100.0, Z, kind="call")["delta"]
    assert 0.0 < d < 1.05


def test_deep_itm_call_rho_has_zero_variance():
    """If every path finishes in the money, per-path rho = K T e^{-rT}: a constant.

    d/dr [e^{-rT} (S_T - K)] = T e^{-rT} S_T - T e^{-rT} (S_T - K) = K T e^{-rT}.
    The estimator's variance is then zero, which is correct, not a bug.
    """
    Z = normals(jax.random.PRNGKey(6), N)
    K, T = 80.0, 0.25
    mean, se = ad_greeks_se(100.0, 0.1, R, T, K, Z)["rho"]
    assert se < 1e-12
    np.testing.assert_allclose(mean, K * T * jnp.exp(-R * T), rtol=1e-12)
