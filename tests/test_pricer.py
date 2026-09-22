"""Deliverable 1: the MC pricer matches Black-Scholes within 3 standard errors."""
import zlib

import jax
import jax.numpy as jnp
import pytest

import mcgreeks  # noqa: F401  (enables float64)
from mcgreeks.black_scholes import bs_price
from mcgreeks.models import normals
from mcgreeks.pricer import mc_price, mc_price_se

N = 200_000
R = 0.05


def test_float64_enabled():
    assert normals(jax.random.PRNGKey(0), 10).dtype == jnp.float64


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("T", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.4])
def test_matches_black_scholes(kind, K, T, sigma):
    # Deterministic seed per case. (Python's hash() of a str changes every run.)
    seed = zlib.crc32(f"{kind}-{K}-{T}-{sigma}".encode())
    Z = normals(jax.random.PRNGKey(seed), N)
    price, se = mc_price_se(100.0, sigma, R, T, K, Z, kind=kind)
    exact = bs_price(100.0, K, R, sigma, T, kind=kind)
    # 4 SE, not 3: with 54 cases, a 3-SE rule fails a correct pricer ~14% of the
    # time by chance (0.27% per case). 4 SE is ~0.3% for the whole suite.
    # The 3-SE criterion is checked on the E1 table, where the z-scores are reported.
    # The 1e-4 floor covers deep out-of-the-money cases (e.g. put, K=80, T=0.25,
    # sigma=0.1): exercise is a ~4.5-sigma event, no path finishes in the money,
    # so the estimate is 0 with SE 0 while the true price is ~1e-6. That is a
    # rare-event problem (fix: importance sampling), not a pricer bug.
    assert abs(price - exact) < 4 * se + 1e-4, f"{kind} K={K} T={T} sigma={sigma}"


def test_z_scores_look_standard_normal():
    """Across many independent cases, (MC - BS)/SE should be ~N(0, 1)."""
    keys = jax.random.split(jax.random.PRNGKey(7), 60)
    z = []
    for i, key in enumerate(keys):
        K = 80.0 + 40.0 * (i % 5) / 4
        Z = normals(key, 50_000)
        p, se = mc_price_se(100.0, 0.2, R, 1.0, K, Z)
        z.append((p - bs_price(100.0, K, R, 0.2, 1.0)) / se)
    z = jnp.array(z)
    assert abs(z.mean()) < 0.5 and 0.6 < z.std() < 1.4


def test_put_call_parity():
    # Same Z for both, so C - P = e^{-rT} mean(S_T) - K e^{-rT} path by path.
    Z = normals(jax.random.PRNGKey(1), N)
    S0, K, sigma, T = 100.0, 100.0, 0.2, 1.0
    c = mc_price(S0, sigma, R, T, K, Z, kind="call")
    p = mc_price(S0, sigma, R, T, K, Z, kind="put")
    parity = S0 - K * jnp.exp(-R * T)
    _, se = mc_price_se(S0, sigma, R, T, K, Z, kind="call")
    assert abs((c - p) - parity) < 3 * se


def test_standard_error_scales_as_inverse_sqrt_n():
    key = jax.random.PRNGKey(2)
    _, se_small = mc_price_se(100.0, 0.2, R, 1.0, 100.0, normals(key, 10_000))
    _, se_big = mc_price_se(100.0, 0.2, R, 1.0, 100.0, normals(key, 1_000_000))
    assert se_small / se_big == pytest.approx(10.0, rel=0.05)
