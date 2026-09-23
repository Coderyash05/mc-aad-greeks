"""Deliverable 1: the MC pricer matches Black-Scholes within 4 standard errors."""
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import (RARE_THRESHOLD, assert_chi2, assert_rare_event, assert_within_se,
                     exercise_probability)

import mcgreeks  # noqa: F401  (enables float64)
from mcgreeks.black_scholes import bs_price
from mcgreeks.models import gbm_terminal, normals
from mcgreeks.pricer import discounted_payoffs, mc_price, mc_price_se

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
    reason = f"{kind} K={K} T={T} sigma={sigma}"
    # 4 SE, not 3: with 54 cases, a 3-SE rule fails a correct pricer ~14% of the
    # time by chance (0.27% per case); 4 SE is ~0.3% for the 54 (docs/TESTING.md).
    # Deep out of the money (e.g. put, K=80, T=0.25, sigma=0.1: exercise is a
    # ~4.5-sigma event) fewer than 100 paths pay, the SE is unreliable, and
    # structural facts are checked instead.
    ST = gbm_terminal(100.0, sigma, R, T, Z)
    n_pay = int(jnp.sum(ST > K if kind == "call" else ST < K))
    if n_pay < RARE_THRESHOLD:
        assert_rare_event(price, se, n_pay, N, exercise_probability(100.0, sigma, R, T, K, kind),
                          reason)
    else:
        assert_within_se(price, se, exact, reason=reason)


def test_z_scores_look_standard_normal():
    """Across 60 independent cases, z = (MC - BS)/SE should be ~N(0, 1): mean z within
    4 SE (1/sqrt(60)) of 0, and sum z^2 within the chi^2_60 band of the same size."""
    keys = jax.random.split(jax.random.PRNGKey(7), 60)
    z = []
    for i, key in enumerate(keys):
        K = 80.0 + 40.0 * (i % 5) / 4
        Z = normals(key, 50_000)
        p, se = mc_price_se(100.0, 0.2, R, 1.0, K, Z)
        z.append(float((p - bs_price(100.0, K, R, 0.2, 1.0)) / se))
    z = np.array(z)
    assert_within_se(z.mean(), 1 / np.sqrt(len(z)), 0.0, reason="mean z")
    assert_chi2(np.sum(z**2), len(z), reason="sum z^2")


def test_put_call_parity():
    """Same Z for both: C - P = e^{-rT} mean(S_T) - K e^{-rT} holds path by path (to
    rounding), and e^{-rT} mean(S_T) estimates S0 (martingale) within 4 of ITS SE."""
    Z = normals(jax.random.PRNGKey(1), N)
    S0, K, sigma, T = 100.0, 100.0, 0.2, 1.0
    c = mc_price(S0, sigma, R, T, K, Z, kind="call")
    p = mc_price(S0, sigma, R, T, K, Z, kind="put")
    disc_ST = jnp.exp(-R * T) * gbm_terminal(S0, sigma, R, T, Z)
    np.testing.assert_allclose(c - p, disc_ST.mean() - K * jnp.exp(-R * T), rtol=1e-12)
    assert_within_se(disc_ST.mean(), disc_ST.std(ddof=1) / np.sqrt(N), S0,
                     reason="discounted S_T is a martingale")


def test_standard_error_scales_as_inverse_sqrt_n():
    """se(10k) / se(1M) = 10 up to the noise in the sample standard deviations.

    The sample sd s of n iid values has relative sd ~ sqrt((kappa - 1) / (4n)) (delta
    method; kappa = kurtosis of the payoff), so the ratio has relative sd
    sqrt((kappa - 1)/4 * (1/n_small + 1/n_big)); allow 4 of those.
    """
    key = jax.random.PRNGKey(2)
    Zs, Zb = normals(key, 10_000), normals(key, 1_000_000)
    _, se_small = mc_price_se(100.0, 0.2, R, 1.0, 100.0, Zs)
    _, se_big = mc_price_se(100.0, 0.2, R, 1.0, 100.0, Zb)
    x = np.asarray(discounted_payoffs(100.0, 0.2, R, 1.0, 100.0, Zb))
    kappa = np.mean((x - x.mean()) ** 4) / np.var(x) ** 2
    rel_sd = np.sqrt((kappa - 1) / 4 * (1 / 10_000 + 1 / 1_000_000))
    assert_within_se(se_small / se_big, 10.0 * rel_sd, 10.0, reason="SE ratio")
