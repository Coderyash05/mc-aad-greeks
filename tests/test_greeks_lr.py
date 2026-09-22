"""Gamma fixes: LR, mixed pathwise-LR and smoothed autodiff all recover the true gamma."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_greeks, bs_price
from mcgreeks.greeks_lr import (lr_delta_samples, lr_gamma_samples, mean_se,
                                pwlr_gamma_samples, smooth_gamma_samples,
                                smooth_price_samples)
from mcgreeks.models import normals
from mcgreeks.pricer import mc_price

N = 500_000
R = 0.05


def exact(K, T, sigma, kind="call"):
    return bs_greeks(100.0, K, R, sigma, T, kind=kind)


@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("T", [0.5, 1.0])
def test_lr_and_pwlr_gamma_unbiased(K, T):
    Z = normals(jax.random.PRNGKey(int(K * 10 + T * 100)), N)
    g = exact(K, T, 0.2)["gamma"]
    for est in (lr_gamma_samples(100.0, 0.2, R, T, K, Z),
                pwlr_gamma_samples(100.0, 0.2, R, T, K, Z)):
        m, se = mean_se(est)
        assert abs(m - g) < 4 * se


@pytest.mark.parametrize("kind", ["call", "put"])
def test_lr_delta_unbiased(kind):
    Z = normals(jax.random.PRNGKey(20), N)
    m, se = mean_se(lr_delta_samples(100.0, 0.2, R, 1.0, 100.0, Z, kind=kind))
    assert abs(m - exact(100.0, 1.0, 0.2, kind)["delta"]) < 4 * se


def test_lr_gamma_same_for_call_and_put():
    """Call - put payoff is linear in S_T, so its true gamma is 0; LR must agree within SE."""
    Z = normals(jax.random.PRNGKey(21), N)
    diff = (lr_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z, kind="call")
            - lr_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z, kind="put"))
    m, se = mean_se(diff)
    assert abs(m) < 4 * se


def test_pwlr_has_lower_variance_than_lr():
    Z = normals(jax.random.PRNGKey(22), N)
    lr = jnp.std(lr_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z))
    pw = jnp.std(pwlr_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z))
    assert pw < lr


def test_smoothed_price_converges_to_unsmoothed():
    Z = normals(jax.random.PRNGKey(23), N)
    raw = mc_price(100.0, 0.2, R, 1.0, 100.0, Z)
    gaps = [abs(smooth_price_samples(100.0, 0.2, R, 1.0, 100.0, Z, e).mean() - raw)
            for e in (2.0, 0.5, 0.05)]
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-3


def test_smoothed_gamma_nonzero_and_close_for_small_eps():
    Z = normals(jax.random.PRNGKey(24), N)
    m, se = mean_se(smooth_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z, 0.5))
    g = exact(100.0, 1.0, 0.2)["gamma"]
    assert m > 0
    assert abs(m - g) < 4 * se + 1e-4  # small smoothing bias allowed
