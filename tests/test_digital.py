"""Digital options: pathwise/autodiff delta fails completely; LR and smoothing work."""
import jax
import numpy as np

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_digital_delta, bs_digital_price
from mcgreeks.greeks_ad import ad_greeks
from mcgreeks.greeks_fd import fd_delta
from mcgreeks.greeks_lr import lr_delta_samples, mean_se, smooth_digital_delta_samples
from mcgreeks.models import normals
from mcgreeks.pricer import mc_price_se

N = 500_000
ARGS = (100.0, 0.2, 0.05, 1.0, 100.0)  # S0, sigma, r, T, K
EXACT_DELTA = float(bs_digital_delta(100.0, 100.0, 0.05, 0.2, 1.0))


def test_digital_price_matches_black_scholes():
    Z = normals(jax.random.PRNGKey(30), N)
    p, se = mc_price_se(*ARGS, Z, kind="digital")
    assert abs(p - bs_digital_price(100.0, 100.0, 0.05, 0.2, 1.0)) < 4 * se


def test_autodiff_delta_is_exactly_zero():
    """The payoff is flat on both sides of the jump, so every path's derivative is 0."""
    Z = normals(jax.random.PRNGKey(31), N)
    assert ad_greeks(*ARGS, Z, kind="digital")["delta"] == 0.0


def test_lr_delta_unbiased():
    Z = normals(jax.random.PRNGKey(32), N)
    m, se = mean_se(lr_delta_samples(*ARGS, Z, kind="digital"))
    assert abs(m - EXACT_DELTA) < 4 * se


def test_smoothed_delta_close_for_moderate_eps():
    Z = normals(jax.random.PRNGKey(33), N)
    m, se = mean_se(smooth_digital_delta_samples(*ARGS, Z, 0.5))
    assert abs(m - EXACT_DELTA) < 4 * se + 1e-4


def test_crn_fd_delta_works_but_is_noisy():
    Z = normals(jax.random.PRNGKey(34), N)
    d = fd_delta(*ARGS, 1.0, Z, Z, kind="digital")
    np.testing.assert_allclose(d, EXACT_DELTA, rtol=0.05)
