"""Digital options: pathwise/autodiff delta fails completely; LR and smoothing work."""
import jax
import jax.numpy as jnp
import numpy as np
from helpers import assert_within_se, smoothed_digital_delta_expectation

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_digital_delta, bs_digital_price
from mcgreeks.greeks_ad import ad_greeks
from mcgreeks.greeks_lr import lr_delta_samples, mean_se, smooth_digital_delta_samples
from mcgreeks.models import normals
from mcgreeks.pricer import discounted_payoffs, mc_price_se

N = 500_000
ARGS = (100.0, 0.2, 0.05, 1.0, 100.0)  # S0, sigma, r, T, K
EXACT_DELTA = float(bs_digital_delta(100.0, 100.0, 0.05, 0.2, 1.0))


def test_digital_price_matches_black_scholes():
    Z = normals(jax.random.PRNGKey(30), N)
    p, se = mc_price_se(*ARGS, Z, kind="digital")
    assert_within_se(p, se, bs_digital_price(100.0, 100.0, 0.05, 0.2, 1.0), reason="price")


def test_autodiff_delta_is_exactly_zero():
    """The payoff is flat on both sides of the jump, so every path's derivative is 0."""
    Z = normals(jax.random.PRNGKey(31), N)
    assert ad_greeks(*ARGS, Z, kind="digital")["delta"] == 0.0


def test_lr_delta_unbiased():
    Z = normals(jax.random.PRNGKey(32), N)
    assert_within_se(*mean_se(lr_delta_samples(*ARGS, Z, kind="digital")), EXACT_DELTA,
                     reason="LR delta")


def test_smoothed_delta_matches_its_expectation():
    """Sigmoid smoothing (eps = 0.5) is biased: its target is d/dS0 of the smoothed
    price, computed exactly by quadrature (helpers). The estimator must hit THAT within
    4 SE. The bias itself (target - true delta, -0.10% at eps = 0.5) is deterministic
    and second order in eps for a symmetric smoothing kernel: halving eps must quarter
    it (ratio 3.994 here; the O(eps^4) term moves it off 4 by ~0.2%)."""
    Z = normals(jax.random.PRNGKey(33), N)
    target = smoothed_digital_delta_expectation(*ARGS, 0.5)
    assert_within_se(*mean_se(smooth_digital_delta_samples(*ARGS, Z, 0.5)), target,
                     reason="smoothed delta vs smoothed target")
    ratio = (target - EXACT_DELTA) / (smoothed_digital_delta_expectation(*ARGS, 0.25) - EXACT_DELTA)
    assert abs(ratio - 4.0) < 0.05, ratio


def test_crn_fd_delta_matches_its_expectation():
    """With common random numbers the central difference with h = 1 has expectation
    E = [D(S0 + h) - D(S0 - h)] / (2h), D = exact digital price: the FD of the exact
    prices. Compared with that within 4 SE (per-path differences give the SE). The gap
    between E and the true delta (-0.034% at h = 1) is deterministic and O(h^2):
    halving h must quarter it."""
    Z = normals(jax.random.PRNGKey(34), N)
    S0, sigma, r, T, K = ARGS
    h = 1.0
    x = (discounted_payoffs(S0 + h, sigma, r, T, K, Z, kind="digital")
         - discounted_payoffs(S0 - h, sigma, r, T, K, Z, kind="digital")) / (2 * h)
    def fd_target(h):
        return float((bs_digital_price(S0 + h, K, r, sigma, T)
                      - bs_digital_price(S0 - h, K, r, sigma, T)) / (2 * h))

    assert_within_se(jnp.mean(x), jnp.std(x, ddof=1) / np.sqrt(N), fd_target(h),
                     reason="CRN FD delta")
    ratio = (fd_target(h) - EXACT_DELTA) / (fd_target(h / 2) - EXACT_DELTA)
    assert abs(ratio - 4.0) < 0.05, ratio
