"""Gamma fixes: LR, mixed pathwise-LR and smoothed autodiff all recover the true gamma."""
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_within_se, smoothed_call_gamma_expectation

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_digital_delta, bs_greeks, bs_price
from mcgreeks.greeks_lr import (lr_delta_samples, lr_digital_delta_parity_samples,
                                lr_gamma_parity_samples, lr_gamma_samples, mean_se,
                                pwlr_gamma_parity_samples, pwlr_gamma_samples,
                                smooth_gamma_samples, smooth_price_samples)
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
        assert_within_se(*mean_se(est), g, reason=f"K={K} T={T}")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_lr_delta_unbiased(kind):
    Z = normals(jax.random.PRNGKey(20), N)
    assert_within_se(*mean_se(lr_delta_samples(100.0, 0.2, R, 1.0, 100.0, Z, kind=kind)),
                     exact(100.0, 1.0, 0.2, kind)["delta"], reason=kind)


def test_lr_gamma_same_for_call_and_put():
    """Call - put payoff is linear in S_T, so its true gamma is 0; LR must agree within SE."""
    Z = normals(jax.random.PRNGKey(21), N)
    diff = (lr_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z, kind="call")
            - lr_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z, kind="put"))
    assert_within_se(*mean_se(diff), 0.0, reason="call - put gamma")


def test_pwlr_has_lower_variance_than_lr():
    Z = normals(jax.random.PRNGKey(22), N)
    lr = jnp.std(lr_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z))
    pw = jnp.std(pwlr_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z))
    assert pw < lr


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_parity_estimators_unbiased(K):
    """LR (parity), pathwise-LR (parity) gamma and LR (parity) digital delta, within 4 SE."""
    Z = normals(jax.random.PRNGKey(zlib.crc32(f"parity-{K}".encode())), N)
    g = exact(K, 1.0, 0.2)["gamma"]
    for est in (lr_gamma_parity_samples(100.0, 0.2, R, 1.0, K, Z),
                pwlr_gamma_parity_samples(100.0, 0.2, R, 1.0, K, Z)):
        assert_within_se(*mean_se(est), g, reason=f"parity gamma K={K}")
    assert_within_se(*mean_se(lr_digital_delta_parity_samples(100.0, 0.2, R, 1.0, K, Z)),
                     bs_digital_delta(100.0, K, R, 0.2, 1.0), reason=f"digital K={K}")


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_parity_switch_picks_the_right_leg(K):
    """K < S0 e^{rT} (80, 100 at T = 1): put leg; otherwise (120) the plain call estimator."""
    Z = normals(jax.random.PRNGKey(25), 1000)
    use_put = K < 100.0 * np.exp(R)
    ST = 100.0 * jnp.exp((R - 0.02) + 0.2 * Z)
    w_lr = (Z**2 - 1 - 0.2 * Z) / (100.0**2 * 0.04)
    put_lr = jnp.exp(-R) * jnp.maximum(K - ST, 0.0) * w_lr
    put_pw = jnp.exp(-R) / 100.0**2 * -1.0 * (ST < K) * ST * (Z / 0.2 - 1.0)
    put_dig = -jnp.exp(-R) * 1.0 * (ST < K) * Z / (100.0 * 0.2)
    pairs = ((lr_gamma_parity_samples, lr_gamma_samples, put_lr),
             (pwlr_gamma_parity_samples, pwlr_gamma_samples, put_pw),
             (lr_digital_delta_parity_samples,
              lambda *a: lr_delta_samples(*a, kind="digital"), put_dig))
    for parity, plain, put in pairs:
        expected = put if use_put else plain(100.0, 0.2, R, 1.0, K, Z)
        np.testing.assert_allclose(parity(100.0, 0.2, R, 1.0, K, Z), expected,
                                   rtol=1e-12, atol=1e-15)


def test_parity_lr_has_lower_variance_in_the_money():
    Z = normals(jax.random.PRNGKey(26), N)
    for plain, parity in ((lr_gamma_samples, lr_gamma_parity_samples),
                          (pwlr_gamma_samples, pwlr_gamma_parity_samples)):
        assert jnp.std(parity(100.0, 0.2, R, 1.0, 80.0, Z)) < jnp.std(plain(100.0, 0.2, R, 1.0, 80.0, Z))
    assert (jnp.std(lr_digital_delta_parity_samples(100.0, 0.2, R, 1.0, 80.0, Z))
            < jnp.std(lr_delta_samples(100.0, 0.2, R, 1.0, 80.0, Z, kind="digital")))


def test_smoothed_price_converges_to_unsmoothed():
    """Path by path, 0 <= eps softplus(x/eps) - max(x, 0) <= eps log 2 (the gap is
    largest at x = 0), and the gap grows with eps (eps softplus(x/eps) is increasing in
    eps: its eps-derivative is softplus(u) - u sigmoid(u) >= 0, u = x/eps). So on the
    same paths the price gap is ordered in eps and bounded by e^{-rT} eps log 2: exact
    inequalities, no tolerance."""
    Z = normals(jax.random.PRNGKey(23), N)
    raw = mc_price(100.0, 0.2, R, 1.0, 100.0, Z)
    eps = (2.0, 0.5, 0.05)
    gaps = [float(smooth_price_samples(100.0, 0.2, R, 1.0, 100.0, Z, e).mean() - raw) for e in eps]
    assert gaps[0] > gaps[1] > gaps[2] > 0
    for e, gap in zip(eps, gaps):
        assert gap <= np.exp(-R) * e * np.log(2) * (1 + 1e-12), (e, gap)


def test_smoothed_gamma_matches_its_expectation():
    """Softplus smoothing (eps = 0.5) is biased: its target is the gamma of the smoothed
    price, computed by quadrature (helpers). The estimator must hit that within 4 SE;
    the bias (-0.10% here) is O(eps^2): halving eps quarters it."""
    Z = normals(jax.random.PRNGKey(24), N)
    target = smoothed_call_gamma_expectation(100.0, 0.2, R, 1.0, 100.0, 0.5)
    assert_within_se(*mean_se(smooth_gamma_samples(100.0, 0.2, R, 1.0, 100.0, Z, 0.5)), target,
                     reason="smoothed gamma vs smoothed target")
    g = exact(100.0, 1.0, 0.2)["gamma"]
    ratio = (target - g) / (smoothed_call_gamma_expectation(100.0, 0.2, R, 1.0, 100.0, 0.25) - g)
    assert abs(ratio - 4.0) < 0.05, ratio
