"""Deliverable 2: autodiff delta, vega, rho match closed-form Black-Scholes."""
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import (RARE_THRESHOLD, assert_count_consistent, assert_rare_event,
                     assert_within_se, exercise_probability)

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_greeks
from mcgreeks.greeks_ad import (ad_greeks, ad_greeks_fwd, ad_greeks_se, strike_deltas_fwd,
                                strike_deltas_rev)
from mcgreeks.models import gbm_terminal, normals
from mcgreeks.pricer import mc_price_strikes

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
    reason = f"{kind} K={K} T={T} sigma={sigma}"
    ST = gbm_terminal(100.0, sigma, R, T, Z)
    n_pay = int(jnp.sum(ST > K if kind == "call" else ST < K))
    p_ex = exercise_probability(100.0, sigma, R, T, K, kind)
    sign = 1.0 if kind == "call" else -1.0
    # Per path, rho = sign K T e^{-rT} 1{exercise}: a function of the exercise indicator
    # only, so the estimate is exactly sign K T e^{-rT} n_pay / N, and its SE comes only
    # from the MINORITY side of the indicator. Rare (docs/TESTING.md) when either side
    # has fewer than 100 paths: deep OTM (few payers) or deep ITM (few non-payers,
    # e.g. call K=80, T=0.25, sigma=0.1, where the SE is ~1e-17: rounding).
    rho_rare = min(n_pay, N - n_pay) < RARE_THRESHOLD
    if rho_rare:
        np.testing.assert_allclose(sign * est["rho"][0], K * T * np.exp(-R * T) * n_pay / N,
                                   rtol=1e-12, atol=1e-300, err_msg=reason)
        if n_pay >= RARE_THRESHOLD:   # few non-payers: check their count
            assert_count_consistent(N - n_pay, N, 1.0 - p_ex, reason)
    if n_pay < RARE_THRESHOLD:
        # Deep OTM: every pathwise derivative is 0 on paths that do not pay; per path,
        # delta = sign e^{-rT} 1{exercise} S_T/S0 has the option's sign. Vega per path
        # has no fixed sign, so with payers present only its zero-payer case is checked.
        assert_rare_event(sign * est["delta"][0], est["delta"][1], n_pay, N, p_ex, reason)
        if n_pay == 0:
            assert est["vega"][0] == 0.0, reason
        return
    for g in ("delta", "vega") + (() if rho_rare else ("rho",)):
        # 4 SE for the same multiple-testing reason as the pricer tests.
        assert_within_se(*est[g], exact[g], reason=f"{g}: {reason}")


def test_grad_of_mean_equals_mean_of_grads():
    """jax.grad on the full pricer and the per-path vmap give identical numbers."""
    Z = normals(jax.random.PRNGKey(3), N)
    a = ad_greeks(100.0, 0.2, R, 1.0, 100.0, Z)
    b = ad_greeks_se(100.0, 0.2, R, 1.0, 100.0, Z)
    for g in ("delta", "vega", "rho"):
        np.testing.assert_allclose(a[g], b[g][0], rtol=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_forward_mode_equals_reverse_mode(kind, K):
    """jvp (forward) and grad (reverse) apply the chain rule in opposite orders to
    the same program: identical derivatives up to floating-point rounding."""
    Z = normals(jax.random.PRNGKey(zlib.crc32(f"fwd-{kind}-{K}".encode())), N)
    rev = ad_greeks(100.0, 0.2, R, 1.0, K, Z, kind=kind)
    fwd = ad_greeks_fwd(100.0, 0.2, R, 1.0, K, Z, kind=kind)
    for g in ("price", "delta", "vega", "rho"):
        np.testing.assert_allclose(fwd[g], rev[g], rtol=1e-10, err_msg=g)


def test_strike_jacobian_forward_equals_reverse():
    """K deltas of K strikes: one jvp == K vjps (chunked or not) == jax.jacrev."""
    Z = normals(jax.random.PRNGKey(8), 20_000)
    strikes = jnp.linspace(70.0, 130.0, 10)
    prices, d_fwd = strike_deltas_fwd(100.0, 0.2, R, 1.0, strikes, Z)
    ref = jax.jacrev(lambda s: mc_price_strikes(s, 0.2, R, 1.0, strikes, Z))(100.0)
    for chunk in (None, 3):
        p_rev, d_rev = strike_deltas_rev(100.0, 0.2, R, 1.0, strikes, Z, chunk=chunk)
        np.testing.assert_allclose(d_rev, d_fwd, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p_rev, prices, rtol=1e-14)
    np.testing.assert_allclose(d_fwd, ref, rtol=0, atol=1e-12)
    # each column entry is the ordinary single-strike pathwise delta
    np.testing.assert_allclose(d_fwd[4], ad_greeks(100.0, 0.2, R, 1.0, strikes[4], Z)["delta"],
                               rtol=1e-12)


def test_delta_parity():
    """Pathwise delta_call - delta_put = e^{-rT} mean(S_T) / S0 path by path (to
    rounding); that mean estimates 1 (martingale), within 4 of its own SE."""
    Z = normals(jax.random.PRNGKey(4), N)
    c = ad_greeks(100.0, 0.2, R, 1.0, 100.0, Z, kind="call")["delta"]
    p = ad_greeks(100.0, 0.2, R, 1.0, 100.0, Z, kind="put")["delta"]
    x = jnp.exp(-0.5 * 0.2**2 + 0.2 * Z)  # e^{-rT} S_T / S0 per path
    np.testing.assert_allclose(c - p, jnp.mean(x), rtol=1e-10)
    assert_within_se(c - p, jnp.std(x, ddof=1) / np.sqrt(N), 1.0, reason="delta parity")


def test_pathwise_delta_is_bounded():
    """A call's pathwise delta per path is e^{-rT} 1{S_T > K} S_T/S0, which lies in
    [0, e^{-rT} S_T/S0], so 0 < estimate <= e^{-rT} mean(S_T)/S0 exactly (to rounding)."""
    Z = normals(jax.random.PRNGKey(5), N)
    d = ad_greeks(100.0, 0.2, R, 1.0, 100.0, Z, kind="call")["delta"]
    upper = jnp.mean(jnp.exp(-0.5 * 0.2**2 + 0.2 * Z))
    assert 0.0 < d <= upper * (1 + 1e-12)


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
