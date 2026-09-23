"""Finite-difference Greeks behave as theory predicts."""
import jax
import numpy as np

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_greeks
from mcgreeks.greeks_ad import ad_greeks
from mcgreeks.greeks_fd import (fd_delta, fd_first, fd_gamma, fd_greeks_loop,
                                fd_greeks_vmap)
from mcgreeks.models import normals

N = 200_000
ARGS = (100.0, 0.2, 0.05, 1.0, 100.0)  # S0, sigma, r, T, K


def test_crn_delta_converges_to_pathwise_delta():
    """With common random numbers, FD delta -> the autodiff (pathwise) delta as h -> 0.

    Paths away from the strike contribute exactly their pathwise derivative;
    only paths within h of the kink differ, so the gap shrinks like O(h).
    """
    Z = normals(jax.random.PRNGKey(10), N)
    ad = ad_greeks(*ARGS, Z)["delta"]
    gaps = [abs(fd_delta(*ARGS, h, Z, Z) - ad) for h in (1.0, 0.1, 0.01)]
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-4


def test_crn_vega_and_rho_match_autodiff():
    Z = normals(jax.random.PRNGKey(11), N)
    ad = ad_greeks(*ARGS, Z)
    np.testing.assert_allclose(fd_first("sigma", *ARGS, 1e-4, Z, Z), ad["vega"], rtol=1e-4)
    np.testing.assert_allclose(fd_first("r", *ARGS, 1e-4, Z, Z), ad["rho"], rtol=1e-4)


def test_crn_gamma_close_to_black_scholes_at_moderate_h():
    Z = normals(jax.random.PRNGKey(12), 1_000_000)
    g = fd_gamma(*ARGS, 1.0, Z, Z, Z)
    exact = bs_greeks(100.0, 100.0, 0.05, 0.2, 1.0)["gamma"]
    assert abs(g - exact) / exact < 0.05


def test_vectorised_bumping_equals_loop():
    """Same 7 bumped pricings on the same Z; only the dispatch differs."""
    Z = normals(jax.random.PRNGKey(14), N)
    for h in (1e-2, 1e-4):
        loop = fd_greeks_loop(*ARGS, h, Z)
        vec = fd_greeks_vmap(*ARGS, h, Z)
        for g in ("price", "delta", "vega", "rho"):
            np.testing.assert_allclose(vec[g], loop[g], rtol=0, atol=1e-10,
                                       err_msg=f"{g}, h={h}")


def test_independent_seeds_small_h_is_useless():
    """Without CRN, a tiny bump divides pure noise by 2h: error explodes."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(13))
    d = fd_delta(*ARGS, 1e-3, normals(k1, N), normals(k2, N))
    exact = bs_greeks(100.0, 100.0, 0.05, 0.2, 1.0)["delta"]
    assert abs(d - exact) > 1.0  # the true delta is ~0.64
