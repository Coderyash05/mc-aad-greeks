"""Finite-difference (bump-and-reprice) Greeks: the baseline autodiff is compared to.

Each function takes the random numbers for every repricing explicitly:
  - common random numbers (CRN): pass the SAME Z to every call
  - independent seeds:           pass a DIFFERENT Z to every call
Making the choice explicit keeps the comparison honest; CRN is what a
practitioner would actually do, independent seeds is the naive version.
"""
from functools import partial

import jax
import jax.numpy as jnp

from .pricer import mc_price

_PARAMS = ("S0", "sigma", "r")


def _bump(S0, sigma, r, name, h):
    p = {"S0": S0, "sigma": sigma, "r": r}
    p[name] = p[name] + h
    return p["S0"], p["sigma"], p["r"]


def fd_first(name, S0, sigma, r, T, K, h, Z_up, Z_down, kind="call"):
    """Central difference dV/d(name): delta ('S0'), vega ('sigma') or rho ('r')."""
    if name not in _PARAMS:
        raise ValueError(f"name must be one of {_PARAMS}")
    up = mc_price(*_bump(S0, sigma, r, name, h), T, K, Z_up, kind=kind)
    down = mc_price(*_bump(S0, sigma, r, name, -h), T, K, Z_down, kind=kind)
    return (up - down) / (2 * h)


def fd_delta(S0, sigma, r, T, K, h, Z_up, Z_down, kind="call"):
    return fd_first("S0", S0, sigma, r, T, K, h, Z_up, Z_down, kind)


def fd_gamma(S0, sigma, r, T, K, h, Z_up, Z_mid, Z_down, kind="call"):
    """Second central difference in S0."""
    up = mc_price(S0 + h, sigma, r, T, K, Z_up, kind=kind)
    mid = mc_price(S0, sigma, r, T, K, Z_mid, kind=kind)
    down = mc_price(S0 - h, sigma, r, T, K, Z_down, kind=kind)
    return (up - 2 * mid + down) / h**2


# ---- price + delta, vega, rho by CRN central differences: 7 pricings ---------------
def fd_greeks_loop(S0, sigma, r, T, K, h, Z, kind="call"):
    """Base price plus 6 bumped pricings, one jit-compiled call each (Python loop)."""
    out = {"price": mc_price(S0, sigma, r, T, K, Z, kind=kind)}
    for greek, name in (("delta", "S0"), ("vega", "sigma"), ("rho", "r")):
        out[greek] = fd_first(name, S0, sigma, r, T, K, h, Z, Z, kind)
    return out


@partial(jax.jit, static_argnames="kind")
def fd_greeks_vmap(S0, sigma, r, T, K, h, Z, kind="call"):
    """The same 7 pricings, vmapped over the bumped parameter sets in one program.

    Rows: base, (S0, sigma, r) + h e_i, (S0, sigma, r) - h e_i. Same bumps and Z as
    fd_greeks_loop, so the Greeks agree to rounding.
    """
    theta = jnp.array([S0, sigma, r], dtype=jnp.float64)
    E = h * jnp.eye(3)
    thetas = jnp.concatenate([theta[None], theta + E, theta - E])
    v = jax.vmap(lambda th: mc_price(th[0], th[1], th[2], T, K, Z, kind=kind))(thetas)
    d = (v[1:4] - v[4:7]) / (2 * h)
    return {"price": v[0], "delta": d[0], "vega": d[1], "rho": d[2]}
