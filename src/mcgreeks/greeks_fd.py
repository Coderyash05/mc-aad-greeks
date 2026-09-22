"""Finite-difference (bump-and-reprice) Greeks: the baseline autodiff is compared to.

Each function takes the random numbers for every repricing explicitly:
  - common random numbers (CRN): pass the SAME Z to every call
  - independent seeds:           pass a DIFFERENT Z to every call
Making the choice explicit keeps the comparison honest; CRN is what a
practitioner would actually do, independent seeds is the naive version.
"""
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
