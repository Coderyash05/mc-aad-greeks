"""Closed-form Black-Scholes prices and Greeks: the ground truth for every test."""
import jax.numpy as jnp
from jax.scipy.stats import norm


def _d1_d2(S0, K, r, sigma, T):
    d1 = (jnp.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    return d1, d1 - sigma * jnp.sqrt(T)


def bs_price(S0, K, r, sigma, T, kind="call"):
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    df = jnp.exp(-r * T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * df * norm.cdf(d2)
    if kind == "put":
        return K * df * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError(f"unknown option kind: {kind}")


def bs_greeks(S0, K, r, sigma, T, kind="call"):
    """Delta, gamma, vega, rho. Used from day 3 onwards."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    df = jnp.exp(-r * T)
    gamma = norm.pdf(d1) / (S0 * sigma * jnp.sqrt(T))
    vega = S0 * norm.pdf(d1) * jnp.sqrt(T)
    if kind == "call":
        delta = norm.cdf(d1)
        rho = K * T * df * norm.cdf(d2)
    elif kind == "put":
        delta = norm.cdf(d1) - 1.0
        rho = -K * T * df * norm.cdf(-d2)
    else:
        raise ValueError(f"unknown option kind: {kind}")
    return {"delta": delta, "gamma": gamma, "vega": vega, "rho": rho}
