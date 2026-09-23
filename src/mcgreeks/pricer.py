"""Vectorised, jit-compiled Monte Carlo pricer for European options under GBM."""
from functools import partial

import jax
import jax.numpy as jnp

from .models import gbm_terminal
from .payoffs import PAYOFFS


def discounted_payoffs(S0, sigma, r, T, K, Z, kind="call"):
    ST = gbm_terminal(S0, sigma, r, T, Z)
    return jnp.exp(-r * T) * PAYOFFS[kind](ST, K)


@partial(jax.jit, static_argnames="kind")
def mc_price(S0, sigma, r, T, K, Z, kind="call"):
    """Scalar price estimate. This is the function we differentiate later."""
    return jnp.mean(discounted_payoffs(S0, sigma, r, T, K, Z, kind))


@jax.jit
def mc_price_strikes(S0, sigma, r, T, strikes, Z):
    """Calls at every strike in `strikes` on the same paths: K prices, shape (K,).

    Many outputs, one input (S0): the regime where forward-mode AD wins.
    """
    ST = gbm_terminal(S0, sigma, r, T, Z)
    return jnp.exp(-r * T) * jnp.mean(jnp.maximum(ST[:, None] - strikes[None, :], 0.0), axis=0)


@partial(jax.jit, static_argnames="kind")
def mc_price_se(S0, sigma, r, T, K, Z, kind="call"):
    """Price estimate and its standard error, std / sqrt(N)."""
    x = discounted_payoffs(S0, sigma, r, T, K, Z, kind)
    return jnp.mean(x), jnp.std(x, ddof=1) / jnp.sqrt(x.shape[0])
