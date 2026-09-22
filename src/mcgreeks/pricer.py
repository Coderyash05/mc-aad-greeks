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


@partial(jax.jit, static_argnames="kind")
def mc_price_se(S0, sigma, r, T, K, Z, kind="call"):
    """Price estimate and its standard error, std / sqrt(N)."""
    x = discounted_payoffs(S0, sigma, r, T, K, Z, kind)
    return jnp.mean(x), jnp.std(x, ddof=1) / jnp.sqrt(x.shape[0])
