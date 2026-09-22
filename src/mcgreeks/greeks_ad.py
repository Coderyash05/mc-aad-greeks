"""Greeks by reverse-mode automatic differentiation (pathwise estimators).

jax.grad of the Monte Carlo price, with the random numbers Z held fixed,
is exactly the pathwise estimator:
    delta = e^{-rT} E[ f'(S_T) dS_T/dS0 ],  and likewise for vega and rho.
It is valid here because call and put payoffs are Lipschitz (a kink, no jump).
"""
from functools import partial

import jax
import jax.numpy as jnp

from .pricer import discounted_payoffs, mc_price

# argnums 0, 1, 2 of the pricer are S0, sigma, r
GREEK_ARGS = {"delta": 0, "vega": 1, "rho": 2}


@partial(jax.jit, static_argnames="kind")
def ad_greeks(S0, sigma, r, T, K, Z, kind="call"):
    """Price plus delta, vega, rho from ONE forward + ONE reverse pass."""
    price_fn = partial(mc_price, kind=kind)
    price, (delta, vega, rho) = jax.value_and_grad(price_fn, argnums=(0, 1, 2))(
        S0, sigma, r, T, K, Z
    )
    return {"price": price, "delta": delta, "vega": vega, "rho": rho}


@partial(jax.jit, static_argnames="kind")
def ad_greeks_se(S0, sigma, r, T, K, Z, kind="call"):
    """Greeks and their standard errors, from per-path pathwise derivatives.

    The gradient of the mean equals the mean of per-path gradients, so we
    differentiate one path's discounted payoff, vmap over all paths, and take
    the sample mean and std / sqrt(N). Same numbers as ad_greeks, plus error bars.
    """
    one_path = partial(discounted_payoffs, kind=kind)  # works on a scalar z
    per_path_grad = jax.vmap(
        jax.grad(one_path, argnums=(0, 1, 2)), in_axes=(None, None, None, None, None, 0)
    )
    grads = per_path_grad(S0, sigma, r, T, K, Z)
    n = Z.shape[0]
    out = {}
    for name, g in zip(GREEK_ARGS, grads):
        out[name] = (jnp.mean(g), jnp.std(g, ddof=1) / jnp.sqrt(n))
    return out
