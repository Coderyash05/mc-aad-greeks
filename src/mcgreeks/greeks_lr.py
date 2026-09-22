"""Likelihood-ratio (LR) and mixed pathwise-LR estimators, and smoothed autodiff gamma.

Under GBM, with Z the normal driving S_T:
  LR delta   = e^{-rT} E[ f(S_T) * Z / (S0 sigma sqrt(T)) ]
  LR gamma   = e^{-rT} E[ f(S_T) * (Z^2 - 1 - sigma sqrt(T) Z) / (S0^2 sigma^2 T) ]
  PW-LR gamma= e^{-rT} / S0^2 * E[ f'(S_T) S_T (Z / (sigma sqrt(T)) - 1) ]
LR differentiates the density instead of the payoff, so it needs no smoothness.
PW-LR takes the pathwise delta and applies LR once more. See Glasserman (2003), ch. 7.

Every function returns per-path samples; use `mean_se` to get estimate and SE.
"""
from functools import partial

import jax
import jax.numpy as jnp

from .models import gbm_terminal
from .payoffs import PAYOFFS, call_payoff_smooth, digital_payoff_smooth


def mean_se(x):
    return jnp.mean(x), jnp.std(x, ddof=1) / jnp.sqrt(x.shape[-1])


@partial(jax.jit, static_argnames="kind")
def lr_delta_samples(S0, sigma, r, T, K, Z, kind="call"):
    ST = gbm_terminal(S0, sigma, r, T, Z)
    w = Z / (S0 * sigma * jnp.sqrt(T))
    return jnp.exp(-r * T) * PAYOFFS[kind](ST, K) * w


@partial(jax.jit, static_argnames="kind")
def lr_gamma_samples(S0, sigma, r, T, K, Z, kind="call"):
    ST = gbm_terminal(S0, sigma, r, T, Z)
    w = (Z**2 - 1 - sigma * jnp.sqrt(T) * Z) / (S0**2 * sigma**2 * T)
    return jnp.exp(-r * T) * PAYOFFS[kind](ST, K) * w


@jax.jit
def pwlr_gamma_samples(S0, sigma, r, T, K, Z):
    """Mixed estimator for a call: f'(S_T) = 1{S_T > K}."""
    ST = gbm_terminal(S0, sigma, r, T, Z)
    w = Z / (sigma * jnp.sqrt(T)) - 1.0
    return jnp.exp(-r * T) / S0**2 * (ST > K) * ST * w


def _smooth_call_one_path(S0, sigma, r, T, K, z, eps):
    ST = gbm_terminal(S0, sigma, r, T, z)
    return jnp.exp(-r * T) * call_payoff_smooth(ST, K, eps)


@jax.jit
def smooth_gamma_samples(S0, sigma, r, T, K, Z, eps):
    """Autodiff gamma (grad of grad) of the softplus-smoothed call, per path."""
    g2 = jax.grad(jax.grad(_smooth_call_one_path, argnums=0), argnums=0)
    return jax.vmap(g2, in_axes=(None, None, None, None, None, 0, None))(
        S0, sigma, r, T, K, Z, eps
    )


@jax.jit
def smooth_price_samples(S0, sigma, r, T, K, Z, eps):
    return jax.vmap(_smooth_call_one_path, in_axes=(None,) * 5 + (0, None))(
        S0, sigma, r, T, K, Z, eps
    )


def _smooth_digital_one_path(S0, sigma, r, T, K, z, eps):
    ST = gbm_terminal(S0, sigma, r, T, z)
    return jnp.exp(-r * T) * digital_payoff_smooth(ST, K, eps)


@jax.jit
def smooth_digital_delta_samples(S0, sigma, r, T, K, Z, eps):
    """Autodiff delta of the sigmoid-smoothed digital, per path."""
    g = jax.grad(_smooth_digital_one_path, argnums=0)
    return jax.vmap(g, in_axes=(None, None, None, None, None, 0, None))(
        S0, sigma, r, T, K, Z, eps
    )
