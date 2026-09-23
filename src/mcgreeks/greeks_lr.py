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


# ---- parity-switched LR / pathwise-LR ------------------------------------------
# Put-call parity: C - P = S0 - K e^{-rT} is linear in S0, so the call and the put
# have the same gamma, exactly. Likewise 1{S_T > K} + 1{S_T < K} = 1 almost surely,
# so the digital call delta is minus the digital put delta. Either payoff gives an
# unbiased LR / pathwise-LR estimator of the same Greek, but not the same variance:
# the LR estimator's second moment is E[f(S_T)^2 w^2], and in the money a call pays
# on most paths (f ~ S_T - K large and nonzero), while the put pays only in the
# tail. Using the out-of-the-money leg drops the linear part (S_T - K), whose true
# gamma is 0 (and, for the digital, the constant 1, whose true delta is 0) but whose
# LR estimate is pure noise. It is parity used as a control variate inside the LR
# estimator (Glasserman 2004, ch. 4 on control variates, ch. 7 on LR).
# The switch is at the forward F = S0 e^{rT}: K < F -> put (the OTM leg in forward terms).
def _use_put(S0, r, T, K):
    return K < S0 * jnp.exp(r * T)


@jax.jit
def lr_gamma_parity_samples(S0, sigma, r, T, K, Z):
    """LR gamma of the call via the put payoff when K < S0 e^{rT} (same gamma by parity)."""
    ST = gbm_terminal(S0, sigma, r, T, Z)
    f = jnp.where(_use_put(S0, r, T, K), PAYOFFS["put"](ST, K), PAYOFFS["call"](ST, K))
    w = (Z**2 - 1 - sigma * jnp.sqrt(T) * Z) / (S0**2 * sigma**2 * T)
    return jnp.exp(-r * T) * f * w


@jax.jit
def pwlr_gamma_parity_samples(S0, sigma, r, T, K, Z):
    """Pathwise-LR gamma with f'(S) = -1{S < K} (put) when K < S0 e^{rT}, else 1{S > K}.

    Same formula as pwlr_gamma_samples: e^{-rT}/S0^2 E[f'(S_T) S_T (Z/(sigma sqrt T) - 1)].
    """
    ST = gbm_terminal(S0, sigma, r, T, Z)
    fprime = jnp.where(_use_put(S0, r, T, K), -1.0 * (ST < K), 1.0 * (ST > K))
    w = Z / (sigma * jnp.sqrt(T)) - 1.0
    return jnp.exp(-r * T) / S0**2 * fprime * ST * w


@jax.jit
def lr_digital_delta_parity_samples(S0, sigma, r, T, K, Z):
    """LR delta of the digital call; uses -LR delta of 1{S_T < K} when K < S0 e^{rT}."""
    ST = gbm_terminal(S0, sigma, r, T, Z)
    f = jnp.where(_use_put(S0, r, T, K), -1.0 * (ST < K), 1.0 * (ST > K))
    return jnp.exp(-r * T) * f * Z / (S0 * sigma * jnp.sqrt(T))


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
