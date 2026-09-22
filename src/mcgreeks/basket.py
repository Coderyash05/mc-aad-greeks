"""Basket call on d correlated GBM assets: where reverse-mode AD pays off.

Payoff: max(sum_i w_i S_T^i - K, 0). No closed form, so Monte Carlo is needed.
Correlation is parametrised by the d(d-1)/2 off-diagonal entries `rho`, so the
gradient w.r.t. rho[k] is the sensitivity to moving ONE correlation pair
(both symmetric entries together), which is what a risk system reports.

One jax.grad call returns d deltas + d vegas + d(d-1)/2 correlation
sensitivities. Bumping needs 2 repricings for each of them.
"""
from functools import partial

import jax
import jax.numpy as jnp


def corr_from_offdiag(rho, d):
    """Build a symmetric correlation matrix from its upper-triangle entries."""
    iu = jnp.triu_indices(d, k=1)
    C = jnp.zeros((d, d)).at[iu].set(rho)
    return C + C.T + jnp.eye(d)


def offdiag_from_corr(C):
    d = C.shape[0]
    return C[jnp.triu_indices(d, k=1)]


def basket_terminal(S0, sigma, rho, r, T, Z):
    """S0, sigma: (d,); rho: (d(d-1)/2,); Z: (N, d) independent normals."""
    d = S0.shape[0]
    L = jnp.linalg.cholesky(corr_from_offdiag(rho, d))
    X = Z @ L.T                                  # correlated normals, (N, d)
    return S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * X)


@jax.jit
def basket_price(S0, sigma, rho, r, T, K, w, Z):
    ST = basket_terminal(S0, sigma, rho, r, T, Z)
    return jnp.exp(-r * T) * jnp.mean(jnp.maximum(ST @ w - K, 0.0))


@jax.jit
def basket_greeks(S0, sigma, rho, r, T, K, w, Z):
    """Price + all deltas, vegas, correlation sensitivities in one reverse pass."""
    price, (delta, vega, corr_sens) = jax.value_and_grad(basket_price, argnums=(0, 1, 2))(
        S0, sigma, rho, r, T, K, w, Z
    )
    return {"price": price, "delta": delta, "vega": vega, "corr": corr_sens}


@jax.jit
def basket_dK(S0, sigma, rho, r, T, K, w, Z):
    """dV/dK, used for the Euler homogeneity check."""
    return jax.grad(basket_price, argnums=5)(S0, sigma, rho, r, T, K, w, Z)
