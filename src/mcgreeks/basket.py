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


# ---- forward mode and bump-and-reprice baselines ------------------------------------
# Every function below returns (price, grad) with grad = concat(deltas, vegas,
# corr sensitivities) = dV/dtheta, theta = concat(S0, sigma, rho), P = len(theta).
#
# `chunk` bounds memory: chunk=None vmaps all P (or 2P + 1) evaluations at once;
# an integer runs them through lax.map(..., batch_size=chunk), i.e. a loop over
# chunks with vmap inside, so live intermediates are ~chunk x one pricing's N x d
# arrays. (Note for cost_analysis: XLA counts a loop body once, whatever the trip
# count, so flops must be read from the chunk=None program.)

def _unpack(theta, d):
    return theta[:d], theta[d:2 * d], theta[2 * d:]


def _map(f, xs, chunk):
    return jax.vmap(f)(xs) if chunk is None else jax.lax.map(f, xs, batch_size=chunk)


@partial(jax.jit, static_argnames="chunk")
def basket_greeks_fwd(S0, sigma, rho, r, T, K, w, Z, chunk=None):
    """Forward (tangent) mode: one Jacobian-vector product per input direction e_i.

    jvp propagates a tangent alongside the forward computation, giving the
    directional derivative dV/dtheta . v for ~2-3x the cost of one pricing
    (Griewank & Walther 2008, ch. 3; Capriotti 2011, sec. 3). Getting the full
    gradient needs P tangents, so cost grows linearly in P, while reverse mode
    gets all P from one backward sweep. This is the forward-vs-adjoint contrast of
    Giles & Glasserman (2006), Fig. 4. Same arithmetic as reverse mode, so the
    gradients agree to rounding.
    """
    d = S0.shape[0]
    theta = jnp.concatenate([S0, sigma, rho])

    def price_of(th):
        return basket_price(*_unpack(th, d), r, T, K, w, Z)

    prices, grad = _map(lambda v: jax.jvp(price_of, (theta,), (v,)), jnp.eye(theta.shape[0]),
                        chunk)
    return prices[0], grad


# Bump-and-reprice with common random numbers, central differences:
#     dV/dtheta_i ~ [V(theta + h e_i) - V(theta - h e_i)] / (2h),
# error O(h^2) from the Taylor remainder plus the kink paths within h of the strike
# (Glasserman 2004, sec. 7.1). Cost: 2P + 1 pricings (the +1 is the base price).

def bump_grad_loop(S0, sigma, rho, r, T, K, w, Z, h=1e-4):
    """Python loop over 2P + 1 calls of the jit-compiled pricer (the naive way)."""
    price = basket_price(S0, sigma, rho, r, T, K, w, Z)
    vals = []
    for idx, x in enumerate((S0, sigma, rho)):
        for i in range(x.shape[0]):
            for s in (h, -h):
                args = [S0, sigma, rho]
                args[idx] = x.at[i].add(s)
                vals.append(basket_price(*args, r, T, K, w, Z))
    vals = jnp.stack(vals).reshape(-1, 2)          # rows: (up, down) per parameter
    return price, (vals[:, 0] - vals[:, 1]) / (2 * h)


@partial(jax.jit, static_argnames="chunk")
def bump_grad_vmap(S0, sigma, rho, r, T, K, w, Z, h=1e-4, chunk=None):
    """All 2P + 1 pricings in ONE compiled program, vmapped (in chunks, see above).

    Same bumps and same Z as bump_grad_loop, so the two gradients agree to
    rounding (~1e-10). Removes Python dispatch; does NOT reduce the arithmetic.
    """
    d = S0.shape[0]
    theta = jnp.concatenate([S0, sigma, rho])
    P = theta.shape[0]
    E = h * jnp.eye(P)
    thetas = jnp.concatenate([theta[None], theta + E, theta - E])  # (2P + 1, P)

    def price_at(th):
        return basket_price(*_unpack(th, d), r, T, K, w, Z)

    vals = _map(price_at, thetas, chunk)
    return vals[0], (vals[1:P + 1] - vals[P + 1:]) / (2 * h)
