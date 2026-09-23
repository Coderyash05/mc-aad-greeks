"""Basket call on d correlated GBM assets: where reverse-mode AD pays off.

Payoff: max(sum_i w_i S_T^i - K, 0). No closed form, so Monte Carlo is needed.
The geometric basket max(prod_i (S_T^i)^{w_i} - K, 0) does have one, and is used to
check the Monte Carlo Greeks exactly (geometric_basket_price).
Correlation is parametrised by the d(d-1)/2 off-diagonal entries `rho`, so the
gradient w.r.t. rho[k] is the sensitivity to moving ONE correlation pair
(both symmetric entries together), which is what a risk system reports.

One jax.grad call returns d deltas + d vegas + d(d-1)/2 correlation
sensitivities. Bumping needs 2 repricings for each of them.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def corr_from_offdiag(rho, d):
    """Build a symmetric correlation matrix from its upper-triangle entries."""
    iu = jnp.triu_indices(d, k=1)
    C = jnp.zeros((d, d)).at[iu].set(rho)
    return C + C.T + jnp.eye(d)


def offdiag_from_corr(C):
    d = C.shape[0]
    return C[jnp.triu_indices(d, k=1)]


def basket_cholesky(rho, d):
    """Lower Cholesky factor L of the correlation matrix, C = L L^T. Depends only on
    the parameters, not on the paths: a chunked adjoint computes it once."""
    return jnp.linalg.cholesky(corr_from_offdiag(rho, d))


def basket_terminal_L(S0, sigma, L, r, T, Z):
    """S_T from the Cholesky factor. S0, sigma: (d,); L: (d, d); Z: (N, d) normals."""
    X = Z @ L.T                                  # correlated normals, (N, d)
    return S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * X)


def basket_terminal(S0, sigma, rho, r, T, Z):
    """S0, sigma: (d,); rho: (d(d-1)/2,); Z: (N, d) independent normals."""
    return basket_terminal_L(S0, sigma, basket_cholesky(rho, S0.shape[0]), r, T, Z)


def basket_price_L(S0, sigma, L, r, T, K, w, Z, geometric=False):
    """Arithmetic basket max(sum_i w_i S_T^i - K, 0), or, with geometric=True, the
    geometric basket max(prod_i (S_T^i)^{w_i} - K, 0) (weights summing to 1), which
    has a closed form (geometric_basket_price) and so serves as an exact test case."""
    if geometric:
        X = Z @ L.T
        logG = (jnp.log(S0) + (r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * X) @ w
        B = jnp.exp(logG)
    else:
        B = basket_terminal_L(S0, sigma, L, r, T, Z) @ w
    return jnp.exp(-r * T) * jnp.mean(jnp.maximum(B - K, 0.0))


@partial(jax.jit, static_argnames="geometric")
def basket_price(S0, sigma, rho, r, T, K, w, Z, geometric=False):
    return basket_price_L(S0, sigma, basket_cholesky(rho, S0.shape[0]), r, T, K, w, Z,
                          geometric)


@partial(jax.jit, static_argnames="geometric")
def basket_greeks(S0, sigma, rho, r, T, K, w, Z, geometric=False):
    """Price + all deltas, vegas, correlation sensitivities in one reverse pass."""
    price, (delta, vega, corr_sens) = jax.value_and_grad(basket_price, argnums=(0, 1, 2))(
        S0, sigma, rho, r, T, K, w, Z, geometric
    )
    return {"price": price, "delta": delta, "vega": vega, "corr": corr_sens}


@partial(jax.jit, static_argnames=("n_batches", "geometric"))
def basket_greeks_se(S0, sigma, rho, r, T, K, w, Z, n_batches=200, geometric=False):
    """Price and Greeks with batch-means standard errors.

    Per-path gradients would need an N x P array (P = 2d + d(d-1)/2: 1,325 at d = 50),
    so Z (N, d) is split into n_batches equal batches and basket_greeks is run on each
    (lax.map: one batch live at a time). The estimate is the mean of the batch
    estimates, which equals the full-sample estimate, and SE = sd(batch estimates) /
    sqrt(n_batches) (batch means; Glasserman 2004, ch. 1 and App. A). With 200
    batches, z = error / SE is t-distributed with 199 degrees of freedom under the
    null, close to N(0, 1).
    """
    g = basket_greeks_batches(S0, sigma, rho, r, T, K, w, Z, n_batches, geometric)
    return {k: (jnp.mean(v, axis=0), jnp.std(v, axis=0, ddof=1) / jnp.sqrt(n_batches))
            for k, v in g.items()}


@partial(jax.jit, static_argnames=("n_batches", "geometric"))
def basket_greeks_batches(S0, sigma, rho, r, T, K, w, Z, n_batches=200, geometric=False):
    """Per-batch price and Greeks, arrays with a leading batch axis (n_batches, ...).
    Needed for statements about the whole family of sensitivities (their correlations;
    mcgreeks.stats.family_zscores)."""
    Zb = Z.reshape(n_batches, -1, Z.shape[-1])
    return jax.lax.map(lambda z: basket_greeks(S0, sigma, rho, r, T, K, w, z, geometric), Zb)


def example_basket(d, seed):
    """A reproducible, deliberately heterogeneous test basket (used by the tests and E9).

    S0 spread over 80-120, sigma over 0.15-0.35, positive weights summing to 1, and a
    correlation matrix from a 2-factor model, C = D (B B^T + I) D with B ~ N(0, 1)
    (d x 2) and D rescaling the diagonal to 1: positive definite by construction,
    with correlations of both signs. Every sensitivity is then different, so a test
    cannot pass by symmetry. Returns (S0, sigma, rho, w) as float64 arrays.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(d, 2))
    C = B @ B.T + np.eye(d)
    D = 1.0 / np.sqrt(np.diag(C))
    C = D[:, None] * C * D[None, :]
    w = rng.uniform(0.5, 1.5, d)
    return (jnp.linspace(80.0, 120.0, d) if d > 1 else jnp.array([100.0]),
            jnp.linspace(0.15, 0.35, d) if d > 1 else jnp.array([0.2]),
            offdiag_from_corr(jnp.asarray(C)), jnp.asarray(w / w.sum()))


def geometric_basket_price(S0, sigma, rho, r, T, K, w):
    """Closed-form geometric basket call, weights w summing to 1.

    log G_T = sum_i w_i log S_T^i
            = sum_i w_i [log S0_i + (r - sigma_i^2/2) T] + sqrt(T) sum_i w_i sigma_i X_i,
    with X ~ N(0, C), C the correlation matrix. So log G_T is normal with
      mean  mu  = sum_i w_i [log S0_i + (r - sigma_i^2/2) T]
      var   s^2 = T (w * sigma)^T C (w * sigma)
    and, as for any lognormal G (same formula as geometric_asian_price),
      price = e^{-rT} [ e^{mu + s^2/2} N(d1) - K N(d2) ],  d1 = (mu - log K + s^2)/s, d2 = d1 - s.
    Glasserman (2004), ch. 4 (geometric baskets as control variates). d = 1, or C = all
    ones with equal sigma and S0, reduces to Black-Scholes. Differentiable in JAX, so
    exact deltas, vegas and correlation sensitivities come from jax.grad.
    """
    C = corr_from_offdiag(rho, S0.shape[0])
    a = w * sigma
    mu = jnp.sum(w * (jnp.log(S0) + (r - 0.5 * sigma**2) * T))
    s2 = T * a @ C @ a
    s = jnp.sqrt(s2)
    d1 = (mu - jnp.log(K) + s2) / s
    d2 = d1 - s
    return jnp.exp(-r * T) * (jnp.exp(mu + 0.5 * s2) * norm.cdf(d1) - K * norm.cdf(d2))


@jax.jit
def geometric_basket_greeks(S0, sigma, rho, r, T, K, w):
    """Exact price, deltas, vegas and correlation sensitivities (grad of the closed form)."""
    price, (delta, vega, corr_sens) = jax.value_and_grad(
        geometric_basket_price, argnums=(0, 1, 2))(S0, sigma, rho, r, T, K, w)
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
