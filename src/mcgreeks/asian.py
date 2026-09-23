"""Discretely monitored Asian call under GBM, simulated step by step with lax.scan.

Monitoring dates t_i = i T / M, i = 1..M. Exact GBM steps in log space:
    log S_{i} = log S_{i-1} + (r - sigma^2/2) dt + sigma sqrt(dt) Z_i,   dt = T / M.
Arithmetic average A = (1/M) sum_i S_{t_i}: no closed form.
Geometric average G = exp((1/M) sum_i log S_{t_i}): lognormal, closed form below,
used to validate the simulation and its autodiff Greeks.

Z is TIME-MAJOR, shape (M, N): scan reads row i at step i with no transpose of the
full array (a transpose would add an N x M temporary to every memory measurement).

Memory: reverse mode through lax.scan saves each step's residuals for the backward
sweep, so a one-shot gradient stores O(N M) numbers. With checkpoint=True the scan
body is wrapped in jax.checkpoint: JAX then saves only each step's input carry
(log S and the running sum, 2 arrays of size N) and recomputes the body in the
backward sweep. Memory is still O(N M), with a smaller constant, and the body is
evaluated twice. (O(sqrt(M)) memory needs nested/binomial checkpointing,
Griewank & Walther 2008, ch. 12, which plain jax.checkpoint on the body does not do.)
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def _step_fn(sigma, r, T, M, geometric):
    dt = T / M
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * jnp.sqrt(dt)

    def step(carry, z):
        logS, acc = carry
        logS = logS + drift + vol * z
        return (logS, acc + (logS if geometric else jnp.exp(logS))), None

    return step


def _average(S0, sigma, r, T, Z, geometric, checkpoint, unroll):
    M = Z.shape[0]
    step = _step_fn(sigma, r, T, M, geometric)
    if checkpoint:
        step = jax.checkpoint(step)
    logS0 = jnp.zeros(Z.shape[1:]) + jnp.log(S0)
    (_, acc), _ = jax.lax.scan(step, (logS0, jnp.zeros(Z.shape[1:])), Z, unroll=unroll)
    return jnp.exp(acc / M) if geometric else acc / M


def asian_payoffs(S0, sigma, r, T, K, Z, geometric=False, checkpoint=False, unroll=1):
    """Discounted payoff per path, shape Z.shape[1:]."""
    A = _average(S0, sigma, r, T, Z, geometric, checkpoint, unroll)
    return jnp.exp(-r * T) * jnp.maximum(A - K, 0.0)


@partial(jax.jit, static_argnames=("geometric", "checkpoint", "unroll"))
def asian_price(S0, sigma, r, T, K, Z, geometric=False, checkpoint=False, unroll=1):
    """Monte Carlo price; Z has shape (M, N). unroll=True is only for cost_analysis,
    which counts a rolled scan body once."""
    return jnp.mean(asian_payoffs(S0, sigma, r, T, K, Z, geometric, checkpoint, unroll))


# ---- normals generated inside the simulation, one time block at a time ------------
# Steps are grouped into M / m blocks of m steps; block j draws its normals from
# jax.random.normal(fold_in(key, j), (m, n_paths)). Nothing of size M x n_paths is
# an input, so memory is set only by what reverse mode stores.
#
# Two-level checkpointing (Griewank & Walther 2008, ch. 12; the sqrt(M) scheme):
# with checkpoint=True each BLOCK (inner scan of m steps, including drawing its
# normals) is wrapped in jax.checkpoint. The outer scan then saves only each
# block's input carry, M/m of them; the backward sweep recomputes one block at a
# time (regenerating its normals from its key) and holds just that block's m steps
# of residuals. Memory ~ (M/m + m) x n_paths, minimised at m ~ sqrt(M): O(sqrt(M))
# instead of O(M), for one extra forward sweep (the recomputation).
# Without checkpointing every step stores its normal z (needed for d/dsigma of
# sigma sqrt(dt) z) and exp(log S) (for d exp): 2 arrays per step, O(M).

def sqrt_block(M):
    """Block length m: the divisor of M closest to sqrt(M) (ties: the smaller)."""
    return min((m for m in range(1, M + 1) if M % m == 0), key=lambda m: (abs(m - M**0.5), m))


def blocked_normals(key, n_paths, M, block):
    """The (M, n_paths) normals asian_price_blocked draws, for validation."""
    Zs = jax.vmap(lambda j: jax.random.normal(jax.random.fold_in(key, j), (block, n_paths),
                                              dtype=jnp.float64))(jnp.arange(M // block))
    return Zs.reshape(M, n_paths)


def asian_price_blocked(S0, sigma, r, T, K, key, n_paths, M, block, checkpoint=False, unroll=1):
    """Arithmetic Asian price with in-simulation normals (see above). Same numbers
    as asian_price on blocked_normals(key, n_paths, M, block). Not jitted itself:
    n_paths, M, block, checkpoint and unroll must be static."""
    if M % block:
        raise ValueError(f"block={block} does not divide M={M}")
    step = _step_fn(sigma, r, T, M, False)

    def run_block(carry, j):
        z = jax.random.normal(jax.random.fold_in(key, j), (block, n_paths), dtype=jnp.float64)
        carry, _ = jax.lax.scan(step, carry, z, unroll=unroll)
        return carry, None

    if checkpoint:
        run_block = jax.checkpoint(run_block)
    init = (jnp.zeros(n_paths) + jnp.log(S0), jnp.zeros(n_paths))
    (_, acc), _ = jax.lax.scan(run_block, init, jnp.arange(M // block), unroll=unroll)
    return jnp.exp(-r * T) * jnp.mean(jnp.maximum(acc / M - K, 0.0))


@partial(jax.jit, static_argnames=("geometric",))
def asian_greeks_se(S0, sigma, r, T, K, Z, geometric=False):
    """Delta and vega with standard errors, from per-path pathwise gradients (vmap
    over the path axis of Z), as in greeks_ad.ad_greeks_se."""
    def one_path(s0, sig, z):
        return asian_payoffs(s0, sig, r, T, K, z[:, None], geometric)[0]

    g = jax.vmap(jax.grad(one_path, argnums=(0, 1)), in_axes=(None, None, 1))(S0, sigma, Z)
    n = Z.shape[1]
    return {name: (jnp.mean(x), jnp.std(x, ddof=1) / jnp.sqrt(n))
            for name, x in zip(("delta", "vega"), g)}


def geometric_asian_price(S0, K, r, sigma, T, M):
    """Closed-form discretely monitored geometric-average Asian call.

    log G = log S0 + (r - sigma^2/2) (1/M) sum_i t_i + sigma (1/M) sum_i W_{t_i} is
    normal, with t_i = i dt, dt = T/M:
      mean  mu    = log S0 + (r - sigma^2/2) T (M+1) / (2M)       [(1/M) sum_i i dt]
      var   s^2   = sigma^2 T (M+1)(2M+1) / (6 M^2)               [(dt/M^2) sum_ij min(i,j)]
    using sum_{i,j<=M} min(i,j) = M(M+1)(2M+1)/6. Then, as for any lognormal G,
      price = e^{-rT} [ e^{mu + s^2/2} N(d1) - K N(d2) ],
      d1 = (mu - log K + s^2) / s,  d2 = d1 - s.
    M = 1 reduces to Black-Scholes. Kemna & Vorst (1990); Glasserman (2004), ch. 4.
    Differentiable in JAX, so exact Greeks come from jax.grad.
    """
    mu = jnp.log(S0) + (r - 0.5 * sigma**2) * T * (M + 1) / (2 * M)
    s2 = sigma**2 * T * (M + 1) * (2 * M + 1) / (6 * M**2)
    s = jnp.sqrt(s2)
    d1 = (mu - jnp.log(K) + s2) / s
    d2 = d1 - s
    return jnp.exp(-r * T) * (jnp.exp(mu + 0.5 * s2) * norm.cdf(d1) - K * norm.cdf(d2))
