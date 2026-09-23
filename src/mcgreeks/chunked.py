"""Chunked (batched) adjoint: the gradient of a Monte Carlo mean, B paths at a time.

Differentiating mean(payoff over all N paths) in one reverse pass stores every
forward intermediate for all N paths, so memory grows like N. But the estimator
is a mean, and the gradient of a mean is the mean of gradients:

    V(theta) = (1/N) sum_n f(theta; Z_n) = (1/n_b) sum_b V_b(theta),
    V_b(theta) = (1/B) sum_{n in batch b} f(theta; Z_n),     N = n_b B,
    dV/dtheta = (1/n_b) sum_b dV_b/dtheta.

So each batch can be differentiated on its own and the batch gradients summed:
only one batch's intermediates are alive at a time, and memory grows like B, not
N. This is the path-by-path adjoint of Giles & Glasserman (2006), with B paths
per step instead of one. Mathematically the result is identical to the one-shot
gradient; numerically it differs only by summation order (~1e-15 relative).

Parameter-only work (`prepare`). Some of the pricing depends on theta alone, e.g.
the basket's Cholesky factor L(rho). Redoing it, and its adjoint, in every batch
costs n_b times. Instead write V_b(theta) = g_b(a(theta)) with a = prepare(theta)
computed once; by the chain rule
    dV/dtheta = (da/dtheta)^T [ (1/n_b) sum_b dg_b/da ],
so the batches accumulate the gradient w.r.t. a, and ONE vector-Jacobian product
of `prepare` (jax.vjp) maps it back to theta. The vjp is linear in its cotangent,
so pulling back the averaged gradient once equals averaging the pulled-back ones.

The batches are accumulated with lax.scan, so the whole thing compiles to one XLA
program. Two ways to feed a batch its random numbers:
  - chunked_value_and_grad: slice batch b out of a precomputed Z (dynamic_slice
    along `batch_axis`; no reshape or transpose of the full Z). Z is an N-sized input.
  - chunked_value_and_grad_keyed: batch b draws its own normals from the key
    jax.random.fold_in(key, b). Nothing N-sized exists, so TOTAL memory is O(B).
    fold_in gives each batch an independent, reproducible stream (the same b gives
    the same numbers every call), so the one-shot gradient on the same numbers can
    be rebuilt for validation.
"""
import jax
import jax.numpy as jnp


def _identity(p):
    return p


def _accumulate(price_fn, prepare, params, n_batches, batch_input):
    aux, pullback = jax.vjp(prepare or _identity, params)
    vg = jax.value_and_grad(price_fn)

    def body(carry, i):
        v, g = vg(aux, batch_input(i))
        return (carry[0] + v, jax.tree_util.tree_map(jnp.add, carry[1], g)), None

    init = (jnp.zeros(()), jax.tree_util.tree_map(jnp.zeros_like, aux))
    (v, g_aux), _ = jax.lax.scan(body, init, jnp.arange(n_batches))
    g_aux = jax.tree_util.tree_map(lambda x: x / n_batches, g_aux)
    (g,) = pullback(g_aux)
    return v / n_batches, g


def chunked_value_and_grad(price_fn, batch_size, batch_axis=0, prepare=None):
    """Return jit-compiled f(params, Z) -> (value, grad), batch by batch.

    price_fn(aux, Z_batch) must return the scalar Monte Carlo mean over the paths
    in Z_batch, where aux = prepare(params) (aux = params if prepare is None);
    params is any pytree. Z's size along `batch_axis` must be a multiple of batch_size.
    """
    @jax.jit
    def f(params, Z):
        n = Z.shape[batch_axis]
        if n % batch_size:
            raise ValueError(f"{n} paths is not a multiple of batch_size={batch_size}")

        def batch(i):
            return jax.lax.dynamic_slice_in_dim(Z, i * batch_size, batch_size, axis=batch_axis)

        return _accumulate(price_fn, prepare, params, n // batch_size, batch)

    return f


def chunked_value_and_grad_keyed(price_fn, n_batches, prepare=None):
    """Return jit-compiled f(params, key) -> (value, grad), batch by batch.

    price_fn(aux, key_b) must generate batch b's random numbers from key_b =
    jax.random.fold_in(key, b) and return the scalar mean over its paths.
    """
    @jax.jit
    def f(params, key):
        return _accumulate(price_fn, prepare, params, n_batches,
                           lambda i: jax.random.fold_in(key, i))

    return f


def chunked_mean_keyed(price_fn, n_batches):
    """The price alone, with the same per-batch random numbers (no gradient): the
    one-pricing reference for the keyed chunked adjoint. f(aux, key) -> value."""
    @jax.jit
    def f(aux, key):
        def body(acc, i):
            return acc + price_fn(aux, jax.random.fold_in(key, i)), None

        total, _ = jax.lax.scan(body, jnp.zeros(()), jnp.arange(n_batches))
        return total / n_batches

    return f


def keyed_normals(key, n_batches, batch_shape):
    """The normals the keyed chunked adjoint uses, concatenated along axis 0:
    batch b is jax.random.normal(fold_in(key, b), batch_shape). For validation
    against a one-shot gradient on the same numbers."""
    Zs = jax.vmap(lambda b: jax.random.normal(jax.random.fold_in(key, b), batch_shape,
                                              dtype=jnp.float64))(jnp.arange(n_batches))
    return Zs.reshape((n_batches * batch_shape[0],) + tuple(batch_shape[1:]))
