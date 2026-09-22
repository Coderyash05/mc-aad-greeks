"""Path models. For European options under GBM one exact step to T is enough."""
import jax
import jax.numpy as jnp


def gbm_terminal(S0, sigma, r, T, Z):
    """Exact GBM terminal price, S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z).

    Z is an input on purpose: with Z fixed, the estimator is a deterministic,
    differentiable function of (S0, sigma, r, T), which is what jax.grad needs.
    """
    return S0 * jnp.exp((r - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * Z)


def normals(key, n_paths):
    """Standard normal draws in float64."""
    return jax.random.normal(key, (n_paths,), dtype=jnp.float64)
