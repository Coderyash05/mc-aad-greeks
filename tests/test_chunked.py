"""Chunked adjoint: batch-by-batch gradients summed with lax.scan == one-shot gradient."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mcgreeks  # noqa: F401
from mcgreeks.asian import asian_price, asian_price_blocked, blocked_normals, sqrt_block
from mcgreeks.basket import basket_cholesky, basket_greeks, basket_price, basket_price_L
from mcgreeks.chunked import (chunked_mean_keyed, chunked_value_and_grad,
                              chunked_value_and_grad_keyed, keyed_normals)
from mcgreeks.greeks_ad import ad_greeks
from mcgreeks.models import normals
from mcgreeks.pricer import mc_price

R, T, K = 0.05, 1.0, 100.0


def _check(v, g, v_ref, g_ref):
    """Same numbers up to summation order: 1e-10 relative."""
    np.testing.assert_allclose(v, v_ref, rtol=1e-10)
    for a, b in zip(g, g_ref):
        np.testing.assert_allclose(a, b, rtol=1e-10)


@pytest.mark.parametrize("B", [1_000, 10_000, 50_000])
def test_european_call(B):
    Z = normals(jax.random.PRNGKey(50), 50_000)
    ref = ad_greeks(100.0, 0.2, R, T, K, Z)
    f = chunked_value_and_grad(lambda p, z: mc_price(p[0], p[1], p[2], T, K, z), B)
    v, g = f((jnp.float64(100.0), jnp.float64(0.2), jnp.float64(R)), Z)
    _check(v, g, ref["price"], (ref["delta"], ref["vega"], ref["rho"]))


@pytest.mark.parametrize("B", [1_000, 10_000])
def test_basket_d5(B):
    d = 5
    S0, sigma = jnp.full(d, 100.0), jnp.linspace(0.15, 0.35, d)
    rho, w = jnp.full(d * (d - 1) // 2, 0.5), jnp.full(d, 1.0 / d)
    Z = jax.random.normal(jax.random.PRNGKey(51), (50_000, d), dtype=jnp.float64)
    ref = basket_greeks(S0, sigma, rho, R, T, K, w, Z)
    f = chunked_value_and_grad(lambda p, z: basket_price(*p, R, T, K, w, z), B)
    v, g = f((S0, sigma, rho), Z)
    _check(v, g, ref["price"], (ref["delta"], ref["vega"], ref["corr"]))


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("B", [1_000, 10_000])
def test_asian(B, checkpoint):
    """Time-major Z (M, N): batches are taken along axis 1."""
    Z = jax.random.normal(jax.random.PRNGKey(52), (12, 50_000), dtype=jnp.float64)

    def price(p, z, ckpt=False):
        return asian_price(p[0], p[1], R, T, K, z, checkpoint=ckpt)

    params = (jnp.float64(100.0), jnp.float64(0.2))
    v_ref, g_ref = jax.value_and_grad(price)(params, Z)
    f = chunked_value_and_grad(lambda p, z: price(p, z, checkpoint), B, batch_axis=1)
    v, g = f(params, Z)
    _check(v, g, v_ref, g_ref)


def test_checkpoint_changes_nothing_numerically():
    """jax.checkpoint only changes what is stored vs recomputed, not the arithmetic."""
    Z = jax.random.normal(jax.random.PRNGKey(53), (52, 20_000), dtype=jnp.float64)
    g = [jax.grad(lambda s, c=c: asian_price(s, 0.2, R, T, K, Z, checkpoint=c))(100.0)
         for c in (False, True)]
    np.testing.assert_allclose(g[1], g[0], rtol=1e-13)


# ---- hoisted parameter work and in-chunk random numbers ----------------------------
def _basket_setup(d=5):
    S0, sigma = jnp.full(d, 100.0), jnp.linspace(0.15, 0.35, d)
    rho, w = jnp.full(d * (d - 1) // 2, 0.5), jnp.full(d, 1.0 / d)
    return S0, sigma, rho, w


def _basket_prepare(p):
    S0, sigma, rho = p
    return S0, sigma, basket_cholesky(rho, S0.shape[0])


@pytest.mark.parametrize("B", [1_000, 10_000])
def test_basket_hoisted_cholesky(B):
    """Cholesky once outside the scan, one vjp back to rho: same gradient."""
    S0, sigma, rho, w = _basket_setup()
    Z = jax.random.normal(jax.random.PRNGKey(55), (50_000, 5), dtype=jnp.float64)
    ref = basket_greeks(S0, sigma, rho, R, T, K, w, Z)
    f = chunked_value_and_grad(lambda a, z: basket_price_L(*a, R, T, K, w, z), B,
                               prepare=_basket_prepare)
    v, g = f((S0, sigma, rho), Z)
    _check(v, g, ref["price"], (ref["delta"], ref["vega"], ref["corr"]))


@pytest.mark.parametrize("B", [1_000, 10_000])
def test_basket_keyed_matches_oneshot_on_same_stream(B):
    """Each batch draws normal(fold_in(key, b)); the one-shot gradient on exactly
    those numbers (rebuilt with keyed_normals) must agree."""
    S0, sigma, rho, w = _basket_setup()
    key, n_b = jax.random.PRNGKey(56), 50_000 // B
    f = chunked_value_and_grad_keyed(
        lambda a, k: basket_price_L(*a, R, T, K, w,
                                    jax.random.normal(k, (B, 5), dtype=jnp.float64)),
        n_b, prepare=_basket_prepare)
    v, g = f((S0, sigma, rho), key)
    Z = keyed_normals(key, n_b, (B, 5))
    ref = basket_greeks(S0, sigma, rho, R, T, K, w, Z)
    _check(v, g, ref["price"], (ref["delta"], ref["vega"], ref["corr"]))
    np.testing.assert_allclose(chunked_mean_keyed(
        lambda a, k: basket_price_L(*a, R, T, K, w,
                                    jax.random.normal(k, (B, 5), dtype=jnp.float64)),
        n_b)(_basket_prepare((S0, sigma, rho)), key), ref["price"], rtol=1e-12)


def test_sqrt_block():
    assert [sqrt_block(M) for M in (12, 52, 252, 1000)] == [3, 4, 14, 25]


@pytest.mark.parametrize("M", [12, 52])
def test_blocked_price_equals_asian_price(M):
    key, n = jax.random.PRNGKey(57), 20_000
    blk = sqrt_block(M)
    for ckpt in (False, True):
        p = jax.jit(lambda s, c=ckpt: asian_price_blocked(s, 0.2, R, T, K, key, n, M, blk, c))
        np.testing.assert_allclose(p(100.0), asian_price(100.0, 0.2, R, T, K,
                                                         blocked_normals(key, n, M, blk)),
                                   rtol=1e-13)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("M", [12, 52])
def test_asian_keyed_matches_oneshot_on_same_stream(M, checkpoint):
    """Chunk b uses key_b = fold_in(key, b); within it, time block j uses
    fold_in(key_b, j). Rebuild that (M, N) array and compare with one-shot."""
    key, B, n_b = jax.random.PRNGKey(58), 5_000, 4
    blk = sqrt_block(M)
    params = (jnp.float64(100.0), jnp.float64(0.2))
    f = chunked_value_and_grad_keyed(
        lambda p, k: asian_price_blocked(p[0], p[1], R, T, K, k, B, M, blk, checkpoint), n_b)
    v, g = f(params, key)
    Z = jnp.concatenate([blocked_normals(jax.random.fold_in(key, b), B, M, blk)
                         for b in range(n_b)], axis=1)
    v_ref, g_ref = jax.value_and_grad(
        lambda p: asian_price(p[0], p[1], R, T, K, Z))(params)
    _check(v, g, v_ref, g_ref)


def test_rejects_uneven_batches():
    f = chunked_value_and_grad(lambda p, z: mc_price(p, 0.2, R, T, K, z), 3_000)
    with pytest.raises(ValueError):
        f(jnp.float64(100.0), normals(jax.random.PRNGKey(54), 10_000))
