"""Greeks by reverse-mode automatic differentiation (pathwise estimators).

jax.grad of the Monte Carlo price, with the random numbers Z held fixed,
is exactly the pathwise estimator:
    delta = e^{-rT} E[ f'(S_T) dS_T/dS0 ],  and likewise for vega and rho.
It is valid here because call and put payoffs are Lipschitz (a kink, no jump).
"""
from functools import partial

import jax
import jax.numpy as jnp

from .pricer import discounted_payoffs, mc_price, mc_price_strikes

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
def ad_greeks_fwd(S0, sigma, r, T, K, Z, kind="call"):
    """Price plus delta, vega, rho by FORWARD (tangent) mode: one jvp per input.

    jax.jvp(f, (theta,), (e_i,)) returns f(theta) and the directional derivative
    grad f . e_i = df/dtheta_i in one forward sweep carrying a tangent. Three
    inputs need three tangents; they are vmapped, so the primal (the pricing) is
    computed once and the three tangents ride along.

    Cost model (Griewank & Walther 2008, ch. 3-4; Capriotti 2011, sec. 3): forward
    mode costs ~1 + c n pricings for n inputs, c ~ 1-2 per tangent; reverse mode
    costs ~3-4 pricings for ONE output, whatever n is. So for a scalar price,
    reverse mode is expected to win for n >= 2 inputs (close at n = 2, clear from
    n = 3; measured: E6 at n = 5 is 5.3x fwd vs 2.9x rev); forward mode wins
    when outputs outnumber inputs (see experiments/e6b_forward_regime.py). Here,
    n = 3: forward 6.6x vs reverse 4.3x one pricing in flops (E1), as expected.
    Identical arithmetic to reverse mode, so the numbers agree to rounding.
    """
    def price_of(theta):
        return mc_price(theta[0], theta[1], theta[2], T, K, Z, kind=kind)

    theta = jnp.array([S0, sigma, r], dtype=jnp.float64)
    price, (delta, vega, rho) = jax.vmap(
        lambda v: jax.jvp(price_of, (theta,), (v,)), out_axes=(None, 0)
    )(jnp.eye(3))
    return {"price": price, "delta": delta, "vega": vega, "rho": rho}


# ---- many outputs, one input: K strikes, delta of each ------------------------------
# The Jacobian dV_k/dS0 of K call prices w.r.t. one spot. Forward mode: ONE jvp
# with tangent dS0 = 1 gives all K deltas (a Jacobian column). Reverse mode: one
# vjp per output with cotangent e_k (a Jacobian row each), which is how jax.jacrev
# builds it. The mirror image of the basket: here forward is flat in K and
# reverse linear (Giles & Glasserman 2006; Griewank & Walther 2008, ch. 3-4).

def _f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


@jax.jit
def strike_deltas_fwd(S0, sigma, r, T, strikes, Z):
    """(prices, deltas), both shape (K,), from one jvp."""
    return jax.jvp(lambda s: mc_price_strikes(s, sigma, r, T, strikes, Z),
                   (_f64(S0),), (_f64(1.0),))


@partial(jax.jit, static_argnames="chunk")
def strike_deltas_rev(S0, sigma, r, T, strikes, Z, chunk=None):
    """(prices, deltas) from one forward sweep + K pullbacks (= jax.jacrev).

    The forward sweep stores its residuals once; each pullback maps the cotangent
    e_k to dV_k/dS0. chunk=None vmaps all K pullbacks (exactly jax.jacrev); an
    integer runs them in chunks with lax.map to bound memory.
    """
    prices, pullback = jax.vjp(lambda s: mc_price_strikes(s, sigma, r, T, strikes, Z),
                               _f64(S0))
    basis = jnp.eye(strikes.shape[0])
    if chunk is None:
        deltas = jax.vmap(lambda e: pullback(e)[0])(basis)
    else:
        deltas = jax.lax.map(lambda e: pullback(e)[0], basis, batch_size=chunk)
    return prices, deltas


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
