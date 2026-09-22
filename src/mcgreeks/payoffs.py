"""Option payoffs. Smoothed and digital versions are added later (days 6-9)."""
import jax.numpy as jnp


def call_payoff(ST, K):
    return jnp.maximum(ST - K, 0.0)


def put_payoff(ST, K):
    return jnp.maximum(K - ST, 0.0)


def digital_payoff(ST, K):
    """Cash-or-nothing digital call: pays 1 if S_T > K. A jump, not a kink."""
    return (ST > K).astype(ST.dtype)


PAYOFFS = {"call": call_payoff, "put": put_payoff, "digital": digital_payoff}


# ---- smoothed payoffs (make pathwise / autodiff second derivatives non-zero) ----
def call_payoff_smooth(ST, K, eps):
    """Softplus call: eps * log(1 + e^{(S-K)/eps}) -> max(S-K, 0) as eps -> 0.

    Its second derivative is a bump of width ~eps around the strike instead of
    a Dirac delta, so autodiff can see the curvature. The price of this is a
    bias that shrinks as eps -> 0 while the variance grows.
    """
    import jax  # local import keeps the module's top clean

    return eps * jax.nn.softplus((ST - K) / eps)


def digital_payoff_smooth(ST, K, eps):
    """Sigmoid digital: 1 / (1 + e^{-(S-K)/eps}) -> 1{S > K} as eps -> 0."""
    import jax

    return jax.nn.sigmoid((ST - K) / eps)
