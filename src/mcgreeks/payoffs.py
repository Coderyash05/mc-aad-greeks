"""Option payoffs. Smoothed and digital versions are added later (days 6-9)."""
import jax.numpy as jnp


def call_payoff(ST, K):
    return jnp.maximum(ST - K, 0.0)


def put_payoff(ST, K):
    return jnp.maximum(K - ST, 0.0)


PAYOFFS = {"call": call_payoff, "put": put_payoff}
