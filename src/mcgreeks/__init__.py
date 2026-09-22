"""Monte Carlo Greeks by algorithmic differentiation.

Importing this package switches JAX to 64-bit floats. Finite-difference
comparisons later in the project are meaningless in float32, so this is
set once, here, before any array is created.
"""
import jax

jax.config.update("jax_enable_x64", True)
