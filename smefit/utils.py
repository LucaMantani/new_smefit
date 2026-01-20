"""
smefit.utils.py

Utility functions for the smefit framework.
"""

import jax
import jax.numpy as jnp


def ensure_list(x):
    """Ensure the input is a list.
    If the input is not a list, wrap it in a list.
    """
    if isinstance(x, list):
        return x
    return [x]


def run_test(eft_model):

    print(eft_model.coefficients.free_names)
    free = jnp.array([5.0, -3.0])

    print([c.name for c in eft_model.operators_to_keep])

    print(eft_model.derive_coeffs(free))

    print(eft_model.forward_map(free))
