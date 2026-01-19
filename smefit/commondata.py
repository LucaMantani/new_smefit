"""
smefit.commondata.py

Common data module of smefit
"""

import jax.numpy as jnp


def data_cv(data):
    """Return array of concatenated central values from data."""
    return jnp.concatenate([d.central_values for d in data], axis=0)
