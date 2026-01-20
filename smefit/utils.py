"""
smefit.utils.py

Utility functions for the smefit framework.
"""

import jax
import jax.numpy as jnp


def build_chi2(eft_model, data, fit_covmat):
    inv_covmat = jnp.linalg.inv(fit_covmat)

    def chi2_fn(coeffs):
        """Compute chi2 given coefficient values."""
        predictions = eft_model.forward_map(coeffs)
        residuals = data.cv - predictions
        chi2 = residuals.T @ inv_covmat @ residuals
        return chi2

    return chi2_fn


def ensure_list(x):
    """Ensure the input is a list.
    If the input is not a list, wrap it in a list.
    """
    if isinstance(x, list):
        return x
    return [x]


def run_test(eft_model, build_chi2):

    print(eft_model.coefficients.free_names)
    free = jnp.array([5.0, -3.0])

    print([c.name for c in eft_model.operators_to_keep])

    print(eft_model.derive_coeffs(free))

    print(eft_model.forward_map(free))

    print(build_chi2(free))
    # compute gradient
    grad_chi2 = jax.grad(build_chi2)(free)
    print(grad_chi2)
