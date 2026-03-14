"""
smefit.chi2.py

Chi2 loss function for the smefit framework.
"""

import jax.numpy as jnp


def build_chi2(eft_model, data, fit_covmat):
    inv_covmat = jnp.linalg.inv(fit_covmat)

    def chi2(coeffs):
        """Compute chi2 given coefficient values."""
        predictions = eft_model.forward_map(coeffs)
        residuals = data.cv - predictions
        return residuals.T @ inv_covmat @ residuals

    return chi2
