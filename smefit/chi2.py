"""
smefit.chi2.py

Chi2 loss function for the smefit framework.
"""

import jax
import jax.numpy as jnp


class Chi2:
    """Callable chi2 with metadata about its composition.

    Parameters
    ----------
    fn : callable
        The chi2 function ``fn(coeffs) -> scalar``.
    param_names : list of str
        The names of the parameters.
    has_external : bool
        Whether the chi2 includes external contributions.
    """

    def __init__(self, fn, param_names, num_data, has_external=False):
        self._fn = fn
        self.param_names = param_names
        self.nparam = len(param_names)
        self.num_data = num_data
        self.has_external = has_external

    @jax.jit(static_argnames=("self",))
    def __call__(self, coeffs):
        return self._fn(coeffs)


def build_chi2(eft_model, data, fit_covmat):
    inv_covmat = jnp.linalg.inv(fit_covmat)

    def _chi2(coeffs):
        """Compute chi2 given coefficient values."""
        predictions = eft_model.forward_map(coeffs)
        residuals = data.cv - predictions
        return residuals.T @ inv_covmat @ residuals

    return _chi2
