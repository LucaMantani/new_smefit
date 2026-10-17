"""
smefit.chi2.py

Chi2 loss function for the smefit framework.
"""

import jax
import jax.numpy as jnp

from smefit.whitening import WhitenTransform


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
    name : str, optional
        A name for the chi2, used in logging and diagnostics.
    baseline : array-like, optional
        Baseline ("default") values of the free parameters, ordered like
        ``param_names``. Used e.g. as the gradient-descent starting/SM point.
        Defaults to a zero vector when not provided.
    """

    def __init__(
        self, fn, param_names, num_data, has_external=False, name=None, baseline=None
    ):
        self._fn = fn
        self.param_names = param_names
        self.nparam = len(param_names)
        self.num_data = num_data
        self.has_external = has_external
        self.name = name
        self.baseline = (
            jnp.zeros(self.nparam) if baseline is None else jnp.asarray(baseline)
        )

    @jax.jit(static_argnames=("self",))
    def __call__(self, coeffs):
        return self._fn(coeffs)

    def whitened(self, transform: WhitenTransform) -> "Chi2":
        """The same chi2 expressed in whitened coordinates c_w.

        Mirrors ``CoefficientGroup.whitened``: everything a sampler needs — the
        callable and the baseline it starts from — comes back in whitened
        space, so callers never have to mix the two coordinate systems.
        """
        return Chi2(
            lambda c_w: self._fn(transform.to_physical(c_w)),
            self.param_names,
            self.num_data,
            has_external=self.has_external,
            name=self.name,
            baseline=transform.to_whitened(self.baseline),
        )


def build_chi2(eft_model, data, fit_covmat):
    inv_covmat = jnp.linalg.inv(fit_covmat)

    def _chi2(coeffs):
        """Compute chi2 given coefficient values."""
        predictions = eft_model.forward_map(coeffs)
        residuals = data.cv - predictions
        return residuals.T @ inv_covmat @ residuals

    return _chi2


def build_datasets_chi2(eft_model, data, fit_covmat):
    """Build a list of per-dataset chi2 functions.

    Datasets use the diagonal block of fit_covmat (cross-dataset
    correlations are ignored).
    """
    chi2_list = []

    if data is not None:

        def _make_chi2_fn(cv, inv_c, s, e):
            def _chi2(coeffs):
                predictions = eft_model.forward_map(coeffs)[s:e]
                residuals = cv - predictions
                return residuals.T @ inv_c @ residuals

            return _chi2

        start = 0
        for name, ndata in zip(data.names, data.ndata_list):
            end = start + ndata
            cv_slice = data.cv[start:end]
            inv_covmat = jnp.linalg.inv(fit_covmat[start:end, start:end])
            chi2_list.append(
                Chi2(
                    _make_chi2_fn(cv_slice, inv_covmat, start, end),
                    eft_model.coefficients.free_names,
                    ndata,
                    name=name,
                    baseline=eft_model.coefficients.baseline_free,
                )
            )
            start = end

    return chi2_list
