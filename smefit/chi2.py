"""
smefit.chi2.py

Chi2 loss function for the smefit framework.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from smefit.whitening import WhitenTransform

if TYPE_CHECKING:
    from smefit.core import DataGroup
    from smefit.model import EFTModel


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
        self,
        fn: Callable[[jnp.ndarray], jnp.ndarray],
        param_names: Sequence[str],
        num_data: int,
        has_external: bool = False,
        name: str | None = None,
        baseline: jnp.ndarray | Sequence[float] | None = None,
    ) -> None:
        self._fn = fn
        self.param_names = list(param_names)
        self.nparam = len(self.param_names)
        self.num_data = num_data
        self.has_external = has_external
        self.name = name
        self.baseline = (
            jnp.zeros(self.nparam) if baseline is None else jnp.asarray(baseline)
        )

    @jax.jit(static_argnames=("self",))
    def __call__(self, coeffs: jnp.ndarray) -> jnp.ndarray:
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


def build_chi2(
    eft_model: "EFTModel", data: "DataGroup", fit_covmat: jnp.ndarray
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    inv_covmat = jnp.linalg.inv(fit_covmat)

    def _chi2(coeffs: jnp.ndarray) -> jnp.ndarray:
        """Compute chi2 given coefficient values."""
        predictions = eft_model.forward_map(coeffs)
        residuals = data.cv - predictions
        return residuals.T @ inv_covmat @ residuals

    return _chi2


def build_datasets_chi2(
    eft_model: "EFTModel | None",
    data: "DataGroup | None",
    fit_covmat: jnp.ndarray | None,
) -> list[Chi2]:
    """Build a list of per-dataset chi2 functions.

    Datasets use the diagonal block of fit_covmat (cross-dataset
    correlations are ignored).
    """
    chi2_list: list[Chi2] = []

    if data is not None:
        assert eft_model is not None and fit_covmat is not None

        def _make_chi2_fn(
            cv: jnp.ndarray, inv_c: jnp.ndarray, s: int, e: int
        ) -> Callable[[jnp.ndarray], jnp.ndarray]:
            def _chi2(coeffs: jnp.ndarray) -> jnp.ndarray:
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
