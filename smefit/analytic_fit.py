"""
smefit.analytic_fit.py

Analytic (linear-theory) fitting routine returning a Fit node.
"""

import logging

import jax
import jax.numpy as jnp

from smefit.fit_result import Fit, name_from_output_path
from smefit.utils import resolve_posterior

log = logging.getLogger(__name__)


def analytic_fit(
    eft_model, data, fit_covmat, chi2, n_samples=10000, seed=42, output_path=None
):
    """Compute the analytic best-fit point and Gaussian uncertainty for a linear EFT model.

    This function is a reportengine provider node: its arguments are resolved by
    name from the dependency graph and its return value (``Fit``) is
    available as ``analytic_fit`` to downstream nodes and actions.

    Parameters
    ----------
    eft_model : EFTModel
        The EFT model.  Must have ``use_quad=False``.
    data : DataGroup
        Observed data (``data.cv``, ``data.num_data``).
    fit_covmat : jnp.ndarray
        Covariance matrix used in the fit.
    chi2 : callable
        Chi-squared closure built by ``produce_chi2``.
    n_samples : int, optional
        Number of Gaussian samples to draw around the best-fit point (default 10000).
    seed : int, optional
        Random seed for sample generation (default 42).
    output_path : pathlib.Path, optional
        Directory the fit will be written to. Its name becomes the fit's name,
        so a fit is known by the same identity while it runs as when it is
        loaded back; without it the fit falls back to its type.

    Returns
    -------
    Fit
    """
    if chi2.has_external:
        raise ValueError(
            "analytic_fit is incompatible with external_chi2. "
            "Remove external_chi2 from the runcard or use a numerical optimizer."
        )

    if eft_model.use_quad:
        raise ValueError(
            "analytic_fit requires a linear EFT model (use_quad=False). "
            "Set 'use_quad: False' in the runcard."
        )

    log.info(
        "Running analytic fit for free coefficients: %s",
        eft_model.coefficients.free_names,
    )
    n_free = len(eft_model.coefficients.free_names)
    zeros = jnp.zeros(n_free)

    # Baseline predictions at zero coefficients (= SM predictions)
    baseline = eft_model.forward_map(zeros)

    # Jacobian of forward_map: shape (n_data, n_free)
    # This ensures the right object when derived coefficients are present,
    # as the forward_map will resolve them.
    J = jax.jacobian(eft_model.forward_map)(zeros)

    # Residual vector at zero
    delta = data.cv - baseline

    # Analytic solution: c_best = (J^T C^-1 J)^-1 J^T C^-1 delta
    inv_fit_covmat = jnp.linalg.inv(fit_covmat)
    H = J.T @ inv_fit_covmat @ J  # (n_free, n_free)
    c_best = jnp.linalg.solve(H, J.T @ inv_fit_covmat @ delta)  # (n_free,)
    cov_c = jnp.linalg.inv(H)  # (n_free, n_free)

    # Chi2 and log-likelihood at best-fit point
    chi2_val = float(chi2(c_best))
    max_loglikelihood = -chi2_val / 2.0

    # Draw Gaussian samples around best-fit point
    key = jax.random.PRNGKey(seed)
    samples_free = jax.random.multivariate_normal(
        key, mean=c_best, cov=cov_c, shape=(n_samples,)
    )  # (n_samples, n_free)

    samples, best_fit_point = resolve_posterior(
        eft_model.coefficients, samples_free, c_best
    )

    return Fit(
        free_parameters=eft_model.coefficients.free_names,
        best_fit_point=best_fit_point,
        max_loglikelihood=max_loglikelihood,
        num_data=data.num_data,
        samples=samples,
        fit_name=name_from_output_path(output_path),
        fit_type="analytic",
        use_quad=eft_model.use_quad,
    )
