"""
smefit.ultranest_fit

UltraNest nested-sampling fitting routine, returning a FitResult node.
"""

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import ultranest
import ultranest.stepsampler as ustepsampler

from smefit.chi2 import Chi2
from smefit.core import CoefficientGroup
from smefit.fit_result import FitResult
from smefit.priors import JointPrior
from smefit.utils import resolve_posterior
from smefit.whitening import WhitenTransformNode

log = logging.getLogger(__name__)


def ultranest_fit(
    prior: JointPrior,
    chi2: Chi2,
    coefficients: CoefficientGroup,
    ultranest_settings: Mapping,
    whitening_transformation: WhitenTransformNode = None,
    n_samples: int = 10000,
) -> FitResult:
    """Run UltraNest nested sampling and return a FitResult.

    Reportengine provider node: arguments resolved by name from the DAG.

    Parameters
    ----------
    prior : Prior
        Joint prior over free coefficients (provides prior_transform).
    chi2 : Chi2
        Chi-squared callable built by produce_chi2.
    coefficients : CoefficientGroup
        Coefficient group (used to resolve derived coefficients from free ones).
    ultranest_settings : dict
        Settings for the UltraNest sampler.
    whitening_transformation : WhitenTransform, optional
        Affine whitening transform. When set, the sampler works in the
        whitened space c_w and evaluates chi2(transform.to_physical(c_w)).
    n_samples : int, optional
        Number of posterior samples to draw from the full set of UltraNest samples.

    Returns
    -------
    FitResult
    """
    if whitening_transformation is not None:
        log.info("Using whitening transformation in UltraNest fit.")
        _chi2 = chi2.whitened(whitening_transformation)
        _coeffs = coefficients.whitened(whitening_transformation)
    else:
        _chi2, _coeffs = chi2, coefficients

    log.info(
        "Running UltraNest fit for free coefficients: %s",
        _coeffs.free_names,
    )

    # set the ultranest seed
    np.random.seed(ultranest_settings["ultranest_seed"])

    log_likelihood_fn: Callable[[np.ndarray], Any]
    prior_transform_fn: Callable[[np.ndarray], np.ndarray]

    if ultranest_settings["ReactiveNS_settings"]["vectorized"]:
        _logl_vmap = jax.jit(jax.vmap(lambda p: -_chi2(p) / 2.0))

        def log_likelihood_vec(params_batch: np.ndarray) -> np.ndarray:
            return np.array(_logl_vmap(jnp.array(params_batch)), dtype=np.float64)

        _pt_vmap = jax.jit(jax.vmap(prior.prior_transform))

        def prior_transform_vec(unit_cubes: np.ndarray) -> np.ndarray:
            return np.array(_pt_vmap(jnp.array(unit_cubes)), dtype=np.float64)

        log_likelihood_fn = log_likelihood_vec
        prior_transform_fn = prior_transform_vec

    else:
        _logl_jit = jax.jit(lambda p: -_chi2(p) / 2.0)

        def log_likelihood_single(params: np.ndarray) -> float:
            return float(_logl_jit(jnp.array(params)))

        _pt_jit = jax.jit(prior.prior_transform)

        def prior_transform_single(unit_cube: np.ndarray) -> np.ndarray:
            return np.array(_pt_jit(jnp.array(unit_cube)), dtype=np.float64)

        log_likelihood_fn = log_likelihood_single
        prior_transform_fn = prior_transform_single

    sampler = ultranest.ReactiveNestedSampler(
        prior.param_names,
        log_likelihood_fn,
        prior_transform_fn,
        **ultranest_settings["ReactiveNS_settings"],
    )

    if ultranest_settings["SliceSampler_settings"]:

        sampler.stepsampler = ustepsampler.SliceSampler(
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            **ultranest_settings["SliceSampler_settings"],
        )

    t1 = time.time()
    result = sampler.run(**ultranest_settings["Run_settings"])
    t2 = time.time()
    log.info(f"UltraNest fit completed in {((t2 - t1) / 60.0):.2f} minutes.")
    sampler.print_results()

    if ultranest_settings["sampler_plot"]:
        log.info("Plotting sampler plots")
        # Store run plots to ultranest_logs folder (within output_path folder)
        sampler.plot()

    # Extract results
    logz = float(result["logz"])
    max_logl = float(result["maximum_likelihood"]["logl"])
    best_free = jnp.array(result["maximum_likelihood"]["point"])

    # Subsample posterior to n_samples
    full_samples = jnp.array(result["samples"])
    n_posterior_samples = n_samples
    if n_posterior_samples > full_samples.shape[0]:
        n_posterior_samples = full_samples.shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior "
            f"samples computed by UltraNest. Setting the number of resampled posterior "
            f"samples to {n_posterior_samples}"
        )
    posterior_free = full_samples[:n_posterior_samples]

    samples, best_fit_point = resolve_posterior(_coeffs, posterior_free, best_free)

    return FitResult(
        free_parameters=_coeffs.free_names,
        best_fit_point=best_fit_point,
        max_loglikelihood=max_logl,
        num_data=_chi2.num_data,
        logz=logz,
        samples=samples,
        prior_specs=prior.prior_specs,
        whitening_transformation=whitening_transformation,
        whitening_active=whitening_transformation is not None,
    )
