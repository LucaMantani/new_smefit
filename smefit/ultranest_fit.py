"""
smefit.ultranest_fit

UltraNest nested-sampling fitting routine, returning a FitResult node.
"""

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import ultranest
import ultranest.stepsampler as ustepsampler

from smefit.fit_result import FitResult

log = logging.getLogger(__name__)


def ultranest_fit(
    prior, chi2, coefficients, ultranest_settings, data=None, whitening_matrix=None
):
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
    data : DataGroup, optional
        Observed data (for num_data). None when running without datasets.
    whitening_matrix : jnp.ndarray, optional
        Unwhitening matrix W (shape n_free x n_free). When set, the sampler
        works in the whitened space c_w and evaluates chi2(W @ c_w).

    Returns
    -------
    FitResult
    """
    if whitening_matrix is not None:
        log.info("Using whitening matrix in UltraNest fit.")
        W = whitening_matrix
        _chi2 = lambda c_w: chi2(W @ c_w)
        resolve_coeffs = coefficients.whitened(W)
    else:
        _chi2 = chi2
        resolve_coeffs = coefficients

    log.info(
        "Running UltraNest fit for free coefficients: %s",
        resolve_coeffs.free_names,
    )

    # set the ultranest seed
    np.random.seed(ultranest_settings["ultranest_seed"])

    if ultranest_settings["ReactiveNS_settings"]["vectorized"]:
        _logl_vmap = jax.jit(jax.vmap(lambda p: -_chi2(p) / 2.0))

        def log_likelihood(params_batch):
            return np.array(_logl_vmap(jnp.array(params_batch)), dtype=np.float64)

        _pt_vmap = jax.jit(jax.vmap(prior.prior_transform))

        def prior_transform(unit_cubes):
            return np.array(_pt_vmap(jnp.array(unit_cubes)), dtype=np.float64)

    else:
        _logl_jit = jax.jit(lambda p: -_chi2(p) / 2.0)

        def log_likelihood(params):
            return float(_logl_jit(jnp.array(params)))

        _pt_jit = jax.jit(prior.prior_transform)

        def prior_transform(unit_cube):
            return np.array(_pt_jit(jnp.array(unit_cube)), dtype=np.float64)

    sampler = ultranest.ReactiveNestedSampler(
        prior.param_names,
        log_likelihood,
        prior_transform,
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

    # Posterior samples: shape (n_samples, n_free)
    posterior_free = jnp.array(result["samples"])
    all_resolved = jax.vmap(resolve_coeffs.resolve)(posterior_free)
    samples = {name: all_resolved[:, i] for i, name in enumerate(resolve_coeffs.names)}

    # Best-fit full coefficient vector (free + derived)
    best_resolved = resolve_coeffs.resolve(best_free)
    best_fit_point = {
        name: float(best_resolved[i]) for i, name in enumerate(resolve_coeffs.names)
    }

    return FitResult(
        free_parameters=resolve_coeffs.free_names,
        best_fit_point=best_fit_point,
        max_loglikelihood=max_logl,
        num_data=data.num_data if data is not None else 0,
        logz=logz,
        samples=samples,
    )
