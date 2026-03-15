"""
smefit.ultranest_fit

UltraNest nested-sampling fitting routine, returning a FitResult node.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import ultranest
import ultranest.stepsampler as ustepsampler

from smefit.fit_result import FitResult

log = logging.getLogger(__name__)


def ultranest_fit(prior, chi2, eft_model, data, ultranest_settings):
    """Run UltraNest nested sampling and return a FitResult.

    Reportengine provider node: arguments resolved by name from the DAG.

    Parameters
    ----------
    prior : Prior
        Joint prior over free coefficients (provides prior_transform).
    chi2 : Chi2
        Chi-squared callable built by produce_chi2.
    eft_model : EFTModel
        EFT model (used to resolve derived coefficients from free ones).
    data : DataGroup
        Observed data (for num_data).
    ultranest_settings : dict
        Settings for the UltraNest sampler.

    Returns
    -------
    FitResult
    """

    # set the ultranest seed
    np.random.seed(ultranest_settings["ultranest_seed"])

    if ultranest_settings["ReactiveNS_settings"]["vectorized"]:
        _logl_vmap = jax.jit(jax.vmap(lambda p: -chi2(p) / 2.0))

        def log_likelihood(params_batch):
            return np.array(_logl_vmap(jnp.array(params_batch)), dtype=np.float64)

        _pt_vmap = jax.jit(jax.vmap(prior.prior_transform))

        def prior_transform(unit_cubes):
            return np.array(_pt_vmap(jnp.array(unit_cubes)), dtype=np.float64)

    else:
        _logl_jit = jax.jit(lambda p: -chi2(p) / 2.0)

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

    result = sampler.run(**ultranest_settings["Run_settings"])
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
    all_resolved = jax.vmap(eft_model.coefficients.resolve)(posterior_free)
    samples = {
        name: all_resolved[:, i] for i, name in enumerate(eft_model.coefficients.names)
    }

    # Best-fit full coefficient vector (free + derived)
    best_resolved = eft_model.coefficients.resolve(best_free)
    best_fit_point = {
        name: float(best_resolved[i])
        for i, name in enumerate(eft_model.coefficients.names)
    }

    return FitResult(
        free_parameters=eft_model.coefficients.free_names,
        best_fit_point=best_fit_point,
        max_loglikelihood=max_logl,
        num_data=data.num_data,
        logz=logz,
        samples=samples,
    )
