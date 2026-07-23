"""
colibri.blackjax_fit.py

This module contains the BlackJAX Bayesian fitting routine of Colibri.

"""

import logging
import os
import time

import anesthetic
import blackjax
import jax
import jax.numpy as jnp
import tqdm
from blackjax.ns.utils import ess, finalise, log_weights, sample
from jax.scipy.special import logsumexp

from smefit.fit_result import FitResult
from smefit.utils import resolve_posterior
from smefit.whitening import apply_whitening

log = logging.getLogger(__name__)


def blackjax_fit(
    prior,
    chi2,
    coefficients,
    blackjax_settings,
    whitening_transformation=None,
    n_samples=10000,
):
    """Run BlackJAX nested sampling and return a FitResult.

    Reportengine provider node: arguments resolved by name from the DAG.

    Parameters
    ----------
    prior : Prior
        Joint prior over free coefficients (provides prior_transform).
    chi2 : Chi2
        Chi-squared callable built by produce_chi2.
    coefficients : CoefficientGroup
        Coefficient group (used to resolve derived coefficients from free ones).
    blackjax_settings : dict
        Settings for the BlackJAX sampler.
    whitening_transformation : WhitenTransform, optional
        Affine whitening transform. When set, the sampler works in the
        whitened space c_w and evaluates chi2(transform.to_physical(c_w)).
    n_samples : int, optional
        Number of posterior samples to draw from the full set of BlackJAX samples.

    Returns
    -------
    FitResult
    """
    if whitening_transformation is not None:
        log.info("Using whitening transformation in BlackJAX fit.")
        _chi2, resolve_coeffs = apply_whitening(
            chi2, coefficients, whitening_transformation
        )
    else:
        _chi2 = chi2
        resolve_coeffs = coefficients

    # set the BlackJAX seed
    rng_key = jax.random.PRNGKey(blackjax_settings["seed"])
    log.info(f"BlackJAX initialisation seed: {rng_key}")
    n_dims = len(prior.param_names)
    n_live = blackjax_settings["n_live"]
    n_delete = int(blackjax_settings["delete_fraction"] * n_live)

    inital_particles = prior.sample(rng_key, n_live)

    log_likelihood = jax.jit(lambda p: -_chi2(p) / 2.0)

    algo = blackjax.nss(
        logprior_fn=prior.log_prob,
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=int(blackjax_settings["repeats"] * n_dims),
    )

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    state = algo.init(inital_particles)

    dead = []

    t0 = time.time()
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not state.logZ_live - state.logZ < blackjax_settings["log_precision"]:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)
    t1 = time.time()

    log.info(f"BlackJAX fit completed in {((t1 - t0) / 60.0):.2f} minutes.")

    final_states = finalise(state, dead)
    rng_key, ess_key, weights_key, sample_key = jax.random.split(rng_key, 4)

    ess_value = int(ess(ess_key, final_states))
    logw = log_weights(rng_key, final_states)
    logzs = logsumexp(logw, axis=0)
    full_samples = sample(sample_key, final_states, ess_value)

    # Get number of posterior samples to resample
    n_posterior_samples = n_samples

    # Check if we have enough samples
    if n_posterior_samples > full_samples.shape[0]:
        n_posterior_samples = full_samples.shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior "
            f"samples computed by BlackJAX. Setting the number of resampled posterior "
            f"samples to {n_posterior_samples}"
        )

    # Take first n_posterior_samples samples from the full posterior samples for resampling
    posterior_free = full_samples[:n_posterior_samples]

    # write out an anesthetic dataframe
    nested_samples = anesthetic.NestedSamples(
        data=final_states.particles,
        logL=final_states.loglikelihood,
        logL_birth=final_states.loglikelihood_birth,
        columns=prior.param_names,
    )
    # write nested_samples.csv to blackjax_logs
    log_dir = blackjax_settings["log_dir"]
    os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist
    nested_samples.to_csv(log_dir + "/nested_samples.csv")

    # Compute bayesian metrics (similar to UltraNest)
    # Find maximum likelihood point
    best_free_index = jnp.argmax(final_states.loglikelihood)
    max_logl = float(final_states.loglikelihood[best_free_index])
    best_free = final_states.particles[best_free_index]

    samples, best_fit_point = resolve_posterior(
        resolve_coeffs, posterior_free, best_free
    )

    return FitResult(
        free_parameters=resolve_coeffs.free_names,
        best_fit_point=best_fit_point,
        max_loglikelihood=max_logl,
        num_data=chi2.num_data,
        logz=float(logzs.mean()),
        samples=samples,
        prior_specs=prior.prior_specs,
        whitening_transformation=whitening_transformation,
        whitening_active=whitening_transformation is not None,
    )
