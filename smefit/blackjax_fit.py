"""
smefit.blackjax_fit

BlackJAX Bayesian fitting routine, producing a FitResult node.

The sampling algorithm is chosen with ``blackjax_settings.algorithm``;
the per-algorithm runners live in ``smefit.blackjax_samplers``.
Everything algorithm-independent — whitening, resolving free coefficients
back to the full set, assembling the FitResult — stays here.
"""

import logging
import os
import time

import jax

from smefit.blackjax_samplers import get_sampler
from smefit.fit_result import FitResult
from smefit.utils import resolve_posterior

log = logging.getLogger(__name__)


def blackjax_fit(
    prior,
    chi2,
    coefficients,
    blackjax_settings,
    whitening_transformation=None,
    n_samples=10000,
):
    """Run a BlackJAX sampler and return a FitResult.

    Reportengine provider node: arguments resolved by name from the DAG.

    Parameters
    ----------
    prior : Prior
        Joint prior over free coefficients.
    chi2 : Chi2
        Chi-squared callable built by produce_chi2.
    coefficients : CoefficientGroup
        Coefficient group (used to resolve derived coefficients from free ones).
    blackjax_settings : dict
        Settings for the BlackJAX sampler, including ``algorithm``.
    whitening_transformation : WhitenTransform, optional
        Affine whitening transform. When set, the sampler works in the
        whitened space c_w and evaluates chi2(transform.to_physical(c_w)).
    n_samples : int, optional
        Number of posterior samples to keep from the sampler's full output.

    Returns
    -------
    FitResult
    """
    if whitening_transformation is not None:
        log.info("Using whitening transformation in BlackJAX fit.")
        _chi2 = chi2.whitened(whitening_transformation)
        _coeffs = coefficients.whitened(whitening_transformation)
    else:
        _chi2, _coeffs = chi2, coefficients

    # Already in the sampler's coordinates: whitened when whitening is active.
    init_point = _chi2.baseline

    algorithm = blackjax_settings.get("algorithm", "nested_sampling")
    runner = get_sampler(algorithm)

    log.info("Fitting free coefficients: %s", ", ".join(_coeffs.free_names))

    rng_key = jax.random.PRNGKey(blackjax_settings["seed"])
    log.info(f"BlackJAX initialisation seed: {rng_key}")

    log_likelihood = jax.jit(lambda p: -_chi2(p) / 2.0)

    log_dir = blackjax_settings["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    t0 = time.time()
    out = runner(
        rng_key, prior, log_likelihood, n_samples, blackjax_settings, init_point
    )
    log.info(
        "BlackJAX '%s' fit completed in %.2f minutes.",
        algorithm,
        (time.time() - t0) / 60.0,
    )

    samples, best_fit_point = resolve_posterior(_coeffs, out.samples, out.best_point)

    return FitResult(
        free_parameters=_coeffs.free_names,
        best_fit_point=best_fit_point,
        max_loglikelihood=out.max_loglikelihood,
        num_data=_chi2.num_data,
        logz=out.logz,
        samples=samples,
        prior_specs=prior.prior_specs,
        whitening_transformation=whitening_transformation,
        whitening_active=whitening_transformation is not None,
    )
