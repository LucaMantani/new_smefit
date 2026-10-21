"""
smefit.bayesian_update.py

Bayesian update: turn the posterior of a previous fit into the prior of a new one.
"""

from collections.abc import Mapping, Sequence

import jax
from reportengine.configparser import ConfigError

from smefit.core import CoefficientGroup
from smefit.fit_result import Fit
from smefit.priors import ExactPosteriorPrior, WhitenedToPhysicalPrior


def build_exact_posterior_prior(
    bayesian_update: Mapping,
    coefficients: CoefficientGroup,
    datasets: Sequence | None,
    external_chi2: Mapping | None = None,
) -> ExactPosteriorPrior:
    """Build ExactPosteriorPrior from a previous fit result and its saved runcard.

    ``bayesian_update`` is the mapping returned by
    :meth:`smefitConfig.parse_bayesian_update`, whose ``path`` is the resolved
    directory of the previous fit.

    Loads fit1 with :meth:`Fit.from_folder` — its numbers and the runcard it was
    run with — rebuilds chi2 for fit1's data from that runcard, and returns an
    ExactPosteriorPrior whose log_prob = log_prior_1 + log_likelihood_1.
    """
    fit_path = bayesian_update["path"]

    # --- Load previous fit: its result and the runcard it was run with ---
    try:
        prev_fit = Fit.from_folder(fit_path)
    except (KeyError, OSError, ValueError) as e:
        raise ConfigError(f"Could not load the previous fit at {fit_path}: {e}") from e

    if prev_fit.individual_fit:
        raise ConfigError(
            f"The fit at {fit_path} was run one coefficient at a time, "
            "so it holds an independent 1D posterior per coefficient rather than "
            "the joint posterior a Bayesian update needs as its prior."
        )

    prev = prev_fit.fit_results

    if prev.free_parameters != coefficients.free_names:
        raise ConfigError(
            f"Free parameters mismatch between previous fit and current runcard.\n"
            f"  Previous fit : {prev.free_parameters}\n"
            f"  Current fit  : {coefficients.free_names}\n"
            f"Both the set and the order of free parameters must match."
        )

    if prev.samples is None:
        raise ConfigError(f"Previous fit at {fit_path} has no posterior samples. ")

    # --- Check for dataset overlap ---
    if datasets:
        current_names = {ds["name"] for ds in datasets}
        prev_names = {ds["name"] for ds in prev_fit.setting("datasets", []) or []}
        overlap = current_names & prev_names
        if overlap:
            raise ConfigError(
                f"Datasets {sorted(overlap)} appear in both the current fit and the "
                "previous fit. This would double-count data in the Bayesian update."
            )

    # --- Check for external_chi2 overlap ---
    if external_chi2:
        current_ext = set(external_chi2.keys())
        prev_ext = set(prev_fit.setting("external_chi2", {}) or {})
        overlap = current_ext & prev_ext
        if overlap:
            raise ConfigError(
                f"External chi2 contributions {sorted(overlap)} appear in both the "
                "current fit and the previous fit. This would double-count data in "
                "the Bayesian update."
            )

    # Local import to avoid circular dependency
    from smefit.api import smefitAPI

    prev_runcard = prev_fit.rebuild_runcard()
    prev_chi2 = smefitAPI.chi2(**prev_runcard)
    log_likelihood_1 = jax.jit(lambda theta: -prev_chi2(theta) / 2.0)

    # --- Reconstruct prior_1 in physical space via the API (handles chains recursively) ---
    prior_1 = smefitAPI.prior(**prev_runcard)
    if prev.whitening_active:
        if prev.whitening_transformation is None:
            raise ConfigError(
                f"Previous fit at {fit_path} used whitening but "
                "no whitening_transformation was saved."
            )
        prior_1 = WhitenedToPhysicalPrior(prior_1, prev.whitening_transformation)

    return ExactPosteriorPrior(
        base_prior=prior_1,
        log_likelihood_1=log_likelihood_1,
        samples_dict=prev.samples,
        param_names=prev.free_parameters,
        source_path=str(fit_path),
    )
