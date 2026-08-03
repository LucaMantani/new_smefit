"""
smefit.utils.py

Utility functions for the smefit framework.
"""

import csv
import logging
import pathlib
import time

import jax
import jax.numpy as jnp
import yaml
from reportengine.configparser import ConfigError

from smefit.fit_result import FitResult
from smefit.priors import ExactPosteriorPrior, _WhitenedToPhysicalPrior

log = logging.getLogger(__name__)


def resolve_posterior(resolve_coeffs, posterior_free, best_free):
    """Resolve posterior samples and best-fit point from free to full coefficient space.

    Returns (samples_dict, best_fit_dict).
    """
    all_resolved = jax.vmap(resolve_coeffs.resolve)(posterior_free)
    samples = {name: all_resolved[:, i] for i, name in enumerate(resolve_coeffs.names)}

    best_resolved = resolve_coeffs.resolve(best_free)
    best_fit_point = {
        name: float(best_resolved[i]) for i, name in enumerate(resolve_coeffs.names)
    }
    return samples, best_fit_point


def ensure_list(x):
    """Ensure the input is a list.
    If the input is not a list, wrap it in a list.
    """
    if isinstance(x, list):
        return x
    return [x]


def chi2_timing(chi2, n_eval=1000):
    """Time the evaluation of the chi2 function."""
    coeffs = jnp.zeros(chi2.nparam)
    # Trigger JIT compilation before timing
    jax.block_until_ready(chi2(coeffs))
    # Now evaluate it n_eval times and time it
    start = time.perf_counter()
    for _ in range(n_eval):
        result = chi2(coeffs)
        jax.block_until_ready(result)
    end = time.perf_counter()
    print(f"Chi2 evaluation time: {(end - start) / n_eval:.4e} seconds")


def time_chi2_vec(
    chi2,
    output_path,
    batch_sample_sizes=None,
):
    """
    Time the vectorized chi2 across different batch sizes.

    Parameters
    ----------
    chi2 : callable
        The chi2 function that takes parameter vector(s)
    batch_sample_sizes : sequence of int, optional
        Batch sizes (number of parameter vectors per batch) to time
    output_path : pathlib.PosixPath
        Path to the output folder where chi2_times.csv will be saved
    """

    # Create vectorized version
    chi2_vec = jax.jit(jax.vmap(chi2, in_axes=(0,), out_axes=0))

    # Batch sizes to test - use provided or default
    if batch_sample_sizes is None:
        sizes = [1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000]
        log.info("Using default batch sample sizes")
    else:
        sizes = batch_sample_sizes
        log.info(f"Using custom batch sample sizes: {sizes}")

    # Set up CSV file path
    save_path = output_path / "chi2_times.csv"

    # Initialize CSV file with headers
    with open(save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["batch_size", "time_seconds", "relative_time"])

    log.info(f"Results will be saved incrementally to {save_path}")

    # Pre-generate samples for the largest size only
    log.info("Generating samples for log likelihood timing...")
    max_size = max(sizes)

    all_samples_batch = jnp.zeros((max_size, chi2.nparam))

    # Create subsets for each size
    samples_list = []
    for size in sizes:
        samples_list.append(all_samples_batch[:size])

    # Now time each batch size
    log.info("Timing different batch sizes...")
    times = []
    successful_sizes = []

    for i, size in enumerate(sizes):
        # Warm-up: compile the function by calling it once and waiting
        log.info("Warming up (JIT compilation)...")
        try:
            jax.block_until_ready(chi2_vec(samples_list[i]))
        except Exception as e:
            log.error(f"Warm-up failed: {e}")
            raise
        try:
            log.info(f"Timing batch size: {size}")
            t0 = time.perf_counter()
            result = chi2_vec(samples_list[i])
            jax.block_until_ready(result)
            t1 = time.perf_counter()
            avg_time = t1 - t0
            times.append(avg_time)
            successful_sizes.append(size)

            # Compute relative time (relative to first successful timing)
            relative_time = avg_time / times[0]

            # Append result to CSV immediately
            with open(save_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([size, avg_time, relative_time])

            log.info(
                f"Size: {size:6d}, Time: {avg_time:.6f} s, Relative: {relative_time:.4f}x"
            )

        except Exception as e:
            log.error(f"Error at batch size {size}: {e}")
            log.warning(
                f"Stopping timing. Results for batch sizes up to {successful_sizes[-1] if successful_sizes else 'none'} have been saved."
            )
            break

    if successful_sizes:
        log.info(f"Timing completed for {len(successful_sizes)} batch sizes")
        log.info(f"Final results saved to {save_path}")
    else:
        log.error("No batch sizes were successfully timed")

    return successful_sizes, times


def build_exact_posterior_prior(
    bayesian_update_path, coefficients, datasets, external_chi2=None
):
    """Build ExactPosteriorPrior from a previous fit result and its saved runcard.

    Reads fit1's fit_results.json and input/runcard.yaml, rebuilds chi2 for
    fit1's data, and returns an ExactPosteriorPrior whose
    log_prob = log_prior_1 + log_likelihood_1.
    """
    # --- Load previous fit result ---
    prev = FitResult.from_json(bayesian_update_path)

    if prev.free_parameters != coefficients.free_names:
        raise ConfigError(
            f"Free parameters mismatch between previous fit and current runcard.\n"
            f"  Previous fit : {prev.free_parameters}\n"
            f"  Current fit  : {coefficients.free_names}\n"
            f"Both the set and the order of free parameters must match."
        )

    if prev.samples is None:
        raise ConfigError(
            f"Previous fit at {bayesian_update_path} has no posterior samples. "
        )

    # --- Load previous runcard and rebuild chi2 via the smefit API ---
    runcard_path = pathlib.Path(bayesian_update_path) / "input" / "runcard.yaml"
    with runcard_path.open() as f:
        prev_rc = yaml.safe_load(f)

    # --- Check for dataset overlap ---
    if datasets:
        current_names = {ds["name"] for ds in datasets}
        prev_names = {ds["name"] for ds in prev_rc.get("datasets", [])}
        overlap = current_names & prev_names
        if overlap:
            raise ConfigError(
                f"Datasets {sorted(overlap)} appear in both the current fit and the "
                "previous fit. This would double-count data in the Bayesian update."
            )

    # --- Check for external_chi2 overlap ---
    if external_chi2:
        current_ext = set(external_chi2.keys())
        prev_ext = set(prev_rc.get("external_chi2", {}).keys())
        overlap = current_ext & prev_ext
        if overlap:
            raise ConfigError(
                f"External chi2 contributions {sorted(overlap)} appear in both the "
                "current fit and the previous fit. This would double-count data in "
                "the Bayesian update."
            )

    # Rebuilding fit1's chi2 below re-runs its RGE evolution from scratch: the
    # API call gets a fresh smefitConfig, so produce_rge_matrix's memo does not
    # apply. The matrix it would recompute is already on disk next to the result
    # we just loaded, so point the runcard at it.
    prev_rge = prev_rc.get("rge")
    if isinstance(prev_rge, dict) and not prev_rge.get("rg_matrix"):
        # Local import: smefit.rge monkey-patches wilson/ckmutil process-wide.
        from smefit.rge import RGEMatrix

        prev_rge_matrix = RGEMatrix.path_in(bayesian_update_path)
        if prev_rge_matrix.exists():
            prev_rge["rg_matrix"] = str(prev_rge_matrix)
            log.info("Reusing the previous fit's RGE matrix at %s.", prev_rge_matrix)

    # Local import to avoid circular dependency
    from smefit.api import smefitAPI

    prev_chi2 = smefitAPI.chi2(**prev_rc)
    log_likelihood_1 = jax.jit(lambda theta: -prev_chi2(theta) / 2.0)

    # --- Reconstruct prior_1 in physical space via the API (handles chains recursively) ---
    prior_1 = smefitAPI.prior(**prev_rc)
    if prev.whitening_active:
        if prev.whitening_transformation is None:
            raise ConfigError(
                f"Previous fit at {bayesian_update_path} used whitening but "
                "no whitening_transformation was saved."
            )
        prior_1 = _WhitenedToPhysicalPrior(prior_1, prev.whitening_transformation)

    return ExactPosteriorPrior(
        base_prior=prior_1,
        log_likelihood_1=log_likelihood_1,
        samples_dict=prev.samples,
        param_names=prev.free_parameters,
        source_path=str(bayesian_update_path),
    )


def run_prior_test(prior):
    n_free = len(prior.param_names)
    print(f"Prior parameters: {prior.param_names}")

    midpoint = prior.prior_transform(jnp.full(n_free, 0.5))
    print(f"prior_transform(0.5): {midpoint}")

    lower = prior.prior_transform(jnp.zeros(n_free))
    print(f"prior_transform(0.0): {lower}")

    upper = prior.prior_transform(jnp.ones(n_free))
    print(f"prior_transform(1.0): {upper}")

    lp = prior.log_prob(midpoint)
    print(f"log_prob(midpoint): {lp}")
    assert jnp.isfinite(lp), "log_prob returned non-finite value for in-support point"

    samples = prior.sample(jax.random.PRNGKey(0), n_samples=10)
    print(f"sample shape: {samples.shape}")

    print("Prior test passed.")


def run_test(eft_model, chi2):

    print(eft_model.coefficients.free_names)
    free = jnp.array([5.0, -3.0])

    print(eft_model.coefficients.resolve(free))

    print(eft_model.forward_map(free))

    print(chi2(free))
    # compute gradient
    grad_chi2 = jax.grad(chi2)(free)
    print(grad_chi2)
