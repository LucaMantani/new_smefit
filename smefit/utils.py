"""
smefit.utils.py

Utility functions for the smefit framework.
"""

import csv
import logging
import time

import jax
import jax.numpy as jnp

from smefit.fit_result import FitResult

log = logging.getLogger(__name__)


def apply_whitening(chi2, coefficients, whitening_matrix):
    """Transform chi2 and coefficients into whitened space.

    Returns the transformed chi2 callable and whitened CoefficientGroup.
    """
    _chi2 = lambda c_w: chi2(whitening_matrix @ c_w)
    resolve_coeffs = coefficients.whitened(whitening_matrix)
    return _chi2, resolve_coeffs


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


def hessian_fit_SM(chi2, output_path):
    """Compute the Hessian of the chi2 function in zero and perform Hessian fit"""
    coeffs = jnp.zeros(chi2.nparam)
    hess_fn = jax.hessian(chi2)
    hess = 0.5 * hess_fn(coeffs)

    # Invert it to get the covariance matrix
    cov = jnp.linalg.inv(hess)

    samples = jax.random.multivariate_normal(
        jax.random.PRNGKey(0), mean=jnp.zeros(chi2.nparam), cov=cov, shape=(10000,)
    )

    fit = FitResult(
        free_parameters=chi2.param_names,
        best_fit_point={name: 0.0 for name in chi2.param_names},
        max_loglikelihood=float(-0.5 * chi2(coeffs)),
        num_data=chi2.num_data,
        logz=None,
        samples={name: samples[:, i] for i, name in enumerate(chi2.param_names)},
    )

    fit.print_summary()
    fit.write(output_path)

    return fit


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
