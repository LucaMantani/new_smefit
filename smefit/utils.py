"""
smefit.utils.py

Low-level utility functions for the smefit framework.

This module sits at the bottom of the import graph: it must not import any
other ``smefit`` module, so that every module can import from it without
creating a cycle. A helper that needs a smefit domain object (a fit result,
a prior, ...) belongs next to that object instead.
"""

import csv
import logging
import time

import jax
import jax.numpy as jnp

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
