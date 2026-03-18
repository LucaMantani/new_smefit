"""
smefit.utils.py

Utility functions for the smefit framework.
"""

import time

import jax
import jax.numpy as jnp


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
