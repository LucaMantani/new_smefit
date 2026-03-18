"""
smefit.utils.py

Utility functions for the smefit framework.
"""

import time

import jax
import jax.numpy as jnp

from smefit.fit_result import FitResult


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
