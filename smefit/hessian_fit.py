"""
smefit.hessian_fit.py

Hessian-based fitting routine returning a FitResult node.

The best-fit point is determined by the ``gd_best_fit`` node (either the SM
point c=0 or the gradient-descent minimum).  The posterior is then
approximated as a multivariate Gaussian whose covariance is the inverse of
the Hessian of the chi2 evaluated at the best-fit point.
"""

import logging

import jax
import jax.numpy as jnp

from smefit.fit_result import FitResult
from smefit.gradient_descent import gd_minimize

log = logging.getLogger(__name__)


def gd_best_fit(chi2, optimizer, gradient_descent_settings):
    """Determine the best-fit coefficient vector via gradient descent.

    This function is a reportengine provider node: its return value is
    available as ``gd_best_fit`` to any downstream node (e.g. ``hessian_fit``).

    Parameters
    ----------
    chi2 : Chi2
        Chi-squared closure built by ``produce_chi2``.
    optimizer : optax.GradientTransformation
        Optax optimizer built by ``produce_optimizer``.
    gradient_descent_settings : dict
        Settings dict produced by ``parse_gradient_descent_settings``.

    Returns
    -------
    jnp.ndarray
        Best-fit coefficient vector of shape ``(n_free,)``.
    """
    sm_solution = gradient_descent_settings.get("sm_solution")
    zeros = jnp.zeros(chi2.nparam)
    if sm_solution:
        log.info("GD best-fit: using SM point (c=0).")
        return zeros
    n_steps = gradient_descent_settings.get("n_steps")
    tol = gradient_descent_settings.get("tol")
    log.info("GD best-fit: running gradient descent (max_steps=%d) from c=0.", n_steps)
    return gd_minimize(chi2, optimizer, zeros, n_steps=n_steps, tol=tol)


def hessian_fit(eft_model, chi2, gd_best_fit, hessian_settings):
    """Approximate the posterior with a Gaussian around the chi2 minimum.

    This function is a reportengine provider node: its arguments are resolved
    by name from the dependency graph and its return value (``FitResult``) is
    available as ``hessian_fit`` to downstream nodes and actions.

    Parameters
    ----------
    eft_model : EFTModel
        The EFT model used to resolve free → full coefficient space.
    chi2 : Chi2
        Chi-squared closure built by ``produce_chi2``.
    gd_best_fit : jnp.ndarray
        Best-fit coefficient vector produced by the ``gd_best_fit`` node.
    hessian_settings : dict
        Settings dict produced by ``parse_hessian_settings``.

    Returns
    -------
    FitResult
    """
    n_samples = hessian_settings.get("n_samples")
    seed = hessian_settings.get("seed")

    c_best = gd_best_fit

    # Hessian of chi2 at the best-fit point.
    # The log-likelihood is -chi2/2, so its Hessian is -(1/2)*d²chi2/dc².
    # The covariance is the inverse of the negative Hessian of the log-likelihood,
    # i.e. cov = inv(0.5 * d²chi2/dc²).
    hess = 0.5 * jax.hessian(chi2)(c_best)
    cov = jnp.linalg.inv(hess)

    chi2_val = float(chi2(c_best))
    max_loglikelihood = -chi2_val / 2.0

    # Draw Gaussian samples around the best-fit point
    key = jax.random.PRNGKey(seed)
    samples_free = jax.random.multivariate_normal(
        key, mean=c_best, cov=cov, shape=(n_samples,)
    )

    # Resolve full coefficient vector (free + derived) for each sample
    all_resolved = jax.vmap(eft_model.coefficients.resolve)(samples_free)
    samples = {
        name: all_resolved[:, i] for i, name in enumerate(eft_model.coefficients.names)
    }

    best_resolved = eft_model.coefficients.resolve(c_best)
    best_fit_point = {
        name: float(best_resolved[i])
        for i, name in enumerate(eft_model.coefficients.names)
    }

    return FitResult(
        free_parameters=eft_model.coefficients.free_names,
        best_fit_point=best_fit_point,
        max_loglikelihood=max_loglikelihood,
        num_data=chi2.num_data,
        samples=samples,
    )
