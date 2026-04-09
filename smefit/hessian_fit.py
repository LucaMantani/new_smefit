"""
smefit.hessian_fit.py

Hessian-based fitting routine returning a FitResult node.

The best-fit point is either assumed to be the SM point (c=0) or found
numerically via gradient descent.  In both cases the posterior is
approximated as a multivariate Gaussian whose covariance is the inverse
of the Hessian of the chi2 evaluated at the best-fit point.
"""

import logging

import jax
import jax.numpy as jnp
import optax

from smefit.fit_result import FitResult

log = logging.getLogger(__name__)


def hessian_fit(eft_model, chi2, optimizer, hessian_settings):
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
    optimizer : optax.GradientTransformation
        Optax optimizer built by ``produce_optimizer``.
    hessian_settings : dict
        Settings dict produced by ``parse_hessian_settings``.

    Returns
    -------
    FitResult
    """
    sm_solution = hessian_settings.get("sm_solution")
    n_steps = hessian_settings.get("n_steps")
    tol = hessian_settings.get("tol")
    n_samples = hessian_settings.get("n_samples")
    seed = hessian_settings.get("seed")

    n_free = len(eft_model.coefficients.free_names)
    zeros = jnp.zeros(n_free)

    if sm_solution:
        log.info("Hessian fit: using SM point (c=0) as the best-fit point.")
        c_best = zeros
    else:
        log.info(
            "Hessian fit: running gradient descent (max_steps=%d) from c=0.", n_steps
        )
        opt_state = optimizer.init(zeros)
        grad_chi2 = jax.jit(jax.value_and_grad(chi2))
        c = zeros
        for step in range(n_steps):
            val, grads = grad_chi2(c)
            updates, opt_state = optimizer.update(grads, opt_state)
            c = optax.apply_updates(c, updates)
            if float(jnp.linalg.norm(grads)) < tol:
                log.info(
                    "Hessian fit: converged at step %d, chi2=%.6f", step, float(val)
                )
                break
        else:
            log.info(
                "Hessian fit: reached max steps (%d), chi2=%.6f",
                n_steps,
                float(chi2(c)),
            )
        c_best = c

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
