"""
smefit.gradient_descent.py

Gradient-descent minimisation utility built on optax.
"""

import logging

import jax
import jax.numpy as jnp
import optax

log = logging.getLogger(__name__)


def best_fit_point(gd_best_fit):
    """The point at which nodes evaluate the likelihood around its best fit.

    The Fisher information matrices, and through them the PCA, are evaluated
    here. Outside a ``fitcontext`` it is the gradient-descent result. Under
    ``fits::fitcontext`` the namespace supplies a fit's stored best-fit point
    instead (see ``produce_fitcontext``), so no gradient descent is run, while
    the nodes needing an actual descent (``hessian_fit``, whitening with
    ``shift: gradient_descent``) keep reading ``gd_best_fit``.
    """
    return gd_best_fit


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
    start = chi2.baseline

    if sm_solution:
        log.info("GD best-fit: using baseline point.")
        return start
    n_steps = gradient_descent_settings.get("n_steps")
    tol = gradient_descent_settings.get("tol")
    log.info(
        "GD best-fit: running gradient descent (max_steps=%d) from baseline.", n_steps
    )
    return gd_minimize(chi2, optimizer, start, n_steps=n_steps, tol=tol)


def gd_minimize(chi2, optimizer, start, n_steps=2000, tol=1e-8):
    """Minimise ``chi2`` via gradient descent using an optax optimizer.

    Parameters
    ----------
    chi2 : callable
        Scalar-valued function to minimise.
    optimizer : optax.GradientTransformation
        Optax optimizer built by ``produce_optimizer``.
    start : jnp.ndarray
        Initial parameter vector.
    n_steps : int, default 2000
        Maximum number of gradient-descent steps.
    tol : float, default 1e-8
        Gradient-norm convergence threshold; stops early when
        ``||grad chi2|| < tol``.

    Returns
    -------
    jnp.ndarray
        Parameter vector at the minimum.
    """
    opt_state = optimizer.init(start)
    grad_chi2 = jax.jit(jax.value_and_grad(chi2))
    c = start
    for step in range(n_steps):
        val, grads = grad_chi2(c)
        updates, opt_state = optimizer.update(grads, opt_state)
        c = optax.apply_updates(c, updates)
        if float(jnp.linalg.norm(grads)) < tol:
            log.info(
                "Gradient descent: converged at step %d, chi2=%.6f", step, float(val)
            )
            break

        if step % 100 == 0:
            log.info(f"Epoch {step}, loss: {val:.3f}")

    else:
        log.info(
            "Gradient descent: reached max steps (%d), chi2=%.6f",
            n_steps,
            float(chi2(c)),
        )
    return c
