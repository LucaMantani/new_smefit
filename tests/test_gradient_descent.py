"""Unit tests for smefit.gradient_descent — gd_minimize and gd_best_fit."""

import jax.numpy as jnp
import optax
import pytest

from smefit.chi2 import Chi2
from smefit.gradient_descent import gd_best_fit, gd_minimize


def _quadratic(c):
    """Simple quadratic chi2 with minimum at c=0."""
    return jnp.sum(c**2)


def _shifted_quadratic(c):
    """Quadratic with minimum at c=[2.0]."""
    return (c[0] - 2.0) ** 2


def test_gd_minimize_converges_to_zero():
    """Starting from a non-zero point, minimise c^2 → should end near 0."""
    optimizer = optax.adam(1e-2)
    start = jnp.array([1.0])
    result = gd_minimize(_quadratic, optimizer, start, n_steps=5000, tol=1e-6)
    assert float(result[0]) == pytest.approx(0.0, abs=1e-3)


def test_gd_minimize_converges_to_known_minimum():
    """Minimise (c-2)^2 → should end near c=2."""
    optimizer = optax.adam(1e-2)
    start = jnp.array([0.0])
    result = gd_minimize(_shifted_quadratic, optimizer, start, n_steps=5000, tol=1e-6)
    assert float(result[0]) == pytest.approx(2.0, abs=1e-2)


def test_gd_minimize_returns_array():
    optimizer = optax.adam(1e-2)
    start = jnp.array([1.0, -1.0])
    result = gd_minimize(_quadratic, optimizer, start, n_steps=100)
    assert result.shape == (2,)


def test_gd_minimize_max_steps_reached(caplog):
    """With tol=0 and very few steps, the max-steps log line should appear."""
    import logging

    optimizer = optax.adam(1e-2)
    start = jnp.array([10.0])
    with caplog.at_level(logging.INFO, logger="smefit.gradient_descent"):
        gd_minimize(_quadratic, optimizer, start, n_steps=3, tol=0.0)
    assert any("max steps" in r.message for r in caplog.records)


def test_gd_minimize_early_stop(caplog):
    """Starting exactly at the minimum (c=0), grad=0 → should stop at step 0."""
    import logging

    optimizer = optax.adam(1e-2)
    start = jnp.zeros(1)
    with caplog.at_level(logging.INFO, logger="smefit.gradient_descent"):
        gd_minimize(_quadratic, optimizer, start, n_steps=1000, tol=1.0)
    assert any("converged" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# gd_best_fit — baseline handling
# ---------------------------------------------------------------------------


def test_gd_best_fit_sm_solution_returns_baseline():
    """sm_solution=True returns the chi2 baseline instead of zeros."""
    baseline = jnp.array([0.7, -0.3])
    chi2 = Chi2(_quadratic, ["OpA", "OpB"], num_data=1, baseline=baseline)
    result = gd_best_fit(chi2, optax.adam(1e-2), {"sm_solution": True})
    assert jnp.allclose(result, baseline)


def test_gd_best_fit_sm_solution_default_baseline_is_zeros():
    """With no baseline set, sm_solution=True still returns zeros (unchanged)."""
    chi2 = Chi2(_quadratic, ["OpA", "OpB"], num_data=1)
    result = gd_best_fit(chi2, optax.adam(1e-2), {"sm_solution": True})
    assert jnp.allclose(result, jnp.zeros(2))


def test_gd_best_fit_starts_from_baseline():
    """Optimisation starts from the baseline point (converges to the minimum)."""
    baseline = jnp.array([5.0])
    chi2 = Chi2(_shifted_quadratic, ["OpA"], num_data=1, baseline=baseline)
    result = gd_best_fit(
        chi2, optax.adam(1e-2), {"sm_solution": False, "n_steps": 5000, "tol": 1e-6}
    )
    assert float(result[0]) == pytest.approx(2.0, abs=1e-2)
