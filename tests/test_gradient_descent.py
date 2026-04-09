"""Unit tests for smefit.gradient_descent — gd_minimize."""

import jax.numpy as jnp
import optax
import pytest

from smefit.gradient_descent import gd_minimize


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
