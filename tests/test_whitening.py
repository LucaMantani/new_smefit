"""Unit tests for smefit.whitening — WhitenTransform and its dispatch workers."""

import jax.numpy as jnp
import pytest

from smefit.chi2 import Chi2
from smefit.core import Coefficient, CoefficientGroup
from smefit.whitening import (
    WhitenTransform,
    _build_matrix,
    _disabled,
    _no_shift,
    _with_shift,
    apply_whitening,
)

_WHITENING = {"sigma_prior": 5.0, "eps": 1e-8, "shift": False}
_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}


def _quadratic_chi2(names=("OpA", "OpB")):
    """chi2 = sum(c**2), Hessian = 2*I everywhere (shift-invariant curvature)."""
    return Chi2(lambda c: jnp.sum(c**2), list(names), num_data=1)


def _free(name):
    return Coefficient(name=name, free=True, prior=_PRIOR)


# ---------------------------------------------------------------------------
# WhitenTransform
# ---------------------------------------------------------------------------


def test_to_physical_and_to_whitened_roundtrip():
    matrix = jnp.array([[2.0, 0.0], [0.5, 1.5]])
    shift = jnp.array([0.3, -0.2])
    transform = WhitenTransform(matrix=matrix, shift=shift)

    c_w = jnp.array([1.0, -1.0])
    c_phys = transform.to_physical(c_w)
    assert jnp.allclose(c_phys, matrix @ c_w + shift)
    assert jnp.allclose(transform.to_whitened(c_phys), c_w)


def test_to_dict_from_dict_roundtrip():
    matrix = jnp.array([[2.0, 0.0], [0.5, 1.5]])
    shift = jnp.array([0.3, -0.2])
    transform = WhitenTransform(matrix=matrix, shift=shift)

    recovered = WhitenTransform.from_dict(transform.to_dict())
    assert jnp.allclose(recovered.matrix, matrix)
    assert jnp.allclose(recovered.shift, shift)


# ---------------------------------------------------------------------------
# _build_matrix / _no_shift / _with_shift / _disabled
# ---------------------------------------------------------------------------


def test_build_matrix_matches_hand_computed_cholesky():
    chi2 = _quadratic_chi2()
    center = jnp.zeros(2)
    matrix = _build_matrix(chi2, _WHITENING, center)
    # H = 2*I + eps*I ≈ 2*I -> L = sqrt(2)*I -> W = L^-T = I/sqrt(2)
    expected = jnp.eye(2) / jnp.sqrt(2.0)
    assert jnp.allclose(matrix, expected, atol=1e-5)


def test_no_shift_evaluates_hessian_at_zero():
    chi2 = _quadratic_chi2()
    transform = _no_shift(chi2, _WHITENING)
    assert jnp.allclose(transform.shift, jnp.zeros(2))
    assert jnp.allclose(transform.matrix, jnp.eye(2) / jnp.sqrt(2.0), atol=1e-5)


def test_with_shift_centers_on_given_point():
    chi2 = _quadratic_chi2()
    gd_best_fit = jnp.array([1.0, -2.0])
    transform = _with_shift(chi2, gd_best_fit, _WHITENING)
    assert jnp.allclose(transform.shift, gd_best_fit)
    # Hessian of sum(c**2) is constant (2*I), so matrix agrees with the no-shift case.
    assert jnp.allclose(transform.matrix, jnp.eye(2) / jnp.sqrt(2.0), atol=1e-5)


def test_disabled_takes_no_arguments_and_returns_none():
    assert _disabled() is None


# ---------------------------------------------------------------------------
# apply_whitening
# ---------------------------------------------------------------------------


def test_apply_whitening_chi2_equivalent():
    """Whitened chi2 evaluated at c_w equals original chi2 at W @ c_w."""
    W = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    transform = WhitenTransform(matrix=W, shift=jnp.zeros(2))
    chi2 = lambda c: jnp.sum(c**2)
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])

    whitened_chi2, _ = apply_whitening(chi2, cg, transform)

    c_w = jnp.array([1.0, 1.0])
    assert float(whitened_chi2(c_w)) == pytest.approx(float(chi2(W @ c_w)))


def test_apply_whitening_chi2_equivalent_with_shift():
    """Whitened chi2 at c_w equals original chi2 at W @ c_w + shift."""
    W = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    shift = jnp.array([0.5, -1.0])
    transform = WhitenTransform(matrix=W, shift=shift)
    chi2 = lambda c: jnp.sum(c**2)
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])

    whitened_chi2, _ = apply_whitening(chi2, cg, transform)

    c_w = jnp.array([1.0, 1.0])
    assert float(whitened_chi2(c_w)) == pytest.approx(float(chi2(W @ c_w + shift)))


def test_apply_whitening_coeff_group():
    """Returned CoefficientGroup should have _transform set (whitening active)."""
    transform = WhitenTransform(matrix=jnp.eye(2), shift=jnp.zeros(2))
    chi2 = lambda c: jnp.sum(c**2)
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])

    _, whitened_cg = apply_whitening(chi2, cg, transform)

    assert whitened_cg._transform is not None
