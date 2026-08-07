"""Unit tests for smefit.whitening — WhitenTransform and its dispatch workers."""

import logging

import jax.numpy as jnp
import pytest

from smefit.chi2 import Chi2
from smefit.core import Coefficient, CoefficientGroup
from smefit.whitening import (
    WhitenTransform,
    _build_matrix,
    _whitening_baseline_shift,
    _whitening_gradient_descent_shift,
    apply_whitening,
)

_WHITENING = {"sigma_prior": 5.0, "eps": 1e-8, "shift": "baseline"}
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
# _build_matrix / _whitening_baseline_shift / _whitening_gradient_descent_shift
# ---------------------------------------------------------------------------


def test_build_matrix_matches_hand_computed_cholesky():
    chi2 = _quadratic_chi2()
    center = jnp.zeros(2)
    matrix = _build_matrix(chi2, _WHITENING, center)
    # H = 2*I + eps*I ≈ 2*I -> L = sqrt(2)*I -> W = L^-T = I/sqrt(2)
    expected = jnp.eye(2) / jnp.sqrt(2.0)
    assert jnp.allclose(matrix, expected, atol=1e-5)


def test_build_matrix_raises_on_non_positive_definite_hessian():
    """Negative curvature -> ValueError, attributed to a saddle centre."""
    chi2 = Chi2(lambda c: -jnp.sum(c**2), ["OpA", "OpB"], num_data=1)
    center = jnp.zeros(2)
    with pytest.raises(ValueError, match="not positive definite") as excinfo:
        _build_matrix(chi2, _WHITENING, center)
    assert "saddle point" in str(excinfo.value)
    # eps regularises flat directions, not negative ones, so the message must
    # point at re-centring rather than at raising eps.
    assert "gradient_descent" in str(excinfo.value)


def test_build_matrix_tiny_negative_eigenvalue_is_not_called_a_saddle():
    """A flat direction rounding slightly negative is noise, not a saddle.

    H = diag(2, -1e-14): lam_min is below -eps, so H + eps*I is still
    indefinite, but it sits well inside the 100*machine_eps*lam_max noise floor
    in either precision (4e-14 in float64, 2e-5 in float32). H is diagonal, so
    eigvalsh recovers the tiny eigenvalue exactly even in float32.
    """
    chi2 = Chi2(lambda c: c[0] ** 2 - 5e-15 * c[1] ** 2, ["OpA", "OpB"], num_data=1)
    whitening = {**_WHITENING, "eps": 1e-15}
    with pytest.raises(ValueError, match="numerically negative") as excinfo:
        _build_matrix(chi2, whitening, jnp.zeros(2))
    assert "saddle" not in str(excinfo.value)


def test_build_matrix_regularises_flat_direction_with_eps():
    """A rank-deficient Hessian still builds, with the flat scale set by eps.

    chi2 = c0**2 leaves OpB flat: H + eps*I = diag(2, eps), so the whitened
    unit step along OpB reaches 1/sqrt(eps) in physical units — the quantity
    the flat-direction warning reports as sigma_prior/sqrt(eps).
    """
    chi2 = Chi2(lambda c: c[0] ** 2, ["OpA", "OpB"], num_data=1)
    matrix = _build_matrix(chi2, _WHITENING, jnp.zeros(2))
    expected = jnp.diag(jnp.array([1.0 / jnp.sqrt(2.0), _WHITENING["eps"] ** -0.5]))
    assert jnp.allclose(matrix, expected, rtol=1e-3)


def test_build_matrix_warns_and_counts_flat_directions(caplog):
    """Flat directions promote the single log record to WARNING."""
    chi2 = Chi2(lambda c: c[0] ** 2, ["OpA", "OpB"], num_data=1)
    with caplog.at_level(logging.INFO, logger="smefit.whitening"):
        _build_matrix(chi2, _WHITENING, jnp.zeros(2))

    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "1/2 below eps=1.0e-08" in record.getMessage()


def test_build_matrix_logs_spectrum_at_info_when_well_conditioned(caplog):
    """Nothing flat -> one INFO record, no warning."""
    chi2 = Chi2(lambda c: c[0] ** 2 + 0.5 * c[1] ** 2, ["OpA", "OpB"], num_data=1)
    with caplog.at_level(logging.INFO, logger="smefit.whitening"):
        _build_matrix(chi2, _WHITENING, jnp.zeros(2))

    (record,) = caplog.records
    assert record.levelno == logging.INFO
    # Eigenvalues 1 and 2, so cond(H + eps*I) = 2.
    assert "eigenvalues of H in [1.000e+00, 2.000e+00]" in record.getMessage()
    assert "cond(H + eps*I) = 2.000e+00" in record.getMessage()


def test_build_matrix_tolerates_roundoff_negative_eigenvalue():
    """A flat direction is only zero to within the AD Hessian's noise floor,
    which scales with the largest eigenvalue. An absolute threshold would reject
    perfectly usable fits — this is the L0-projection case, where the SM point
    is the exact minimum and the smallest eigenvalue lands on either side of
    zero by luck.
    """
    big = 1e8
    # curvature big in one direction, -1.0 in the other: |−1| is far below the
    # 1e-8 * big = 1.0 noise floor, so it must be treated as flat, not as a saddle.
    chi2 = Chi2(lambda c: big * c[0] ** 2 - 0.4 * c[1] ** 2, ["OpA", "OpB"], num_data=1)
    matrix = _build_matrix(
        chi2, {"sigma_prior": 5.0, "eps": 1.0, "shift": "baseline"}, jnp.zeros(2)
    )
    assert bool(jnp.all(jnp.isfinite(matrix)))


def test_baseline_shift_evaluates_hessian_at_default_zero_baseline():
    """With no baseline set, chi2.baseline defaults to zeros."""
    chi2 = _quadratic_chi2()
    transform = _whitening_baseline_shift(chi2, _WHITENING)
    assert jnp.allclose(transform.shift, jnp.zeros(2))
    assert jnp.allclose(transform.matrix, jnp.eye(2) / jnp.sqrt(2.0), atol=1e-5)


def test_baseline_shift_evaluates_hessian_at_coefficients_baseline():
    """shift should track chi2.baseline, not a hardcoded zero vector."""
    baseline = jnp.array([1.0, -2.0])
    chi2 = Chi2(lambda c: jnp.sum(c**2), ["OpA", "OpB"], num_data=1, baseline=baseline)
    transform = _whitening_baseline_shift(chi2, _WHITENING)
    assert jnp.allclose(transform.shift, baseline)
    # Hessian of sum(c**2) is constant (2*I) everywhere, so the matrix is unchanged.
    assert jnp.allclose(transform.matrix, jnp.eye(2) / jnp.sqrt(2.0), atol=1e-5)


def test_gradient_descent_shift_centers_on_given_point():
    chi2 = _quadratic_chi2()
    gd_best_fit = jnp.array([1.0, -2.0])
    transform = _whitening_gradient_descent_shift(chi2, gd_best_fit, _WHITENING)
    assert jnp.allclose(transform.shift, gd_best_fit)
    # Hessian of sum(c**2) is constant (2*I), so matrix agrees with the baseline case.
    assert jnp.allclose(transform.matrix, jnp.eye(2) / jnp.sqrt(2.0), atol=1e-5)


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
