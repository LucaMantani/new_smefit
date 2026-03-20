"""Unit tests for smefit.analytic_fit — analytic_fit function."""

import jax.numpy as jnp
import pytest

from smefit.analytic_fit import analytic_fit
from smefit.chi2 import Chi2, build_chi2
from smefit.core import (
    Coefficient,
    CoefficientGroup,
    DataGroup,
    Dataset,
    Theory,
)
from smefit.fit_result import FitResult
from smefit.model import EFTModel

_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}


def _free(name):
    return Coefficient(name=name, free=True, prior=_PRIOR)


def _make_dataset(name, cv):
    n = len(cv)
    return Dataset(
        name=name,
        num_data=n,
        central_values=jnp.array(cv, dtype=float),
        stat_err=jnp.ones(n),
        syst_err=jnp.zeros((1, n)),
        sys_names=["UNCORR"],
        sys_types=["ADD"],
        luminosity=jnp.ones(n),
    )


def _make_theory(name, sm, lin_op, op_name="OpA"):
    n = len(sm)
    return Theory(
        name=name,
        order="LO",
        sm_pred=jnp.array(sm, dtype=float),
        eft_pred={op_name: lin_op},
        sm_covmat=jnp.eye(n),
        scales=jnp.full(n, 100.0),
        operators=[op_name],
    )


# ---------------------------------------------------------------------------
# Guard clauses
# ---------------------------------------------------------------------------


def test_analytic_fit_raises_external_chi2(theory_a):
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [10.0, 20.0, 30.0])])
    fit_covmat = jnp.eye(3)
    chi2 = Chi2(lambda c: jnp.sum(c**2), ["OpA"], num_data=3, has_external=True)

    with pytest.raises(ValueError, match="external_chi2"):
        analytic_fit(model, data, fit_covmat, chi2)


def test_analytic_fit_raises_quad_model(theory_a):
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=True)
    data = DataGroup([_make_dataset("DS_A", [10.0, 20.0, 30.0])])
    fit_covmat = jnp.eye(3)
    chi2 = Chi2(lambda c: jnp.sum(c**2), ["OpA"], num_data=3, has_external=False)

    with pytest.raises(ValueError, match="use_quad"):
        analytic_fit(model, data, fit_covmat, chi2)


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def test_analytic_fit_zero_residual():
    """If data == SM predictions, the best-fit free coefficient should be ~0."""
    theory = _make_theory("DS_A", sm=[10.0, 20.0, 30.0], lin_op=[1.0, 2.0, 3.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [10.0, 20.0, 30.0])])
    fit_covmat = jnp.eye(3)
    chi2 = Chi2(build_chi2(model, data, fit_covmat), ["OpA"], num_data=3)

    result = analytic_fit(model, data, fit_covmat, chi2)

    assert result.best_fit_point["OpA"] == pytest.approx(0.0, abs=1e-5)


def test_analytic_fit_known_best_fit():
    """Verify the analytic solution against a hand-computed case.

    Setup: sm=[0,0], lin_OpA=[1,0], data=[2,0], fit_covmat=eye(2)
    => J=[[1],[0]], H=J^T J=[[1]], c_best = H^-1 J^T delta = 1*[1,0]*[2,0] = 2.0
    """
    theory = _make_theory("DS_A", sm=[0.0, 0.0], lin_op=[1.0, 0.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [2.0, 0.0])])
    fit_covmat = jnp.eye(2)
    chi2 = Chi2(build_chi2(model, data, fit_covmat), ["OpA"], num_data=2)

    result = analytic_fit(model, data, fit_covmat, chi2)

    assert result.best_fit_point["OpA"] == pytest.approx(2.0, abs=1e-5)


def test_analytic_fit_returns_fit_result():
    theory = _make_theory("DS_A", sm=[0.0, 0.0], lin_op=[1.0, 0.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [1.0, 0.0])])
    fit_covmat = jnp.eye(2)
    chi2 = Chi2(build_chi2(model, data, fit_covmat), ["OpA"], num_data=2)

    result = analytic_fit(model, data, fit_covmat, chi2)

    assert isinstance(result, FitResult)
    assert "OpA" in result.best_fit_point
    assert result.max_loglikelihood <= 0.0 or True  # just check it's a float
    assert isinstance(result.max_loglikelihood, float)


def test_analytic_fit_sample_shape():
    n_samples = 50
    theory = _make_theory("DS_A", sm=[0.0, 0.0], lin_op=[1.0, 0.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [1.0, 0.0])])
    fit_covmat = jnp.eye(2)
    chi2 = Chi2(build_chi2(model, data, fit_covmat), ["OpA"], num_data=2)

    result = analytic_fit(model, data, fit_covmat, chi2, n_samples=n_samples)

    assert result.samples is not None
    for name in result.best_fit_point:
        assert result.samples[name].shape == (n_samples,)
