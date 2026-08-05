"""Unit tests for smefit.hessian_fit — hessian_fit function."""

import jax.numpy as jnp
import optax
import pytest

from smefit.chi2 import Chi2, build_chi2
from smefit.core import (
    Coefficient,
    CoefficientGroup,
    DataGroup,
    Dataset,
    Theory,
)
from smefit.fit_result import Fit
from smefit.gradient_descent import gd_best_fit
from smefit.hessian_fit import hessian_fit
from smefit.model import EFTModel

_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}

_GD_SETTINGS_SM = {"sm_solution": True, "n_steps": 100, "tol": 1e-8}
_GD_SETTINGS_GD = {"sm_solution": False, "n_steps": 5000, "tol": 1e-6}
_HESSIAN_SETTINGS = {"n_samples": 50, "seed": 0}


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


def _make_setup(sm, data_cv, lin_op):
    """Build (eft_model, chi2) for a 1-free-param linear problem."""
    theory = _make_theory("DS_A", sm=sm, lin_op=lin_op)
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", data_cv)])
    fit_covmat = jnp.eye(len(sm))
    chi2 = Chi2(build_chi2(model, data, fit_covmat), ["OpA"], num_data=len(sm))
    return model, chi2


_OPTIMIZER = optax.adam(1e-2)


# ---------------------------------------------------------------------------
# gd_best_fit — SM solution (sm_solution=True)
# ---------------------------------------------------------------------------


def test_gd_best_fit_sm_returns_zeros():
    """SM solution must return the zero vector regardless of the data."""
    _, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[2.0, 0.0], lin_op=[1.0, 0.0])
    c = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    assert jnp.allclose(c, jnp.zeros(1))


def test_gd_best_fit_sm_shape():
    _, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[0.0, 0.0], lin_op=[1.0, 0.0])
    c = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    assert c.shape == (1,)


# ---------------------------------------------------------------------------
# gd_best_fit — gradient descent (sm_solution=False)
# ---------------------------------------------------------------------------


def test_gd_best_fit_gd_finds_known_minimum():
    """Gradient descent should converge to c_best ≈ 2.0.

    Setup: sm=[0,0], lin=[1,0], data=[2,0], covmat=I
    Analytic minimum: c = 2.0
    """
    _, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[2.0, 0.0], lin_op=[1.0, 0.0])
    c = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_GD)
    assert float(c[0]) == pytest.approx(2.0, abs=1e-2)


def test_gd_best_fit_gd_lower_chi2_than_sm():
    """GD best-fit should have lower chi2 than the SM point when data != SM."""
    _, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[2.0, 0.0], lin_op=[1.0, 0.0])
    c_gd = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_GD)
    c_sm = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    assert float(chi2(c_gd)) < float(chi2(c_sm))


# ---------------------------------------------------------------------------
# hessian_fit — SM solution (sm_solution=True)
# ---------------------------------------------------------------------------


def test_hessian_fit_sm_zero_residual():
    """When data == SM predictions the minimum is at c=0 — SM solution should recover it."""
    model, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[0.0, 0.0], lin_op=[1.0, 0.0])
    c_best = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    result = hessian_fit(model, chi2, c_best, _HESSIAN_SETTINGS)
    assert result.best_fit_point["OpA"] == pytest.approx(0.0, abs=1e-8)


def test_hessian_fit_sm_solution_chi2():
    """With data=[0,0] and SM=[0,0], chi2 at c=0 should be 0."""
    model, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[0.0, 0.0], lin_op=[1.0, 0.0])
    c_best = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    result = hessian_fit(model, chi2, c_best, _HESSIAN_SETTINGS)
    assert result.chi2_val == pytest.approx(0.0, abs=1e-6)


def test_hessian_fit_sm_solution_ignores_true_minimum():
    """With data != SM, the SM solution pins the best-fit to c=0 regardless."""
    model, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[2.0, 0.0], lin_op=[1.0, 0.0])
    c_best = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    result = hessian_fit(model, chi2, c_best, _HESSIAN_SETTINGS)
    assert result.best_fit_point["OpA"] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# hessian_fit — gradient descent (sm_solution=False)
# ---------------------------------------------------------------------------


def test_hessian_fit_gd_finds_known_minimum():
    """Gradient descent should converge to c_best ≈ 2.0.

    Setup: sm=[0,0], lin=[1,0], data=[2,0], covmat=I
    Analytic minimum: c = 2.0
    """
    model, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[2.0, 0.0], lin_op=[1.0, 0.0])
    c_best = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_GD)
    result = hessian_fit(model, chi2, c_best, _HESSIAN_SETTINGS)
    assert result.best_fit_point["OpA"] == pytest.approx(2.0, abs=1e-2)


def test_hessian_fit_gd_lower_chi2_than_sm():
    """Gradient-descent best fit should have lower chi2 than the SM point."""
    model, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[2.0, 0.0], lin_op=[1.0, 0.0])
    c_gd = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_GD)
    c_sm = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    result_gd = hessian_fit(model, chi2, c_gd, _HESSIAN_SETTINGS)
    result_sm = hessian_fit(model, chi2, c_sm, _HESSIAN_SETTINGS)
    assert result_gd.chi2_val < result_sm.chi2_val


# ---------------------------------------------------------------------------
# Fit structure
# ---------------------------------------------------------------------------


def test_hessian_fit_returns_fit_result():
    model, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[1.0, 0.0], lin_op=[1.0, 0.0])
    c_best = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    result = hessian_fit(model, chi2, c_best, _HESSIAN_SETTINGS)
    assert isinstance(result, Fit)
    assert result.fit_type == "hessian"
    assert result.fit_name == "hessian"


def test_hessian_fit_sample_shape():
    model, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[1.0, 0.0], lin_op=[1.0, 0.0])
    c_best = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    result = hessian_fit(model, chi2, c_best, _HESSIAN_SETTINGS)
    assert result.samples is not None
    assert result.samples["OpA"].shape == (50,)


def test_hessian_fit_free_parameters():
    model, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[1.0, 0.0], lin_op=[1.0, 0.0])
    c_best = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    result = hessian_fit(model, chi2, c_best, _HESSIAN_SETTINGS)
    assert result.free_parameters == ["OpA"]


def test_hessian_fit_num_data():
    model, chi2 = _make_setup(sm=[0.0, 0.0], data_cv=[1.0, 0.0], lin_op=[1.0, 0.0])
    c_best = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    result = hessian_fit(model, chi2, c_best, _HESSIAN_SETTINGS)
    assert result.num_data == 2


# ---------------------------------------------------------------------------
# Derived coefficients
# ---------------------------------------------------------------------------


def test_hessian_fit_resolves_derived_coefficients():
    """Derived coefficients (expr) must appear in best_fit_point and samples."""
    theory = _make_theory("DS_A", sm=[0.0, 0.0], lin_op=[1.0, 0.0])
    cg = CoefficientGroup(
        [
            _free("OpA"),
            Coefficient(name="OpB", free=False, vars=["OpA"], expr="OpA**2"),
        ]
    )
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [0.0, 0.0])])
    fit_covmat = jnp.eye(2)
    chi2 = Chi2(build_chi2(model, data, fit_covmat), ["OpA"], num_data=2)

    c_best = gd_best_fit(chi2, _OPTIMIZER, _GD_SETTINGS_SM)
    result = hessian_fit(model, chi2, c_best, _HESSIAN_SETTINGS)

    assert "OpA" in result.best_fit_point
    assert "OpB" in result.best_fit_point
    assert "OpA" in result.samples
    assert "OpB" in result.samples
    # At c=0: OpB = OpA**2 = 0
    assert result.best_fit_point["OpB"] == pytest.approx(0.0, abs=1e-8)
