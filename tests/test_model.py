"""Unit tests for smefit.model — EFTModel.forward_map."""

import logging

import jax.numpy as jnp

from smefit.core import Coefficient, CoefficientGroup, TheoryGroup
from smefit.model import EFTModel

_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}


def _free(name):
    return Coefficient(name=name, free=True, prior=_PRIOR)


# ---------------------------------------------------------------------------
# forward_map
# ---------------------------------------------------------------------------


def test_forward_map_zero_coeffs_is_sm(theory_a):
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=False)
    result = model.forward_map(jnp.array([0.0]))
    assert jnp.allclose(result, theory_a.sm_pred)


def test_forward_map_linear(theory_a):
    # Only OpA declared; lin_corr[:, OpA] = [1, 2, 3]
    # coeffs = [2.0] → predictions = [10+2, 20+4, 30+6] = [12, 24, 36]
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=False)
    result = model.forward_map(jnp.array([2.0]))
    assert jnp.allclose(result, jnp.array([12.0, 24.0, 36.0]))


def test_forward_map_quadratic(theory_a):
    # Only OpA declared; OpA*OpA = [0.1, 0.2, 0.3]
    # coeffs = [2.0]:
    #   linear  = sm + [1,2,3]*2 = [12, 24, 36]
    #   quad    = [0.1, 0.2, 0.3]*2^2 = [0.4, 0.8, 1.2]
    #   total   = [12.4, 24.8, 37.2]
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=True)
    result = model.forward_map(jnp.array([2.0]))
    assert jnp.allclose(result, jnp.array([12.4, 24.8, 37.2]))


def test_forward_map_no_match_warns(theory_a, caplog):
    group = TheoryGroup([theory_a])
    cg = CoefficientGroup([_free("OpX")])  # OpX not in theory_a
    with caplog.at_level(logging.WARNING, logger="smefit.model"):
        model = EFTModel(group, cg, use_quad=False)
    assert "None of the declared coefficients enter the predictions" in caplog.text
    assert "DS_A" in caplog.text
    # the "inactive coefficients" warning would be redundant here
    assert "not present in any theory dataset" not in caplog.text
    # the coefficient simply does not enter: predictions stay at the SM
    assert jnp.allclose(model.forward_map(jnp.array([2.0])), group.sm_pred)


def test_forward_map_subset_operators(theory_a):
    # theory_a has OpA, OpB; only OpA declared → OpB contribution silently ignored
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=False)
    result = model.forward_map(jnp.array([0.0]))
    assert jnp.allclose(result, theory_a.sm_pred)
