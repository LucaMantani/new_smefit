"""Unit tests for smefit.core — Dataset, Theory, Coefficient, *Group classes."""

import jax.numpy as jnp
import numpy as np
import pytest

from smefit.core import (
    Coefficient,
    CoefficientGroup,
    DataGroup,
    Dataset,
    Theory,
    TheoryGroup,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_syst_err_mult(dataset_a):
    mult = dataset_a.syst_err_mult
    expected = dataset_a.syst_err / dataset_a.central_values[None, :]
    assert jnp.allclose(mult, expected)


# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------


def test_theory_eft_lin_pred_shape(theory_a):
    # 3 data points, 2 operators
    assert theory_a.eft_lin_pred.shape == (3, 2)


def test_theory_eft_quad_pred_shape(theory_a):
    # 3 data points, 2 operators → upper-triangular (2, 2) per data point
    assert theory_a.eft_quad_pred.shape == (3, 2, 2)
    # Lower triangle must be zero (upper-triangular convention)
    assert jnp.all(theory_a.eft_quad_pred[:, 1, 0] == 0)


def test_theory_operators_sorted():
    th = Theory(
        name="X",
        order="NLO",
        sm_pred=jnp.array([1.0, 2.0]),
        eft_pred={"OpB": [0.5, 1.0], "OpA": [1.0, 2.0]},
        sm_covmat=jnp.eye(2),
        scales=jnp.array([100.0, 100.0]),
        operators=["OpB", "OpA"],  # intentionally unsorted
    )
    assert th.operators == ["OpA", "OpB"]


def test_theory_missing_linear_op():
    """Operator present only in quadratic → zero column in eft_lin_pred."""
    th = Theory(
        name="X",
        order="NLO",
        sm_pred=jnp.array([1.0, 2.0]),
        eft_pred={"OpA*OpB": [0.5, 1.0]},  # no linear keys at all
        sm_covmat=jnp.eye(2),
        scales=jnp.array([100.0, 100.0]),
        operators=["OpA", "OpB"],
    )
    # Both columns should be all-zero because neither OpA nor OpB has a linear entry
    assert jnp.all(th.eft_lin_pred == 0)


# ---------------------------------------------------------------------------
# Coefficient validation
# ---------------------------------------------------------------------------


def test_coefficient_free_validation():
    with pytest.raises(ValueError):
        Coefficient(name="X", free=True, value=1.0)
    with pytest.raises(ValueError):
        Coefficient(name="X", free=True, expr="y**2")


def test_coefficient_fixed_validation():
    # Neither value nor expr → error
    with pytest.raises(ValueError):
        Coefficient(name="X", free=False)
    # Both value and expr → error
    with pytest.raises(ValueError):
        Coefficient(name="X", free=False, value=1.0, expr="y**2", vars=["y"])


def test_coefficient_expr_validation():
    # expr without vars → error
    with pytest.raises(ValueError):
        Coefficient(name="X", free=False, expr="y**2")


def test_coefficient_constrain_fixed():
    c = Coefficient(name="X", free=False, value=3.14)
    assert c.constrain() == pytest.approx(3.14)
    assert c.constrain(1.0, 2.0) == pytest.approx(3.14)  # args ignored


def test_coefficient_constrain_expr():
    c = Coefficient(name="X", free=False, vars=["y"], expr="y**2 + 1.0")
    assert float(c.constrain(2.0)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# DataGroup
# ---------------------------------------------------------------------------


def test_data_group_shapes(dataset_a, dataset_b):
    dg = DataGroup([dataset_a, dataset_b])
    N = 6  # 3 + 3
    assert dg.exp_covmat.shape == (N, N)
    assert len(dg.cv) == N
    assert dg.num_data == N


def test_data_group_sorted_by_name(dataset_a, dataset_b):
    # Pass in reverse order; DataGroup must sort alphabetically
    dg = DataGroup([dataset_b, dataset_a])
    assert dg.names == ["DS_A", "DS_B"]


# ---------------------------------------------------------------------------
# TheoryGroup
# ---------------------------------------------------------------------------


def test_theory_group_operator_union(theory_a, theory_b):
    # theory_a: [OpA, OpB], theory_b: [OpA, OpC] → union = [OpA, OpB, OpC]
    tg = TheoryGroup([theory_a, theory_b])
    assert tg.operators == ["OpA", "OpB", "OpC"]


def test_theory_group_eft_lin_pred_shape(theory_a, theory_b):
    tg = TheoryGroup([theory_a, theory_b])
    assert tg.eft_lin_pred.shape == (6, 3)  # 3+3 data, 3 ops in union


def test_theory_group_sm_covmat_block_diag(theory_a, theory_b):
    tg = TheoryGroup([theory_a, theory_b])
    # Both off-diagonal 3×3 blocks must be zero
    assert jnp.allclose(tg.sm_covmat[0:3, 3:6], jnp.zeros((3, 3)))
    assert jnp.allclose(tg.sm_covmat[3:6, 0:3], jnp.zeros((3, 3)))


# ---------------------------------------------------------------------------
# CoefficientGroup
# ---------------------------------------------------------------------------


def test_coeff_group_resolve_free(coeff_free, coeff_fixed):
    # sorted: OpA (0, free), OpB (1, fixed=2.0)
    cg = CoefficientGroup([coeff_free, coeff_fixed])
    result = cg.resolve(jnp.array([5.0]))
    assert float(result[0]) == pytest.approx(5.0)
    assert float(result[1]) == pytest.approx(2.0)


def test_coeff_group_resolve_fixed(coeff_free, coeff_fixed):
    # Fixed coefficient retains its value regardless of the free slot's input
    cg = CoefficientGroup(
        [coeff_free, coeff_fixed]
    )  # sorted: OpA (0, free), OpB (1, fixed=2.0)
    result = cg.resolve(jnp.array([0.0]))
    assert float(result[1]) == pytest.approx(2.0)
    result2 = cg.resolve(jnp.array([99.0]))
    assert float(result2[1]) == pytest.approx(2.0)  # unchanged regardless of free value


def test_coeff_group_resolve_expr(coeff_free, coeff_expr):
    # sorted: OpA (0, free), OpC (1, expr=OpA**2)
    cg = CoefficientGroup([coeff_free, coeff_expr])
    result = cg.resolve(jnp.array([3.0]))
    assert float(result[0]) == pytest.approx(3.0)
    assert float(result[1]) == pytest.approx(9.0)


def test_coeff_group_whitened_resolve():
    c1 = Coefficient(
        name="OpA", free=True, prior={"dist": "uniform", "low": -1.0, "high": 1.0}
    )
    c2 = Coefficient(
        name="OpB", free=True, prior={"dist": "uniform", "low": -1.0, "high": 1.0}
    )
    cg = CoefficientGroup([c1, c2])
    # sorted: OpA (0), OpB (1)
    W = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    wcg = cg.whitened(W)
    result = wcg.resolve(jnp.array([1.0, 1.0]))
    # un-whitened: W @ [1, 1] = [2, 3]; both free → result = [2, 3]
    assert jnp.allclose(result, jnp.array([2.0, 3.0]))


def test_coeff_group_single_free():
    c1 = Coefficient(
        name="OpA", free=True, prior={"dist": "uniform", "low": -1.0, "high": 1.0}
    )
    c2 = Coefficient(
        name="OpB", free=True, prior={"dist": "uniform", "low": -1.0, "high": 1.0}
    )
    cg = CoefficientGroup([c1, c2])
    sg = cg.single_free("OpA")
    assert sg.free_names == ["OpA"]
    # OpB is now fixed at 0; sorted: OpA (0), OpB (1)
    result = sg.resolve(jnp.array([5.0]))
    assert jnp.allclose(result, jnp.array([5.0, 0.0]))


def test_coeff_group_resolve_all_fixed_empty_array():
    """resolve(jnp.array([])) must not raise a dtype error (empty int index)."""
    c1 = Coefficient(name="OpA", free=False, value=1.0)
    c2 = Coefficient(name="OpB", free=False, value=2.0)
    cg = CoefficientGroup([c1, c2])
    result = cg.resolve(jnp.array([]))
    assert float(result[cg.coeff_index["OpA"]]) == pytest.approx(1.0)
    assert float(result[cg.coeff_index["OpB"]]) == pytest.approx(2.0)


def test_coeff_group_expr_var_fixed_value_allowed():
    """An expression coefficient may reference a fixed-value coefficient."""
    c_fixed = Coefficient(name="OpWB", free=False, value=0.1)
    c_expr = Coefficient(name="OpD", free=False, vars=["OpWB"], expr="0.5*OpWB**2")
    cg = CoefficientGroup([c_fixed, c_expr])
    result = cg.resolve(jnp.array([]))
    assert float(result[cg.coeff_index["OpWB"]]) == pytest.approx(0.1)
    assert float(result[cg.coeff_index["OpD"]]) == pytest.approx(0.5 * 0.1**2)


def test_coeff_group_expr_var_referencing_expr_coeff_rejected():
    """An expression coefficient may NOT reference another expression coefficient."""
    c_val = Coefficient(name="A", free=False, value=1.0)
    c_expr1 = Coefficient(name="B", free=False, vars=["A"], expr="2*A")
    c_expr2 = Coefficient(name="C", free=False, vars=["B"], expr="3*B")
    with pytest.raises(
        ValueError, match="must be a free coefficient or a fixed coefficient"
    ):
        CoefficientGroup([c_val, c_expr1, c_expr2])


# ---------------------------------------------------------------------------
# resolve_slim / slim_names
# ---------------------------------------------------------------------------


def test_resolve_slim_free_only():
    """slim_names == free_names when there are no expression coefficients."""
    c1 = Coefficient(
        name="OpA", free=True, prior={"dist": "uniform", "low": -1.0, "high": 1.0}
    )
    c2 = Coefficient(
        name="OpB", free=True, prior={"dist": "uniform", "low": -1.0, "high": 1.0}
    )
    cg = CoefficientGroup([c1, c2])
    assert set(cg.slim_names) == {"OpA", "OpB"}
    slim = cg.resolve_slim(jnp.array([1.0, 2.0]))
    assert slim.shape == (2,)
    # Values must match the free coefficient values (sorted: OpA=1, OpB=2)
    full = cg.resolve(jnp.array([1.0, 2.0]))
    for i, name in enumerate(cg.slim_names):
        assert float(slim[i]) == pytest.approx(float(full[cg.coeff_index[name]]))


def test_resolve_slim_skips_fixed():
    """Fixed-to-constant coefficients must not appear in slim_names."""
    c_free = Coefficient(
        name="OpA", free=True, prior={"dist": "uniform", "low": -1.0, "high": 1.0}
    )
    c_fixed = Coefficient(name="OpFixed", free=False, value=3.0)
    cg = CoefficientGroup([c_free, c_fixed])
    assert "OpA" in cg.slim_names
    assert "OpFixed" not in cg.slim_names
    slim = cg.resolve_slim(jnp.array([5.0]))
    assert slim.shape == (1,)
    assert float(slim[0]) == pytest.approx(5.0)


def test_resolve_slim_includes_dependent_expr():
    """Expression-constrained coefficients that depend on a free param are included."""
    c_free = Coefficient(
        name="OpA", free=True, prior={"dist": "uniform", "low": -1.0, "high": 1.0}
    )
    c_expr = Coefficient(name="OpB", free=False, vars=["OpA"], expr="OpA**2")
    cg = CoefficientGroup([c_free, c_expr])
    assert "OpA" in cg.slim_names
    assert "OpB" in cg.slim_names
    slim = cg.resolve_slim(jnp.array([3.0]))
    # sorted alphabetically: OpA=3.0, OpB=9.0
    slim_dict = {name: float(slim[i]) for i, name in enumerate(cg.slim_names)}
    assert slim_dict["OpA"] == pytest.approx(3.0)
    assert slim_dict["OpB"] == pytest.approx(9.0)


def test_resolve_slim_excludes_independent_expr():
    """Expression-constrained coefficients with only fixed deps are excluded."""
    c_fixed = Coefficient(name="OpA", free=False, value=2.0)
    c_expr = Coefficient(name="OpB", free=False, vars=["OpA"], expr="OpA**2")
    cg = CoefficientGroup([c_fixed, c_expr])
    # No free parameters — slim_names should be empty
    assert cg.slim_names == []
    slim = cg.resolve_slim(jnp.array([]))
    assert slim.shape == (0,)
