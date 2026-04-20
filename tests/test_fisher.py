"""Unit tests for smefit.fisher — fisher_information_matrices function."""

import jax.numpy as jnp
import pandas as pd

from smefit.chi2 import Chi2
from smefit.core import (
    Coefficient,
    CoefficientGroup,
    DataGroup,
    Dataset,
    Theory,
    TheoryGroup,
)
from smefit.fisher import (
    constraining_power_matrix,
    fisher_information_matrices,
)
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


def test_fisher_returns_dict():
    """Return type is dict[str, pd.DataFrame] with correct keys and index."""
    theory = _make_theory("DS_A", sm=[10.0, 20.0], lin_op=[1.0, 2.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [10.0, 20.0])])
    fit_covmat = jnp.eye(2)

    result = fisher_information_matrices(model, data, fit_covmat)

    assert isinstance(result, dict)
    assert list(result.keys()) == ["DS_A"]
    assert result["DS_A"].index.tolist() == ["OpA"]
    assert result["DS_A"].columns.tolist() == ["OpA"]


def test_fisher_single_dataset_shape():
    """Single dataset: one entry, matrix shape (n_free, n_free)."""
    theory = _make_theory("DS_A", sm=[10.0, 20.0, 30.0], lin_op=[1.0, 2.0, 3.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [10.0, 20.0, 30.0])])
    fit_covmat = jnp.eye(3)

    result = fisher_information_matrices(model, data, fit_covmat)

    assert list(result.keys()) == ["DS_A"]
    assert result["DS_A"].shape == (1, 1)


def test_fisher_two_datasets_shape():
    """Two datasets: two entries, each (n_free, n_free)."""
    theory_a = _make_theory("DS_A", sm=[10.0, 20.0], lin_op=[1.0, 2.0])
    theory_b = _make_theory("DS_B", sm=[30.0, 40.0, 50.0], lin_op=[3.0, 4.0, 5.0])
    theory_group = TheoryGroup([theory_a, theory_b])

    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_group, cg, use_quad=False)
    data = DataGroup(
        [
            _make_dataset("DS_A", [10.0, 20.0]),
            _make_dataset("DS_B", [30.0, 40.0, 50.0]),
        ]
    )
    fit_covmat = jnp.eye(5)

    result = fisher_information_matrices(model, data, fit_covmat)

    assert list(result.keys()) == ["DS_A", "DS_B"]
    assert result["DS_A"].shape == (1, 1)
    assert result["DS_B"].shape == (1, 1)


def test_fisher_two_free_coefficients_shape():
    """Two free coefficients: index/columns have both names, shape is (2, 2)."""
    theory = Theory(
        name="DS_A",
        order="LO",
        sm_pred=jnp.array([10.0, 20.0, 30.0]),
        eft_pred={"OpA": [1.0, 2.0, 3.0], "OpB": [0.5, 1.0, 1.5]},
        sm_covmat=jnp.eye(3),
        scales=jnp.full(3, 100.0),
        operators=["OpA", "OpB"],
    )
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [10.0, 20.0, 30.0])])
    fit_covmat = jnp.eye(3)

    result = fisher_information_matrices(model, data, fit_covmat)

    assert result["DS_A"].index.tolist() == ["OpA", "OpB"]
    assert result["DS_A"].shape == (2, 2)


def test_fisher_known_value():
    """Hand-computed case.

    sm=[0,0], lin_OpA=[1,0], fit_covmat=I
    J = [[1],[0]], F = J^T @ I @ J = [[1]]
    """
    theory = _make_theory("DS_A", sm=[0.0, 0.0], lin_op=[1.0, 0.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [0.0, 0.0])])
    fit_covmat = jnp.eye(2)

    result = fisher_information_matrices(model, data, fit_covmat)

    assert jnp.allclose(jnp.array(result["DS_A"].values), jnp.array([[1.0]]), atol=1e-5)


def test_fisher_ordering_matches_data_names():
    """Matrices are returned in data.names order (alphabetical)."""
    theory_b = _make_theory("DS_B", sm=[10.0, 20.0], lin_op=[2.0, 4.0])
    theory_c = _make_theory("DS_C", sm=[30.0], lin_op=[3.0])
    theory_group = TheoryGroup([theory_b, theory_c])

    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_group, cg, use_quad=False)
    data = DataGroup(
        [
            _make_dataset("DS_B", [10.0, 20.0]),
            _make_dataset("DS_C", [30.0]),
        ]
    )
    fit_covmat = jnp.eye(3)

    result = fisher_information_matrices(model, data, fit_covmat)

    assert list(result.keys()) == ["DS_B", "DS_C"]
    # DS_B: J=[2,4]^T, F = 2^2 + 4^2 = 20
    # DS_C: J=[3]^T,   F = 9
    assert jnp.allclose(
        jnp.array(result["DS_B"].values), jnp.array([[20.0]]), atol=1e-5
    )
    assert jnp.allclose(jnp.array(result["DS_C"].values), jnp.array([[9.0]]), atol=1e-5)


def test_fisher_symmetry():
    """Fisher matrix must be symmetric."""
    theory = Theory(
        name="DS_A",
        order="LO",
        sm_pred=jnp.array([10.0, 20.0, 30.0]),
        eft_pred={"OpA": [1.0, 2.0, 3.0], "OpB": [0.5, 1.5, 2.5]},
        sm_covmat=jnp.eye(3),
        scales=jnp.full(3, 100.0),
        operators=["OpA", "OpB"],
    )
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [10.0, 20.0, 30.0])])
    fit_covmat = jnp.array(
        [
            [2.0, 0.5, 0.0],
            [0.5, 3.0, 0.1],
            [0.0, 0.1, 1.5],
        ]
    )

    result = fisher_information_matrices(model, data, fit_covmat)

    F = jnp.array(result["DS_A"].values)
    assert jnp.allclose(F, F.T, atol=1e-5)


# ---------------------------------------------------------------------------
# External chi2 tests
# ---------------------------------------------------------------------------


def _make_ext_chi2(name, n_free, lin):
    """Gaussian external chi2: chi2(c) = sum_i (lin_i * c_0)^2.

    Fisher = 0.5 * H = 0.5 * 2 * sum(lin_i^2) = sum(lin_i^2).
    """
    lin_arr = jnp.array(lin)

    def fn(coeffs):
        residuals = lin_arr * coeffs[0]
        return jnp.dot(residuals, residuals)

    return Chi2(fn, param_names=[f"OpA"] * n_free, num_data=len(lin), name=name)


def test_fisher_with_external_chi2_appended():
    """External chi2 entries appear after dataset entries."""
    theory = _make_theory("DS_A", sm=[10.0], lin_op=[1.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [10.0])])
    fit_covmat = jnp.eye(1)
    ext = _make_ext_chi2("MyExtChi2", n_free=1, lin=[2.0, 3.0])

    result = fisher_information_matrices(model, data, fit_covmat, ext_chi2_func=[ext])

    assert list(result.keys()) == ["DS_A", "MyExtChi2"]
    assert result["MyExtChi2"].shape == (1, 1)


def test_fisher_external_chi2_known_value():
    """Fisher from external chi2: chi2(c) = (2c)^2 + (3c)^2, F = 0.5*H = 13."""
    theory = _make_theory("DS_A", sm=[0.0], lin_op=[0.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [0.0])])
    fit_covmat = jnp.eye(1)
    ext = _make_ext_chi2("Ext", n_free=1, lin=[2.0, 3.0])  # F = 4 + 9 = 13

    result = fisher_information_matrices(model, data, fit_covmat, ext_chi2_func=[ext])

    assert jnp.allclose(jnp.array(result["Ext"].values), jnp.array([[13.0]]), atol=1e-5)


def test_fisher_external_chi2_fallback_name():
    """Chi2 with name=None gets a fallback name 'ext_chi2_0'."""
    theory = _make_theory("DS_A", sm=[0.0], lin_op=[0.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [0.0])])
    fit_covmat = jnp.eye(1)

    fn = lambda c: jnp.sum(c**2)
    ext = Chi2(fn, param_names=["OpA"], num_data=1, name=None)

    result = fisher_information_matrices(model, data, fit_covmat, ext_chi2_func=[ext])

    assert "ext_chi2_0" in result


def test_fisher_no_external_chi2():
    """ext_chi2_func=None (default) returns only dataset entries."""
    theory = _make_theory("DS_A", sm=[10.0], lin_op=[1.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [10.0])])
    fit_covmat = jnp.eye(1)

    result = fisher_information_matrices(model, data, fit_covmat)

    assert list(result.keys()) == ["DS_A"]


# ---------------------------------------------------------------------------
# ConstrainingPowerMatrix tests
# ---------------------------------------------------------------------------


def _fisher_two_sources():
    """Helper: two datasets, one free coefficient, identity covmat."""
    theory_a = _make_theory("DS_A", sm=[0.0, 0.0], lin_op=[2.0, 0.0])
    theory_b = _make_theory("DS_B", sm=[0.0], lin_op=[1.0])
    from smefit.core import TheoryGroup

    theory_group = TheoryGroup([theory_a, theory_b])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_group, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [0.0, 0.0]), _make_dataset("DS_B", [0.0])])
    fit_covmat = jnp.eye(3)
    return fisher_information_matrices(model, data, fit_covmat)


def test_constraining_power_matrix_type():
    """Result is a DataFrame with correct index and columns."""
    fim = _fisher_two_sources()
    result = constraining_power_matrix(fim)

    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist() == ["OpA"]
    assert result.columns.tolist() == ["DS_A", "DS_B"]


def test_constraining_power_matrix_shape():
    """alpha has shape (n_ops, n_sources)."""
    fim = _fisher_two_sources()
    result = constraining_power_matrix(fim)

    assert result.shape == (1, 2)


def test_constraining_power_matrix_rows_sum_to_one():
    """Each row of alpha sums to 1."""
    fim = _fisher_two_sources()
    result = constraining_power_matrix(fim)

    assert jnp.allclose(jnp.array(result.sum(axis=1).values), jnp.ones(1), atol=1e-5)


def test_constraining_power_matrix_known_values():
    """Hand-computed case with one coefficient.

    DS_A: lin=[2,0] → F_A = 4, DS_B: lin=[1] → F_B = 1
    F_total = 5, Sigma = 1/5
    alpha_A = (Sigma @ F_A @ Sigma) / Sigma = F_A / F_total = 4/5
    alpha_B = F_B / F_total = 1/5
    """
    fim = _fisher_two_sources()
    result = constraining_power_matrix(fim)

    assert jnp.allclose(jnp.array(result.iloc[0, 0]), 4.0 / 5.0, atol=1e-5)
    assert jnp.allclose(jnp.array(result.iloc[0, 1]), 1.0 / 5.0, atol=1e-5)
