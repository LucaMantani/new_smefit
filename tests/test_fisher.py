"""Unit tests for smefit.fisher — fisher_information_matrices function."""

import logging

import jax.numpy as jnp
import pandas as pd

from smefit.chi2 import Chi2, build_datasets_chi2
from smefit.core import (
    Coefficient,
    CoefficientGroup,
    DataGroup,
    Dataset,
    Theory,
    TheoryGroup,
)
from smefit.fisher import (
    _resolve_groups,
    aggregate_fisher_information_matrices,
    fisher_information_matrices,
)
from smefit.model import EFTModel

_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}


def _free(name):
    return Coefficient(name=name, free=True, prior=_PRIOR)


def _sm_point(n):
    """Return a zero (SM) best-fit vector of length n."""
    return jnp.zeros(n)


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

    result = fisher_information_matrices(
        build_datasets_chi2(model, data, fit_covmat), _sm_point(1)
    )

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

    result = fisher_information_matrices(
        build_datasets_chi2(model, data, fit_covmat), _sm_point(1)
    )

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

    result = fisher_information_matrices(
        build_datasets_chi2(model, data, fit_covmat), _sm_point(1)
    )

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

    result = fisher_information_matrices(
        build_datasets_chi2(model, data, fit_covmat), _sm_point(2)
    )

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

    result = fisher_information_matrices(
        build_datasets_chi2(model, data, fit_covmat), _sm_point(1)
    )

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

    result = fisher_information_matrices(
        build_datasets_chi2(model, data, fit_covmat), _sm_point(1)
    )

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

    result = fisher_information_matrices(
        build_datasets_chi2(model, data, fit_covmat), _sm_point(2)
    )

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

    datasets_chi2 = build_datasets_chi2(model, data, fit_covmat) + [ext]
    result = fisher_information_matrices(datasets_chi2, _sm_point(1))

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

    datasets_chi2 = build_datasets_chi2(model, data, fit_covmat) + [ext]
    result = fisher_information_matrices(datasets_chi2, _sm_point(1))

    assert jnp.allclose(jnp.array(result["Ext"].values), jnp.array([[13.0]]), atol=1e-5)


def test_fisher_external_chi2_named():
    """External Chi2 appears in result under its name."""
    theory = _make_theory("DS_A", sm=[0.0], lin_op=[0.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [0.0])])
    fit_covmat = jnp.eye(1)

    fn = lambda c: jnp.sum(c**2)
    ext = Chi2(fn, param_names=["OpA"], num_data=1, name="MyPrior")

    datasets_chi2 = build_datasets_chi2(model, data, fit_covmat) + [ext]
    result = fisher_information_matrices(datasets_chi2, _sm_point(1))

    assert "MyPrior" in result


def test_fisher_no_external_chi2():
    """ext_chi2_func=None (default) returns only dataset entries."""
    theory = _make_theory("DS_A", sm=[10.0], lin_op=[1.0])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup([_make_dataset("DS_A", [10.0])])
    fit_covmat = jnp.eye(1)

    result = fisher_information_matrices(
        build_datasets_chi2(model, data, fit_covmat), _sm_point(1)
    )

    assert list(result.keys()) == ["DS_A"]


# ---------------------------------------------------------------------------
# hessian_fit best-fit-point tests
# ---------------------------------------------------------------------------


def _make_quadratic_chi2(name):
    """chi2(c) = (c[0] + c[0]^2)^2.

    Hessian at c=0: 2  → F = 1.0
    Hessian at c=1: 26 → F = 13.0
    """

    def fn(c):
        return (c[0] + c[0] ** 2) ** 2

    return Chi2(fn, param_names=["OpA"], num_data=1, name=name)


def test_fisher_uses_gd_best_fit_point():
    """Fisher is evaluated at the point given by gd_best_fit.

    chi2(c) = (c[0] + c[0]^2)^2
    Hessian at c=0 → F = 1.0
    Hessian at c=1 → F = 13.0
    """
    chi2 = _make_quadratic_chi2("DS_Q")

    result_sm = fisher_information_matrices([chi2], _sm_point(1))
    result_mle = fisher_information_matrices([chi2], jnp.array([1.0]))

    assert jnp.allclose(
        jnp.array(result_sm["DS_Q"].values), jnp.array([[1.0]]), atol=1e-5
    )
    assert jnp.allclose(
        jnp.array(result_mle["DS_Q"].values), jnp.array([[13.0]]), atol=1e-5
    )


# ---------------------------------------------------------------------------
# _resolve_groups / aggregate_fisher_information_matrices
# ---------------------------------------------------------------------------


def test_aggregate_returns_unchanged_when_no_groups():
    """data_groups=None returns the original dict unchanged (same object)."""
    fim = {"DS_A": pd.DataFrame([[1.0]], index=["OpA"], columns=["OpA"])}

    result = aggregate_fisher_information_matrices(fim, data_groups=None)

    assert result is fim


def test_resolve_groups_declared_group_first():
    """Declared groups come first, in data_groups order, using source-name order."""
    groups = _resolve_groups(["DS_A", "DS_B", "DS_C"], {"Group1": ["DS_B", "DS_C"]})

    assert groups == [("Group1", [1, 2]), ("DS_A", [0])]


def test_resolve_groups_ungrouped_kept_individually():
    """With no groups declared, every source is kept as an individual entry."""
    groups = _resolve_groups(["DS_A", "DS_B"], {})

    assert groups == [("DS_A", [0]), ("DS_B", [1])]


def test_resolve_groups_empty_match_skipped(caplog):
    """A group matching no source is skipped, with a warning logged."""
    with caplog.at_level(logging.WARNING):
        groups = _resolve_groups(["DS_A"], {"EmptyGroup": ["NOPE"]})

    assert groups == [("DS_A", [0])]
    assert "EmptyGroup" in caplog.text


def test_aggregate_sums_grouped_matrices():
    """Grouped sources are summed elementwise; ungrouped sources are kept as-is."""
    fim = {
        "DS_A": pd.DataFrame(
            [[1.0, 0.0], [0.0, 2.0]], index=["OpA", "OpB"], columns=["OpA", "OpB"]
        ),
        "DS_B": pd.DataFrame(
            [[3.0, 0.0], [0.0, 4.0]], index=["OpA", "OpB"], columns=["OpA", "OpB"]
        ),
        "DS_C": pd.DataFrame(
            [[5.0, 0.0], [0.0, 6.0]], index=["OpA", "OpB"], columns=["OpA", "OpB"]
        ),
    }
    data_groups = {"Merged": ["DS_A", "DS_B"]}

    result = aggregate_fisher_information_matrices(fim, data_groups)

    assert list(result.keys()) == ["Merged", "DS_C"]
    expected_merged = pd.DataFrame(
        [[4.0, 0.0], [0.0, 6.0]], index=["OpA", "OpB"], columns=["OpA", "OpB"]
    )
    pd.testing.assert_frame_equal(result["Merged"], expected_merged)
    pd.testing.assert_frame_equal(result["DS_C"], fim["DS_C"])
