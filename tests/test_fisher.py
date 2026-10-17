"""Unit tests for smefit.fisher — the per-source and total Fisher matrices."""

import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from smefit.chi2 import Chi2, build_chi2, build_datasets_chi2
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
    global_information_matrix,
    total_fisher_information_matrix,
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


def _linear_setup(fit_covmat):
    """Two datasets, one free coefficient, linear EFT corrections only.

    Returns the total Chi2 and the (ndata, n_ops) matrix of linear corrections,
    so a test can compare the Fisher matrix against L^T C^-1 L directly.
    """
    theory_a = _make_theory("DS_A", sm=[10.0, 20.0], lin_op=[1.0, 2.0])
    theory_b = _make_theory("DS_B", sm=[30.0, 40.0], lin_op=[3.0, 4.0])
    theory = TheoryGroup([theory_a, theory_b])
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory, cg, use_quad=False)
    data = DataGroup(
        [_make_dataset("DS_A", [10.0, 20.0]), _make_dataset("DS_B", [30.0, 40.0])]
    )
    total = Chi2(
        build_chi2(model, data, fit_covmat),
        param_names=["OpA"],
        num_data=4,
    )
    per_dataset = build_datasets_chi2(model, data, fit_covmat)
    lin = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    return total, per_dataset, lin


def test_total_fisher_is_the_full_covmat_contraction():
    """F = L^T C^-1 L for a linear model, using the whole covariance matrix."""
    fit_covmat = jnp.eye(4).at[0, 2].set(0.4).at[2, 0].set(0.4)
    total, _, lin = _linear_setup(fit_covmat)

    result = total_fisher_information_matrix(total, _sm_point(1))

    expected = lin.T @ jnp.linalg.inv(fit_covmat) @ lin
    assert result.index.tolist() == ["OpA"]
    assert result.columns.tolist() == ["OpA"]
    assert jnp.allclose(jnp.array(result.values), expected)


def test_total_fisher_differs_from_the_sum_of_per_dataset_fishers():
    """The per-dataset matrices see diagonal blocks only, so they miss the
    cross-dataset correlation the total one includes."""
    fit_covmat = jnp.eye(4).at[0, 2].set(0.4).at[2, 0].set(0.4)
    total, per_dataset, _ = _linear_setup(fit_covmat)
    c0 = _sm_point(1)

    combined = total_fisher_information_matrix(total, c0).values.sum()
    summed = sum(
        df.values.sum() for df in fisher_information_matrices(per_dataset, c0).values()
    )

    assert not np.isclose(combined, summed)


def test_total_fisher_equals_the_sum_when_the_covmat_is_block_diagonal():
    """With no cross-dataset correlation the two agree, as they must."""
    fit_covmat = jnp.eye(4)
    total, per_dataset, _ = _linear_setup(fit_covmat)
    c0 = _sm_point(1)

    combined = total_fisher_information_matrix(total, c0).values.sum()
    summed = sum(
        df.values.sum() for df in fisher_information_matrices(per_dataset, c0).values()
    )

    assert np.isclose(combined, summed)


# ---------------------------------------------------------------------------
# Global information matrix
# ---------------------------------------------------------------------------


def _fim(names, **sources):
    return {
        source: pd.DataFrame(np.asarray(F, dtype=float), index=names, columns=names)
        for source, F in sources.items()
    }


def test_gim_diagonal_fishers_reduce_to_the_diagonal_shares():
    """Uncorrelated coefficients: alpha[i, k] = F_k[i, i] / F[i, i]."""
    fim = _fim(["OpA", "OpB"], DS_A=np.diag([3.0, 1.0]), DS_B=np.diag([1.0, 3.0]))

    gim = global_information_matrix(fim)

    assert gim.index.tolist() == ["OpA", "OpB"]
    assert gim.columns.tolist() == ["DS_A", "DS_B"]
    np.testing.assert_allclose(gim.to_numpy(), [[0.75, 0.25], [0.25, 0.75]])


def test_gim_credits_a_source_constraining_only_through_correlations():
    """DS_A constrains OpA alone, DS_B only the sum OpA + OpB.

    Sigma = F^{-1} = [[1, -1], [-1, 2]]. OpA is pinned by DS_A, so DS_B gets
    none of its variance despite having half of its Fisher diagonal. DS_A has
    no curvature along OpB, yet gets half of OpB's variance: fixing OpA is what
    lets DS_B's sum constrain OpB.
    """
    fim = _fim(
        ["OpA", "OpB"],
        DS_A=[[1.0, 0.0], [0.0, 0.0]],
        DS_B=[[1.0, 1.0], [1.0, 1.0]],
    )

    gim = global_information_matrix(fim)

    np.testing.assert_allclose(gim.to_numpy(), [[1.0, 0.0], [0.5, 0.5]], atol=1e-12)


def test_gim_rows_sum_to_one():
    """Shares of one coefficient's variance add up to all of it."""
    rng = np.random.default_rng(0)
    names = ["OpA", "OpB", "OpC"]
    sources = {}
    for k in range(4):
        m = rng.normal(size=(3, 3))
        sources[f"DS_{k}"] = m @ m.T

    gim = global_information_matrix(_fim(names, **sources))

    np.testing.assert_allclose(gim.sum(axis=1).to_numpy(), 1.0)


def test_gim_of_grouped_sources_is_the_sum_of_their_columns():
    """Grouping before the attribution agrees with summing columns after it,
    alpha being linear in F_k."""
    rng = np.random.default_rng(1)
    names = ["OpA", "OpB"]
    sources = {}
    for k in "ABC":
        m = rng.normal(size=(2, 2))
        sources[f"DS_{k}"] = m @ m.T
    fim = _fim(names, **sources)

    grouped = global_information_matrix(
        aggregate_fisher_information_matrices(fim, {"Merged": ["DS_A", "DS_B"]})
    )
    ungrouped = global_information_matrix(fim)

    assert grouped.columns.tolist() == ["Merged", "DS_C"]
    np.testing.assert_allclose(grouped["Merged"], ungrouped["DS_A"] + ungrouped["DS_B"])
    np.testing.assert_allclose(grouped["DS_C"], ungrouped["DS_C"])


def test_gim_refuses_a_flat_direction_and_names_it():
    """A flat combination OpB - OpC raises, rather than inverting to ~1e16."""
    v = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
    fim = _fim(["OpA", "OpB", "OpC"], DS_A=np.eye(3) - np.outer(v, v))

    with pytest.raises(ValueError, match="1 flat") as err:
        global_information_matrix(fim)

    assert "OpB, OpC" in str(err.value) or "OpC, OpB" in str(err.value)
    assert "OpA" not in str(err.value)


def test_gim_refuses_negative_total_curvature():
    """A total Fisher matrix that is not positive definite has no covariance."""
    fim = _fim(["OpA", "OpB"], DS_A=[[1.0, 0.0], [0.0, -1.0]])

    with pytest.raises(ValueError, match="OpB"):
        global_information_matrix(fim)


def test_gim_warns_on_negative_shares(caplog):
    """An indefinite F_k gives a negative share, kept and logged; the row still
    sums to 1, so another share goes above it."""
    fim = _fim(
        ["OpA", "OpB"],
        DS_A=[[2.0, 1.5], [1.5, 2.0]],
        DS_B=[[1.0, -1.0], [-1.0, -0.5]],
    )

    with caplog.at_level(logging.WARNING):
        gim = global_information_matrix(fim)

    assert gim.loc["OpB", "DS_B"] < 0
    assert gim.loc["OpB", "DS_A"] > 1
    np.testing.assert_allclose(gim.sum(axis=1).to_numpy(), 1.0)
    assert "OpB <- DS_B" in caplog.text


def test_gim_silent_when_every_source_is_positive_semidefinite(caplog):
    fim = _fim(["OpA", "OpB"], DS_A=np.eye(2), DS_B=[[1.0, 1.0], [1.0, 1.0]])

    with caplog.at_level(logging.WARNING):
        global_information_matrix(fim)

    assert "negative share" not in caplog.text
