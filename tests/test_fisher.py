"""Unit tests for smefit.fisher — the per-source and total Fisher matrices."""

from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml
from reportengine import namespaces
from reportengine.configparser import ConfigError
from reportengine.resourcebuilder import FuzzyTarget, ResourceBuilder

from smefit.api import smefitAPI
from smefit.chi2 import Chi2, build_chi2, build_datasets_chi2
from smefit.config import smefitConfig
from smefit.core import (
    Coefficient,
    CoefficientGroup,
    DataGroup,
    Dataset,
    Theory,
    TheoryGroup,
)
from smefit.fisher import (
    _aggregate_fisher_from_fit,
    _aggregate_fisher_from_runcard,
    _resolve_groups,
    fisher_information_matrices,
    global_information_matrix,
    total_fisher_information_matrix,
)
from smefit.fit_result import Fit, FitResult, FitResultGroup
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
# _resolve_groups / _aggregate_fisher_from_runcard
# ---------------------------------------------------------------------------


def test_aggregate_returns_unchanged_when_no_groups():
    """data_groups=None returns the original dict unchanged (same object)."""
    fim = {"DS_A": pd.DataFrame([[1.0]], index=["OpA"], columns=["OpA"])}

    result = _aggregate_fisher_from_runcard(fim, data_groups=None)

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

    result = _aggregate_fisher_from_runcard(fim, data_groups)

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
        _aggregate_fisher_from_runcard(fim, {"Merged": ["DS_A", "DS_B"]})
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


def test_gim_refuses_a_flat_direction_in_float32():
    """Under -f32 a flat direction is only flat to ~1e-7, far above the float64
    threshold, and must still be caught."""
    v = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
    F = (np.eye(3) - np.outer(v, v)).astype(np.float32)
    # rounding noise of float32 size along the flat direction
    F += np.float32(1e-7) * np.outer(v, v).astype(np.float32)
    names = ["OpA", "OpB", "OpC"]
    fim = {"DS_A": pd.DataFrame(F, index=names, columns=names)}

    with pytest.raises(ValueError, match="1 flat"):
        global_information_matrix(fim)


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


# ---------------------------------------------------------------------------
# aggregate_fisher_information_matrices of a fit that has already been run
# ---------------------------------------------------------------------------

_BEST_FIT = {"OpA": 0.02, "OpB": -0.01}


def _fit_runcard(data_path: Path, theory_path: Path, group: str | None = None) -> dict:
    dataset = {"name": "TESTDATA", "order": "NLO"}
    if group is not None:
        dataset["group"] = group
    return {
        "data_path": str(data_path),
        "theory_path": str(theory_path),
        "use_theory_covmat": False,
        "use_t0": False,
        "use_quad": True,
        "datasets": [dataset],
        "coefficients": {name: {"free": True, "prior": _PRIOR} for name in _BEST_FIT},
        "actions_": ["run_analytic_fit"],
    }


def _write_fit(
    tmp_path: Path, runcard: dict, best_fit: dict[str, float] = _BEST_FIT
) -> Path:
    """A fit directory as a smefit run leaves it, with a best fit off the SM."""
    fit_dir = tmp_path / "my_fit"
    (fit_dir / "input").mkdir(parents=True)
    (fit_dir / "input" / "runcard.yaml").write_text(yaml.dump(runcard))
    FitResult(
        free_parameters=list(best_fit),
        best_fit_point=best_fit,
        max_loglikelihood=-1.0,
        num_data=3,
    ).write(fit_dir)
    return fit_dir


def test_fisher_of_a_fit_is_the_hessian_of_its_runcard_at_its_best_fit(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    fit = Fit.from_folder(_write_fit(tmp_path, runcard))

    result = _aggregate_fisher_from_fit(fit)

    chi2 = smefitAPI.datasets_chi2(**runcard)[0]
    c0 = jnp.array(list(_BEST_FIT.values()))
    expected = 0.5 * jax.hessian(chi2)(c0)
    assert list(result) == ["TESTDATA"]
    assert result["TESTDATA"].index.tolist() == list(_BEST_FIT)
    np.testing.assert_allclose(result["TESTDATA"].values, expected)
    # with quadratics the point matters: the test would pass on the SM otherwise
    at_sm = 0.5 * jax.hessian(chi2)(jnp.zeros(2))
    assert not np.allclose(result["TESTDATA"].values, at_sm)


def test_fisher_of_a_fit_uses_the_groups_of_its_runcard(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path, group="Top")
    fit = Fit.from_folder(_write_fit(tmp_path, runcard))

    assert list(_aggregate_fisher_from_fit(fit)) == ["Top"]


def test_fisher_of_a_fit_leaves_its_runcard_record_untouched(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    fit = Fit.from_folder(_write_fit(tmp_path, runcard))

    _aggregate_fisher_from_fit(fit)

    assert fit.fit_runcard == runcard


def test_fisher_node_takes_a_fit_key_with_no_datasets_of_its_own(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """The graph must resolve without datasets_chi2 and gd_best_fit."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    # at the minimum, as a real fit is: the GIM refuses negative total curvature
    minimum = smefitAPI.gd_best_fit(**runcard, gradient_descent_settings={})
    _write_fit(tmp_path, runcard, dict(zip(_BEST_FIT, map(float, minimum))))

    result = smefitAPI.global_information_matrix(
        fit={"name": "my_fit", "path": str(tmp_path)}
    )

    assert result.index.tolist() == list(_BEST_FIT)
    assert result.columns.tolist() == ["TESTDATA"]


def test_fisher_node_runs_once_per_entry_of_fits(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """``- fits <action>``, the form a fits:-only runcard uses."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    points = {"a": {"OpA": 0.02, "OpB": -0.01}, "b": {"OpA": -0.03, "OpB": 0.01}}
    for name, point in points.items():
        _write_fit(tmp_path / name, runcard, point)
    fits = [{"name": "my_fit", "path": str(tmp_path / name)} for name in points]

    target = "aggregate_fisher_information_matrices"
    config = smefitConfig({"fits": fits}, environment=smefitAPI.loadedenv)
    builder = ResourceBuilder(
        config,
        smefitAPI.provider_loaded,
        [FuzzyTarget(target, ("fits",), (), ())],
        perform_final=False,
    )
    builder.rootns.update(smefitAPI.loadedenv.ns_dump())
    builder.resolve_fuzzytargets()
    builder.execute_sequential()

    chi2 = smefitAPI.datasets_chi2(**runcard)[0]
    for index, point in enumerate(points.values()):
        result = namespaces.resolve(builder.rootns, [("fits", index)])[target]
        expected = 0.5 * jax.hessian(chi2)(jnp.array(list(point.values())))
        np.testing.assert_allclose(result["TESTDATA"].values, expected)


def test_fisher_node_without_a_fit_still_reads_the_runcard(
    fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """No fit in scope: datasets_chi2 and gd_best_fit, as always."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path, group="Top")
    runcard["gradient_descent_settings"] = {"sm_solution": True}

    result = smefitAPI.aggregate_fisher_information_matrices(**runcard)

    chi2 = smefitAPI.datasets_chi2(**runcard)[0]
    assert list(result) == ["Top"]
    np.testing.assert_allclose(
        result["Top"].values, 0.5 * jax.hessian(chi2)(jnp.zeros(2))
    )


# A quadratic bowl in OpA and OpB, with a curvature its constructor is given.
_EXTERNAL_CHI2 = """
import jax.numpy as jnp


class ExtChi2:
    def __init__(self, coefficients, rge_dict=None, curvature=1.0):
        self.num_data = 1
        self.param_names = coefficients.free_names
        self.curvature = curvature

    def compute_chi2(self, c):
        return self.curvature * jnp.sum(c**2)
"""


def _external_entry(tmp_path: Path, group: str | None = None) -> dict:
    module = tmp_path / "ext_chi2_for_fisher.py"
    module.write_text(_EXTERNAL_CHI2)
    entry: dict = {"path": str(module), "curvature": 2.0}
    if group is not None:
        entry["group"] = group
    return {"ExtChi2": entry}


def test_fisher_of_a_fit_merges_an_external_chi2_into_its_group(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """A group shared by a dataset and an external chi2 is one source."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path, group="Top")
    runcard["external_chi2"] = _external_entry(tmp_path, group="Top")
    fit = Fit.from_folder(_write_fit(tmp_path, runcard))

    result = _aggregate_fisher_from_fit(fit)

    assert list(result) == ["Top"]
    data_chi2 = smefitAPI.datasets_chi2(**fit.rebuild_runcard())[0]
    data_fisher = 0.5 * jax.hessian(data_chi2)(jnp.array(list(_BEST_FIT.values())))
    # F = 0.5 * H of curvature * c^2 is curvature on the diagonal
    np.testing.assert_allclose(result["Top"].values, data_fisher + 2.0 * np.eye(2))


def test_fisher_of_a_fit_keeps_an_external_chi2_in_its_own_group(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path, group="Top")
    runcard["external_chi2"] = _external_entry(tmp_path, group="DY")
    fit = Fit.from_folder(_write_fit(tmp_path, runcard))

    assert sorted(_aggregate_fisher_from_fit(fit)) == ["DY", "Top"]


def test_fisher_of_an_external_only_fit_uses_its_groups(tmp_path: Path) -> None:
    """No datasets: the groups come from the external chi2s alone."""
    runcard = {
        "external_chi2": _external_entry(tmp_path, group="DY"),
        "coefficients": {name: {"free": True, "prior": _PRIOR} for name in _BEST_FIT},
        "actions_": ["run_analytic_fit"],
    }
    fit = Fit.from_folder(_write_fit(tmp_path, runcard))

    result = _aggregate_fisher_from_fit(fit)

    assert list(result) == ["DY"]
    np.testing.assert_allclose(result["DY"].values, 2.0 * np.eye(2))


def test_data_groups_need_no_other_node_to_parse_external_chi2(tmp_path: Path) -> None:
    """Asked for on its own, the node still sees the external chi2 groups."""
    assert smefitAPI.data_groups(
        datasets=[{"name": "A", "order": "LO", "group": "G"}],
        external_chi2=_external_entry(tmp_path, group="G"),
    ) == {"G": ["A", "ExtChi2"]}


# ---------------------------------------------------------------------------
# Fits whose stored runcard does not describe the joint likelihood
# ---------------------------------------------------------------------------


def test_fisher_node_refuses_individual_fits(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    runcard["actions_"] = ["run_individual_analytic_fits"]
    fit_dir = tmp_path / "my_fit"
    (fit_dir / "input").mkdir(parents=True)
    (fit_dir / "input" / "runcard.yaml").write_text(yaml.dump(runcard))
    FitResultGroup(
        [
            FitResult(
                free_parameters=[name],
                best_fit_point={name: value},
                max_loglikelihood=-1.0,
                num_data=3,
                samples={name: jnp.array([value - 0.1, value, value + 0.1])},
            )
            for name, value in _BEST_FIT.items()
        ]
    ).write_summary(fit_dir)

    with pytest.raises(ConfigError, match="individual fits"):
        smefitAPI.aggregate_fisher_information_matrices(
            fit={"name": "my_fit", "path": str(tmp_path)}
        )


def test_fisher_of_a_bayesian_update_warns_it_covers_only_the_update(
    tmp_path: Path,
    fixtures_data_path: Path,
    fixtures_theory_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The previous fit is not needed: datasets_chi2 does not depend on the prior."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    runcard["bayesian_update"] = {"name": "previous_fit"}
    fit = Fit.from_folder(_write_fit(tmp_path, runcard))

    with caplog.at_level(logging.WARNING, logger="smefit.fisher"):
        result = _aggregate_fisher_from_fit(fit)

    assert "Bayesian update" in caplog.text
    assert list(result) == ["TESTDATA"]


def test_fisher_of_an_ordinary_fit_does_not_warn(
    tmp_path: Path,
    fixtures_data_path: Path,
    fixtures_theory_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    fit = Fit.from_folder(_write_fit(tmp_path, runcard))

    with caplog.at_level(logging.WARNING, logger="smefit.fisher"):
        _aggregate_fisher_from_fit(fit)

    assert "Bayesian update" not in caplog.text
