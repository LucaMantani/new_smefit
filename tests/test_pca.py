"""Unit tests for smefit.pca — the PCA dataclass and the pca node."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from smefit.pca import PCA, _fix_signs, pca

SETTINGS = {"threshold": 1.0e-3, "min_weight": 0.01}


def _fisher(matrix, names):
    return pd.DataFrame(np.asarray(matrix, dtype=float), index=names, columns=names)


def _pca_of(matrix, names, **overrides):
    """Run the pca node on a hand-built Fisher matrix."""
    return pca(_fisher(matrix, names), {**SETTINGS, **overrides})


# ----------------------------------------------------------------------
# the decomposition
# ----------------------------------------------------------------------


def test_pca_diagonal_fisher_is_the_coefficient_basis():
    """A diagonal Fisher has the unit vectors as principal directions."""
    result = _pca_of(np.diag([1.0, 4.0, 0.0]), ["OpA", "OpB", "OpC"])

    assert np.allclose(result.eigenvalues, [4.0, 1.0, 0.0])
    assert np.allclose(np.abs(result.eigenvectors), np.eye(3)[:, [1, 0, 2]])
    assert result.coeff_names == ["OpA", "OpB", "OpC"]
    assert result.component_names == ["PC1", "PC2", "PC3"]


def test_pca_constraints_are_inverse_root_eigenvalues():
    """sigma_i = 1/sqrt(lambda_i), and infinite along a flat direction."""
    result = _pca_of(np.diag([4.0, 1.0, 0.0]), ["OpA", "OpB", "OpC"])

    assert np.allclose(result.constraints[:2], [0.5, 1.0])
    assert not np.isfinite(result.constraints[2])


def test_pca_flags_the_flat_direction():
    result = _pca_of(np.diag([1.0, 1e-9]), ["OpA", "OpB"])

    assert result.flat_mask.tolist() == [False, True]
    assert result.n_flat == 1


def test_pca_flat_threshold_is_relative_to_the_largest_eigenvalue():
    """A direction 100x weaker than the strongest is flat at threshold=1e-1."""
    weak = np.diag([1.0, 1e-2])

    assert _pca_of(weak, ["OpA", "OpB"], threshold=1e-3).n_flat == 0
    assert _pca_of(weak, ["OpA", "OpB"], threshold=1e-1).n_flat == 1


def test_pca_recovers_a_rotated_direction():
    """The eigenvector of a correlated Fisher is the combination it constrains."""
    # Curvature only along (OpA + OpB)/sqrt(2); the orthogonal direction is flat.
    v = np.array([1.0, 1.0]) / np.sqrt(2.0)
    result = _pca_of(4.0 * np.outer(v, v), ["OpA", "OpB"])

    assert np.allclose(result.eigenvalues, [4.0, 0.0])
    assert np.allclose(result.eigenvectors[:, 0], v)
    assert result.n_flat == 1


def test_pca_signs_are_stable_under_an_overall_flip():
    """The sign convention makes the reported directions reproducible."""
    v = np.array([-0.6, 0.8])
    first = _pca_of(np.outer(v, v) + 1e-3 * np.eye(2), ["OpA", "OpB"])
    second = _pca_of(np.outer(-v, -v) + 1e-3 * np.eye(2), ["OpA", "OpB"])

    assert np.allclose(first.eigenvectors, second.eigenvectors)


def test_fix_signs_makes_the_largest_component_positive():
    vectors = np.array([[0.6, -0.8], [-0.8, -0.6]])
    fixed = _fix_signs(vectors)

    pivot = np.argmax(np.abs(fixed), axis=0)
    assert np.all(fixed[pivot, np.arange(2)] > 0)


def test_pca_matches_the_svd_of_the_scaled_linear_corrections():
    """The Fisher route reproduces the classic linear-corrections PCA.

    For a linear-only chi2, chi2(c) = (L c)^T Sigma^-1 (L c) up to constants, so
    F = 0.5 * d2chi2/dc2 = L^T Sigma^-1 L and the eigenvalues of F are the
    squared singular values of Sigma^{-1/2} L.
    """
    rng = np.random.default_rng(0)
    lin = rng.normal(size=(6, 3))  # linear EFT corrections, (ndata, n_ops)
    sigma_exp = np.array([0.5, 1.0, 2.0, 1.5, 0.8, 1.2])
    inv_cov = np.diag(1.0 / sigma_exp**2)

    result = _pca_of(lin.T @ inv_cov @ lin, ["OpA", "OpB", "OpC"])

    singular = np.linalg.svd(lin / sigma_exp[:, None], compute_uv=False)
    assert np.allclose(result.eigenvalues, singular**2)


def test_pca_carries_the_settings_onto_the_result():
    """The settings live on the PCA object, so the tables need not re-read them."""
    result = _pca_of(
        np.diag([1.0, 2.0]), ["OpA", "OpB"], threshold=1e-2, min_weight=0.2
    )

    assert result.threshold == 1e-2
    assert result.min_weight == 0.2


def test_pca_raises_when_nothing_is_constrained():
    with pytest.raises(ValueError, match="no positive eigenvalue"):
        _pca_of(np.diag([-1.0, -2.0]), ["OpA", "OpB"])


def test_pca_warns_on_a_saddle_point(caplog):
    """A negative eigenvalue means the centre was not a minimum."""
    with caplog.at_level(logging.WARNING, logger="smefit.pca"):
        _pca_of(np.diag([1.0, -0.5]), ["OpA", "OpB"])

    assert "saddle" in caplog.text


def test_pca_warns_on_degenerate_constrained_directions(caplog):
    with caplog.at_level(logging.WARNING, logger="smefit.pca"):
        _pca_of(np.diag([1.0, 1.0]), ["OpA", "OpB"])

    assert "degenerate" in caplog.text


def test_pca_does_not_call_flat_directions_degenerate(caplog):
    """Flat directions are all ~0 and so trivially degenerate; that is not news."""
    with caplog.at_level(logging.WARNING, logger="smefit.pca"):
        _pca_of(np.diag([1.0, 0.0, 0.0]), ["OpA", "OpB", "OpC"])

    assert "degenerate" not in caplog.text


# ----------------------------------------------------------------------
# presentation
# ----------------------------------------------------------------------


def _two_direction_pca(**overrides):
    return PCA(
        eigenvalues=np.array([4.0, 1e-9]),
        eigenvectors=np.array([[0.8, -0.6], [0.6, 0.8]]),
        coeff_names=["OpA", "OpB"],
        **{**SETTINGS, **overrides},
    )


def test_describe_orders_by_weight_and_signs_the_terms():
    assert _two_direction_pca().describe(0) == "0.80 OpA + 0.60 OpB"
    assert _two_direction_pca().describe(1) == "0.80 OpB - 0.60 OpA"


def test_describe_drops_components_below_min_weight():
    assert _two_direction_pca(min_weight=0.7).describe(0) == "0.80 OpA"


def test_describe_always_keeps_the_leading_term():
    """A direction is never described as nothing, however high min_weight is."""
    assert _two_direction_pca(min_weight=0.99).describe(0) == "0.80 OpA"


def test_as_frame_labels_rows_and_columns():
    frame = _two_direction_pca().as_frame()

    assert frame.index.tolist() == ["OpA", "OpB"]
    assert frame.columns.tolist() == ["PC1", "PC2"]
    assert frame.loc["OpA", "PC1"] == pytest.approx(0.8)


def test_to_dict_is_json_serialisable_with_null_for_infinite_widths():
    result = _pca_of(np.diag([4.0, 0.0]), ["OpA", "OpB"])

    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["eigenvalues"] == pytest.approx([4.0, 0.0])
    assert payload["constraints"][0] == pytest.approx(0.5)
    assert payload["constraints"][1] is None
    assert payload["flat"] == [False, True]
    assert payload["coeff_names"] == ["OpA", "OpB"]


def test_write_serialises_to_pca_json(tmp_path):
    """write() creates its output directory, as FitResult.write does."""
    _pca_of(np.diag([4.0, 1.0]), ["OpA", "OpB"]).write(tmp_path / "out")

    payload = json.loads((tmp_path / "out" / "pca.json").read_text())
    assert payload["component_names"] == ["PC1", "PC2"]
    assert payload["eigenvalues"] == pytest.approx([4.0, 1.0])
    assert payload["flat"] == [False, False]


def test_print_summary_names_every_direction(capsys):
    """The console table has a row per PC, and flags the flat ones."""
    _pca_of(np.diag([4.0, 0.0]), ["OpA", "OpB"]).print_summary()

    out = capsys.readouterr().out
    assert "PC1" in out and "PC2" in out
    assert "n_flat" in out
    assert "flat" in out
