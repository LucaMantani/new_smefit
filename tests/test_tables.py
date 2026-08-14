"""Unit tests for smefit.tables — the Fisher and PCA report tables."""

import numpy as np
import pandas as pd
import pytest

from smefit.op_to_latex import coeff_info_latex
from smefit.pca import PCA
from smefit.tables import fisher_diagonals_normalised, pca_components, pca_spectrum


def _fim_entry(coeff_names, diag):
    return pd.DataFrame(np.diag(diag), index=coeff_names, columns=coeff_names)


def test_fisher_diagonals_normalised_values_and_rows_sum_to_one():
    """Rows are the per-source diagonal shares, normalised to sum to 1."""
    fim = {
        "DS_A": _fim_entry(["OpA", "OpZZ"], [2.0, 6.0]),
        "DS_B": _fim_entry(["OpA", "OpZZ"], [8.0, 2.0]),
    }

    result = fisher_diagonals_normalised(fim)

    assert result.columns.tolist() == ["DS_A", "DS_B"]
    assert result.loc["OpA"].tolist() == [0.2, 0.8]
    assert result.loc["OpZZ"].tolist() == [0.75, 0.25]
    assert np.allclose(result.sum(axis=1).to_numpy(), 1.0)


def test_fisher_diagonals_normalised_uses_latex_label_when_known():
    """Coefficient names present in coeff_info_latex are relabelled in the index."""
    fim = {
        "DS_A": _fim_entry(["OQQ1"], [3.0]),
        "DS_B": _fim_entry(["OQQ1"], [1.0]),
    }

    result = fisher_diagonals_normalised(fim)

    assert result.index.tolist() == [r"$c_{QQ}^{\scriptscriptstyle 1}$"]
    assert result.iloc[0].tolist() == [0.75, 0.25]


def test_fisher_diagonals_normalised_unknown_name_falls_back_to_raw():
    """Coefficient names absent from coeff_info_latex keep their raw name."""
    fim = {"DS_A": _fim_entry(["NotARealOp"], [1.0])}

    result = fisher_diagonals_normalised(fim)

    assert result.index.tolist() == ["NotARealOp"]


def test_fisher_diagonals_normalised_restricts_and_orders_by_params_to_plot():
    """The runcard's params_to_plot picks the rows, in the order written. Each
    row is normalised on its own, so the kept ones still sum to 1 — dropping a
    coefficient does not redistribute its share."""
    fim = {
        "DS_A": _fim_entry(["OpA", "OpB", "OpZZ"], [2.0, 1.0, 6.0]),
        "DS_B": _fim_entry(["OpA", "OpB", "OpZZ"], [8.0, 3.0, 2.0]),
    }

    result = fisher_diagonals_normalised(fim, params_to_plot=["OpZZ", "OpA"])

    assert result.index.tolist() == ["OpZZ", "OpA"]
    assert result.loc["OpZZ"].tolist() == [0.75, 0.25]
    assert result.loc["OpA"].tolist() == [0.2, 0.8]


def test_fisher_diagonals_normalised_single_source_is_all_ones():
    """With a single source, each row's only entry is trivially 1.0."""
    fim = {"DS_A": _fim_entry(["OpA", "OpZZ"], [5.0, 9.0])}

    result = fisher_diagonals_normalised(fim)

    assert result.columns.tolist() == ["DS_A"]
    assert result["DS_A"].tolist() == [1.0, 1.0]


# ---------------------------------------------------------------------------
# pca_components / pca_spectrum
# ---------------------------------------------------------------------------


def _pca(**kwargs):
    return PCA(
        eigenvalues=np.array([4.0, 1.0, 1e-9]),
        eigenvectors=np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]),
        coeff_names=["OpA", "OpZZ", "OpC"],
        **{"threshold": 1.0e-3, "min_weight": 0.01, **kwargs},
    )


def test_pca_components_shape_and_labels():
    result = pca_components(_pca())

    assert result.columns.tolist() == ["PC1", "PC2", "PC3"]
    assert result.shape == (3, 3)
    assert result.iloc[0, 0] == 0.8


def test_pca_components_uses_latex_label_when_known():
    result = pca_components(_pca())

    assert result.index.tolist() == [
        coeff_info_latex.get("OpA", "OpA"),
        coeff_info_latex.get("OpZZ", "OpZZ"),
        "OpC",
    ]


def test_pca_components_keeps_every_coefficient():
    """No params_to_plot: a column is a unit vector over all of them."""
    result = pca_components(_pca())

    assert result.shape == (3, 3)
    assert np.allclose((result.values**2).sum(axis=0), 1.0)


def test_pca_spectrum_columns_and_order():
    result = pca_spectrum(_pca())

    assert result.index.tolist() == ["PC1", "PC2", "PC3"]
    assert result["Eigenvalue"].tolist() == [4.0, 1.0, 1e-9]
    assert result["Sigma"].tolist()[:2] == [0.5, 1.0]
    assert result["Ratio"].tolist()[:2] == [1.0, 0.25]
    assert result["Flat"].tolist() == [False, False, True]


def test_pca_spectrum_cumulative_reaches_one():
    result = pca_spectrum(_pca())

    assert result["Cumulative"].iloc[-1] == pytest.approx(1.0)
    assert result["Cumulative"].iloc[0] == pytest.approx(0.8)


def test_pca_spectrum_names_the_direction():
    result = pca_spectrum(_pca())

    assert result["Direction"].iloc[0] == "0.80 OpA + 0.60 OpZZ"
