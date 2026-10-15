"""Unit tests for smefit.tables — the Fisher, PCA, scan and coefficient
bounds report tables."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from smefit import fit_result
from smefit.core import Coefficient, CoefficientGroup
from smefit.fit_result import Fit, FitResult, FitResultGroup
from smefit.latex_labels import default_latex_labels
from smefit.pca import PCA
from smefit.tables import (
    chi2_scan_table,
    coefficient_bounds_table,
    fisher_diagonals_normalised,
    mass_scan_table,
    pca_components,
    pca_spectrum,
)


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
    """Coefficient names present in default_latex_labels are relabelled in the index."""
    fim = {
        "DS_A": _fim_entry(["OQQ1"], [3.0]),
        "DS_B": _fim_entry(["OQQ1"], [1.0]),
    }

    result = fisher_diagonals_normalised(fim)

    assert result.index.tolist() == [r"$c_{QQ}^{\scriptscriptstyle 1}$"]
    assert result.iloc[0].tolist() == [0.75, 0.25]


def test_fisher_diagonals_normalised_applies_latex_labels_to_both_axes():
    """Runcard labels override a coefficient's built-in label and name a
    source (data group) the defaults do not know."""
    fim = {
        "MyGroup": _fim_entry(["OQQ1"], [3.0]),
        "DS_B": _fim_entry(["OQQ1"], [1.0]),
    }

    result = fisher_diagonals_normalised(
        fim, latex_labels={"OQQ1": "$c_1$", "MyGroup": r"$t\bar{t}$"}
    )

    assert result.index.tolist() == ["$c_1$"]
    assert result.columns.tolist() == [r"$t\bar{t}$", "DS_B"]
    assert result.iloc[0].tolist() == [0.75, 0.25]


def test_fisher_diagonals_normalised_unknown_name_falls_back_to_raw():
    """Coefficient names absent from default_latex_labels keep their raw name."""
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
# coefficient_bounds_table
# ---------------------------------------------------------------------------

# 1001 evenly spaced samples: the p-th percentile is exactly 10 * p
_RAMP: list[float] = [float(value) for value in range(1001)]


def _result(samples: dict[str, list[float]]) -> FitResult:
    return FitResult(
        free_parameters=list(samples),
        best_fit_point={name: 0.0 for name in samples},
        max_loglikelihood=0.0,
        num_data=1,
        samples={name: jnp.array(values) for name, values in samples.items()},
    )


def _joint_fit(samples: dict[str, list[float]]) -> Fit:
    return Fit(fit_results=_result(samples), fit_name="joint")


def _individual_fit(samples: dict[str, list[float]]) -> Fit:
    return Fit(
        fit_results=FitResultGroup(
            [_result({name: values}) for name, values in samples.items()]
        ),
        fit_name="individual",
    )


def test_coefficient_bounds_table_quotes_both_intervals() -> None:
    table = coefficient_bounds_table(_joint_fit({"OtG": _RAMP}))

    assert table.columns.tolist() == ["68% CL (ETI)", "95% CL (ETI)"]
    assert table.index.tolist() == [r"$c_{tG}$"]
    assert table.iloc[0].tolist() == ["[160.000, 840.000]", "[25.000, 975.000]"]


def test_coefficient_bounds_table_rounds_to_round_val_without_a_negative_zero() -> None:
    """-0.0004 at two decimals is written 0.00, not -0.00."""
    table = coefficient_bounds_table(_joint_fit({"OtG": [-0.0004] * 10}), round_val=2)

    assert table.iloc[0].tolist() == ["[0.00, 0.00]", "[0.00, 0.00]"]


def test_coefficient_bounds_table_keeps_params_to_plot_in_its_order() -> None:
    fit = _joint_fit({"OtG": _RAMP, "OpQM": _RAMP, "OpA": _RAMP})

    table = coefficient_bounds_table(fit, params_to_plot=["OpA", "OtG"])

    assert table.index.tolist() == ["OpA", r"$c_{tG}$"]


@pytest.mark.parametrize("params_to_plot", [None, ["OpQM"]])
def test_coefficient_bounds_table_reads_an_individual_fit_like_a_joint_one(
    params_to_plot: list[str] | None,
) -> None:
    samples = {"OtG": _RAMP, "OpQM": [1.0, 2.0, 3.0]}

    pd.testing.assert_frame_equal(
        coefficient_bounds_table(
            _individual_fit(samples), params_to_plot=params_to_plot
        ),
        coefficient_bounds_table(_joint_fit(samples), params_to_plot=params_to_plot),
    )


def test_coefficient_bounds_table_quotes_the_bounds_levels_asked_for() -> None:
    table = coefficient_bounds_table(
        _joint_fit({"OtG": _RAMP}), bounds_levels=[90, 99.99994]
    )

    assert table.columns.tolist() == ["90% CL (ETI)", "99.99994% CL (ETI)"]
    assert table.iloc[0].tolist()[0] == "[50.000, 950.000]"


def test_coefficient_bounds_table_takes_a_single_level() -> None:
    """``bounds_levels: 90`` in a runcard, as for ``params_to_plot: OtG``."""
    table = coefficient_bounds_table(_joint_fit({"OtG": _RAMP}), bounds_levels=90)

    assert table.columns.tolist() == ["90% CL (ETI)"]


def test_coefficient_bounds_table_heads_each_column_with_its_interval_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header names the interval asked for, and the cells are that
    interval's bounds. ``interval_types: fake`` in a runcard."""
    monkeypatch.setitem(
        fit_result._INTERVAL_TYPES, "fake", lambda _v, _l: [(-1.0, 1.0)]
    )

    table = coefficient_bounds_table(
        _joint_fit({"OtG": _RAMP}), bounds_levels=90, interval_types="fake"
    )

    assert table.columns.tolist() == ["90% CL (FAKE)"]
    assert table.iloc[0].tolist() == ["[-1.000, 1.000]"]


def test_coefficient_bounds_table_compares_interval_types_side_by_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several interval types are columns of the same table, grouped by level
    so that the types of one level sit next to each other."""
    monkeypatch.setitem(
        fit_result._INTERVAL_TYPES, "fake", lambda _v, _l: [(-1.0, 1.0)]
    )

    table = coefficient_bounds_table(
        _joint_fit({"OtG": _RAMP}),
        bounds_levels=[90, 95],
        interval_types=["eti", "fake"],
    )

    assert table.columns.tolist() == [
        "90% CL (ETI)",
        "90% CL (FAKE)",
        "95% CL (ETI)",
        "95% CL (FAKE)",
    ]
    assert table.iloc[0]["90% CL (ETI)"] == "[50.000, 950.000]"
    assert table.iloc[0]["90% CL (FAKE)"] == "[-1.000, 1.000]"


def test_coefficient_bounds_table_joins_the_pieces_of_a_disjoint_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multimodal region is one cell, its pieces joined by a union."""
    monkeypatch.setitem(
        fit_result._INTERVAL_TYPES, "fake", lambda _v, _l: [(-2.0, -1.0), (1.0, 2.0)]
    )

    table = coefficient_bounds_table(
        _joint_fit({"OtG": _RAMP}), bounds_levels=90, round_val=1, interval_types="fake"
    )

    assert table.iloc[0].tolist() == ["[-2.0, -1.0] ∪ [1.0, 2.0]"]


def test_coefficient_bounds_table_rejects_empty_bounds_levels() -> None:
    with pytest.raises(ValueError, match="bounds_levels is empty"):
        coefficient_bounds_table(_joint_fit({"OtG": _RAMP}), bounds_levels=[])


def test_coefficient_bounds_table_rejects_empty_interval_types() -> None:
    with pytest.raises(ValueError, match="interval_types is empty"):
        coefficient_bounds_table(_joint_fit({"OtG": _RAMP}), interval_types=[])


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
        default_latex_labels.get("OpA", "OpA"),
        default_latex_labels.get("OpZZ", "OpZZ"),
        "OpC",
    ]


def test_pca_components_applies_latex_labels():
    result = pca_components(_pca(), latex_labels={"OpC": "$c_C$"})

    assert result.index.tolist()[2] == "$c_C$"


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


# ---------------------------------------------------------------------------
# chi2_scan_table
# ---------------------------------------------------------------------------


def test_chi2_scan_table_builds_multiindex_columns_per_coefficient():
    scans = [
        {"OpA": {"points": [-1.0, 0.0, 1.0], "chi2": [4.0, 0.0, 4.0]}},
        {"OpZZ": {"points": [-2.0, 2.0], "chi2": [1.0, 1.0]}},
    ]

    result = chi2_scan_table(scans)

    assert list(result.columns.get_level_values(0).unique()) == ["OpA", "OpZZ"]
    assert result[("OpA", "value")].dropna().tolist() == [-1.0, 0.0, 1.0]
    assert result[("OpA", "chi2")].dropna().tolist() == [4.0, 0.0, 4.0]
    assert result[("OpZZ", "value")].dropna().tolist() == [-2.0, 2.0]


def test_chi2_scan_table_uses_latex_label_when_known():
    scans = [{"OQQ1": {"points": [0.0, 1.0], "chi2": [0.0, 2.0]}}]

    result = chi2_scan_table(scans)

    assert result.columns.get_level_values(0).unique().tolist() == [
        r"$c_{QQ}^{\scriptscriptstyle 1}$"
    ]


def test_chi2_scan_table_applies_latex_labels():
    scans = [{"OQQ1": {"points": [0.0, 1.0], "chi2": [0.0, 2.0]}}]

    result = chi2_scan_table(scans, latex_labels={"OQQ1": "$c_1$"})

    assert result.columns.get_level_values(0).unique().tolist() == ["$c_1$"]


def test_chi2_scan_table_merges_multiple_namespace_entries():
    """individual_chi2_scans is a list of single-key dicts, one per coefficient."""
    scans = [
        {"OpA": {"points": [0.0], "chi2": [0.0]}},
        {"OpZZ": {"points": [1.0], "chi2": [3.0]}},
        {"OpYY": {"points": [2.0], "chi2": [6.0]}},
    ]

    result = chi2_scan_table(scans)

    assert result.columns.get_level_values(0).unique().tolist() == [
        "OpA",
        "OpZZ",
        "OpYY",
    ]


# ---------------------------------------------------------------------------
# mass_scan_table
# ---------------------------------------------------------------------------


def _single_free_coeff_group(name):
    return CoefficientGroup(
        [
            Coefficient(
                name=name,
                free=True,
                prior={"dist": "uniform", "low": 0.0, "high": 1.0},
            )
        ]
    )


def test_mass_scan_table_columns_and_values():
    coefficients = _single_free_coeff_group("OpM")

    result = mass_scan_table(coefficients, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0])

    assert result.columns.tolist() == ["OpM", "chi2"]
    assert result["OpM"].tolist() == [1.0, 2.0, 3.0]
    assert result["chi2"].tolist() == [10.0, 20.0, 30.0]


def test_mass_scan_table_uses_latex_label_when_known():
    coefficients = _single_free_coeff_group("OQQ1")

    result = mass_scan_table(coefficients, [0.5], [1.5])

    assert result.columns.tolist() == [r"$c_{QQ}^{\scriptscriptstyle 1}$", "chi2"]


def test_mass_scan_table_applies_latex_labels():
    coefficients = _single_free_coeff_group("OpM")

    result = mass_scan_table(coefficients, [0.5], [1.5], latex_labels={"OpM": "$M$"})

    assert result.columns.tolist() == ["$M$", "chi2"]


def test_mass_scan_table_uses_first_free_coefficient_name():
    """mass_scan_table is only meaningful for a single free coefficient; it
    reads the first free name from `coefficients`."""
    coefficients = CoefficientGroup(
        [
            Coefficient(
                name="OpM",
                free=True,
                prior={"dist": "uniform", "low": 0.0, "high": 1.0},
            ),
            Coefficient(name="OpFixed", free=False, value=1.0),
        ]
    )

    result = mass_scan_table(coefficients, [1.0], [5.0])

    assert result.columns.tolist() == ["OpM", "chi2"]
