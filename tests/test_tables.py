"""Unit tests for smefit.tables — fisher_diagonals_normalised and
coefficient_bounds_table."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from smefit.fit_result import Fit, FitResult, FitResultGroup
from smefit.tables import coefficient_bounds_table, fisher_diagonals_normalised


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


def test_coefficient_bounds_table_quotes_the_mean_and_both_intervals() -> None:
    table = coefficient_bounds_table(_joint_fit({"OtG": _RAMP}))

    assert table.columns.tolist() == ["mean", "68% CL", "95% CL"]
    assert table.index.tolist() == [r"$c_{tG}$"]
    assert table.iloc[0].tolist() == [
        "500.000",
        "[160.000, 840.000]",
        "[25.000, 975.000]",
    ]


def test_coefficient_bounds_table_rounds_to_round_val_without_a_negative_zero() -> None:
    """-0.0004 at two decimals is written 0.00, not -0.00."""
    table = coefficient_bounds_table(_joint_fit({"OtG": [-0.0004] * 10}), round_val=2)

    assert table.iloc[0].tolist() == ["0.00", "[0.00, 0.00]", "[0.00, 0.00]"]


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
