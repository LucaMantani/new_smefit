"""Unit tests for smefit.tables — fisher_diagonals_normalised."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from smefit.fit_result import Fit, FitResult, FitResultGroup
from smefit.op_to_latex import coeff_info_latex
from smefit.tables import (
    _REACH_COLUMN,
    coefficient_bounds_table,
    fisher_diagonals_normalised,
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


def _fit(samples, fit_name="my_fit", label=None, action="run_analytic_fit"):
    """A joint fit holding the given posterior samples."""
    return Fit(
        fit_results=FitResult(
            free_parameters=list(samples),
            best_fit_point={},
            max_loglikelihood=-1.0,
            num_data=10,
            samples={name: jnp.array(vals) for name, vals in samples.items()},
        ),
        fit_name=fit_name,
        label=label,
        fit_runcard={"actions_": [action]},
    )


def _individual_fit(samples, fit_name="individual_fit"):
    """A fit run one coefficient at a time."""
    return Fit(
        fit_results=FitResultGroup(
            [
                FitResult(
                    free_parameters=[name],
                    best_fit_point={name: 0.0},
                    max_loglikelihood=-1.0,
                    num_data=10,
                    samples={name: jnp.array(vals)},
                )
                for name, vals in samples.items()
            ]
        ),
        fit_name=fit_name,
        fit_runcard={"actions_": ["run_individual_analytic_fits"]},
    )


def _ramp_fit(**kwargs):
    """Exact percentiles: 0..1000 puts the 68% interval at [160, 840] and the
    95% one at [25, 975], around a mean of 500."""
    ramp = np.arange(0.0, 1001.0).tolist()
    return _fit({"OpA": ramp, "OpZZ": ramp}, **kwargs)


def test_coefficient_bounds_table_quotes_the_percentile_intervals():
    """One level by default, so a table is readable as it comes."""
    table = coefficient_bounds_table([_ramp_fit()])

    row = table.loc["OpZZ"]
    assert list(table.columns.get_level_values(1)) == ["best", r"95\% CL"]
    assert row[("my_fit", "best")] == "500.000"
    assert row[("my_fit", r"95\% CL")] == "[25.000, 975.000]"


def test_coefficient_bounds_table_quotes_the_68_percent_interval_on_request():
    table = coefficient_bounds_table([_ramp_fit()], bounds_levels=[68, 95])

    assert table.loc["OpZZ", ("my_fit", r"68\% CL")] == "[160.000, 840.000]"


def test_coefficient_bounds_table_columns_are_grouped_by_fit():
    table = coefficient_bounds_table(
        [
            _ramp_fit(fit_name="fit_a", label=r"$\mathrm{Analytic}$"),
            _ramp_fit(fit_name="fit_b"),
        ]
    )

    assert table.columns.tolist() == [
        (r"$\mathrm{Analytic}$", "best"),
        (r"$\mathrm{Analytic}$", r"95\% CL"),
        ("fit_b", "best"),
        ("fit_b", r"95\% CL"),
    ]


def test_coefficient_bounds_table_rows_are_latex_labels_in_order():
    table = coefficient_bounds_table([_ramp_fit()], params_to_plot=["OpZZ", "OpA"])

    assert table.index.tolist() == ["OpZZ", coeff_info_latex.get("OpA", "OpA")]


def test_coefficient_bounds_table_rounds_to_round_val():
    table = coefficient_bounds_table(
        [_fit({"OpA": [0.0, 0.123456, 0.25], "OpZZ": [0.0, 1.0]})], round_val=2
    )

    assert table.loc["OpA", ("my_fit", "best")] == "0.12"


def test_coefficient_bounds_table_reach_column_is_off_by_default():
    table = coefficient_bounds_table([_ramp_fit()])

    assert _REACH_COLUMN not in table.columns.get_level_values(1)


def test_coefficient_bounds_table_shows_the_reach_when_asked():
    """The same number the mass-reach plot draws: 95% of the ramp is
    [25, 975], so the bound is 475."""
    table = coefficient_bounds_table([_ramp_fit()], show_reach=True)

    reach = table.loc["OpZZ", ("my_fit", _REACH_COLUMN)]
    assert reach == f"{1 / np.sqrt(475.0):.3f}"


def test_coefficient_bounds_table_reach_follows_its_own_confidence_level():
    """The reach level does not touch the bounds columns, which are named
    after the levels they quote."""
    table = coefficient_bounds_table(
        [_ramp_fit()], show_reach=True, confidence_level=68, bounds_levels=[95]
    )

    assert table.loc["OpZZ", ("my_fit", _REACH_COLUMN)] == f"{1 / np.sqrt(340.0):.3f}"
    assert table.loc["OpZZ", ("my_fit", r"95\% CL")] == "[25.000, 975.000]"


def test_coefficient_bounds_table_reach_can_use_the_full_interval():
    table = coefficient_bounds_table([_ramp_fit()], show_reach=True, full_interval=True)

    assert table.loc["OpZZ", ("my_fit", _REACH_COLUMN)] == f"{1 / np.sqrt(950.0):.3f}"


def test_coefficient_bounds_table_can_show_the_reach_alone():
    table = coefficient_bounds_table([_ramp_fit()], show_bounds=False, show_reach=True)

    assert table.columns.get_level_values(1).tolist() == [_REACH_COLUMN]


def test_coefficient_bounds_table_shows_bounds_and_reach_together():
    table = coefficient_bounds_table(
        [_ramp_fit()], show_reach=True, bounds_levels=[68, 95]
    )

    assert table.columns.get_level_values(1).tolist() == [
        "best",
        r"68\% CL",
        r"95\% CL",
        _REACH_COLUMN,
    ]


def test_coefficient_bounds_table_rejects_a_table_of_nothing():
    with pytest.raises(ValueError, match="show_bounds"):
        coefficient_bounds_table([_ramp_fit()], show_bounds=False, show_reach=False)


def test_coefficient_bounds_table_gives_a_second_solution_its_own_row():
    bimodal = np.concatenate(
        [
            np.random.default_rng(2).normal(0.0, 0.1, 400),
            np.random.default_rng(3).normal(5.0, 0.1, 400),
        ]
    ).tolist()

    table = coefficient_bounds_table(
        [_fit({"OpA": bimodal, "OpZZ": bimodal})],
        params_to_plot=["OpA"],
        double_solution=["OpA"],
    )

    label = coeff_info_latex.get("OpA", "OpA")
    assert table.index.tolist() == [label, f"{label} (2)"]
    # the branch nearest the SM comes first, as in the plots
    assert float(table.iloc[0][("my_fit", "best")]) < 1.0
    assert float(table.iloc[1][("my_fit", "best")]) > 4.0


def test_coefficient_bounds_table_marks_a_solution_a_fit_does_not_have():
    """A coefficient split in one fit only still needs the row, and the fit
    that resolved a single solution says so rather than leaving it blank."""
    bimodal = np.concatenate(
        [
            np.random.default_rng(2).normal(0.0, 0.1, 400),
            np.random.default_rng(3).normal(5.0, 0.1, 400),
        ]
    ).tolist()
    fits = [
        _fit({"OpA": bimodal, "OpZZ": bimodal}, fit_name="split"),
        _fit({"OpA": bimodal, "OpZZ": bimodal}, fit_name="whole"),
    ]

    table = coefficient_bounds_table(
        fits, params_to_plot=["OpA"], double_solution={"split": ["OpA"]}
    )

    assert table.shape[0] == 2
    assert table.iloc[1][("whole", "best")] == "—"
    assert table.iloc[1][("split", "best")] != "—"


def test_coefficient_bounds_table_reads_an_individual_fit():
    ramp = np.arange(0.0, 1001.0).tolist()

    table = coefficient_bounds_table([_individual_fit({"OpA": ramp, "OpZZ": ramp})])

    assert table.loc["OpZZ", ("individual_fit", r"95\% CL")] == "[25.000, 975.000]"


def test_coefficient_bounds_table_rejects_an_empty_fits_list():
    with pytest.raises(ValueError, match="No fits"):
        coefficient_bounds_table([])


def test_coefficient_bounds_table_rejects_a_fit_without_samples():
    fit = Fit(
        fit_results=FitResult(
            free_parameters=["OpA"],
            best_fit_point={},
            max_loglikelihood=-1.0,
            num_data=10,
            samples=None,
        ),
        fit_name="sampleless",
        fit_runcard={"actions_": ["run_analytic_fit"]},
    )

    with pytest.raises(ValueError, match="no posterior samples"):
        coefficient_bounds_table([fit])


def test_coefficient_bounds_table_levels_are_choosable():
    """A table of one level is a common ask, and any level works — the
    columns are named after whichever were asked for."""
    table = coefficient_bounds_table([_ramp_fit()], bounds_levels=[95])

    assert table.columns.get_level_values(1).tolist() == ["best", r"95\% CL"]
    assert table.loc["OpZZ", ("my_fit", r"95\% CL")] == "[25.000, 975.000]"


def test_coefficient_bounds_table_levels_accept_a_bare_value():
    """`bounds_levels: 95` is a plausible way to write a one-element list."""
    table = coefficient_bounds_table([_ramp_fit()], bounds_levels=95)

    assert table.columns.get_level_values(1).tolist() == ["best", r"95\% CL"]


def test_coefficient_bounds_table_levels_keep_the_requested_order():
    table = coefficient_bounds_table([_ramp_fit()], bounds_levels=[95, 68])

    assert table.columns.get_level_values(1).tolist() == [
        "best",
        r"95\% CL",
        r"68\% CL",
    ]


def test_coefficient_bounds_table_takes_more_than_two_levels():
    table = coefficient_bounds_table([_ramp_fit()], bounds_levels=[68, 90, 95])

    assert table.columns.get_level_values(1).tolist() == [
        "best",
        r"68\% CL",
        r"90\% CL",
        r"95\% CL",
    ]


def test_coefficient_bounds_table_names_a_five_sigma_level_in_full():
    """99.99994% must not be rounded to 100% in its own column header, or the
    table says something it does not quote."""
    table = coefficient_bounds_table([_ramp_fit()], bounds_levels=[99.99994])

    assert table.columns.get_level_values(1).tolist() == ["best", r"99.99994\% CL"]


def test_coefficient_bounds_table_keeps_one_best_column_per_fit():
    """The central value is one number however many intervals surround it."""
    table = coefficient_bounds_table([_ramp_fit()], bounds_levels=[68, 90, 95])

    assert list(table.columns.get_level_values(1)).count("best") == 1


def test_coefficient_bounds_table_cells_are_fixed_decimal_strings():
    """reportengine formats a float column to four significant figures
    (-1.000E-2, -0), which reads nothing like the interval beside it."""
    table = coefficient_bounds_table(
        [_fit({"OpA": [0.0, 0.02], "OpZZ": [0.0, 1.0]})], round_val=2, show_reach=True
    )

    row = table.loc[coeff_info_latex.get("OpA", "OpA")]
    assert row[("my_fit", "best")] == "0.01"
    assert all(isinstance(cell, str) for cell in row)


def test_coefficient_bounds_table_does_not_print_a_negative_zero():
    """A value too small for the precision asked for reads as a defect when
    it comes out signed."""
    table = coefficient_bounds_table(
        [_fit({"OpA": [-0.002, -0.001], "OpZZ": [0.0, 1.0]})], round_val=2
    )

    assert table.loc[coeff_info_latex.get("OpA", "OpA"), ("my_fit", "best")] == "0.00"
