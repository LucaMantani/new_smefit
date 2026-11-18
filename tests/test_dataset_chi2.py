"""Tests of the per-dataset chi2 at a fit's best fit, and of the table comparing
it across fits (``smefit.dataset_chi2``, ``datasets_chi2_comparison_table``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pandas as pd
import pytest
import yaml
from reportengine.resourcebuilder import FuzzyTarget, ResourceBuilder

from smefit.api import smefitAPI
from smefit.chi2 import Chi2
from smefit.config import smefitConfig
from smefit.dataset_chi2 import UNGROUPED, best_fit_datasets_chi2
from smefit.fit_result import FitResult
from smefit.tables import (
    _ALL,
    _CHI2_PER_POINT,
    _NDAT,
    _TOTAL,
    datasets_chi2_comparison_table,
)

_BEST_FIT = {"OpA": 0.02, "OpB": -0.01}


def _plain(name):
    return name


def _chi2(name: str, ndat: int, value: float) -> Chi2:
    return Chi2(lambda c: value + 0.0 * c.sum(), ["c"], ndat, name=name)


def _per_fit(rows: dict[str, tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {name: {"group": g, "ndat": n, "chi2": c} for name, (g, n, c) in rows.items()},
        orient="index",
        columns=["group", "ndat", "chi2"],
    )


def _table(fits_rows, labels=None, round_val=3) -> pd.DataFrame:
    labels = labels or [f"fit{i}" for i in range(len(fits_rows))]
    fits = [SimpleNamespace(plot_label=label) for label in labels]
    return datasets_chi2_comparison_table(
        fits, [_per_fit(rows) for rows in fits_rows], _plain, round_val
    )


# ---------------------------------------------------------------------------
# best_fit_datasets_chi2
# ---------------------------------------------------------------------------


def test_best_fit_datasets_chi2_evaluates_each_dataset_with_its_group() -> None:
    chi2s = [_chi2("A", 4, 2.0), _chi2("B", 5, 10.0), _chi2("ext", 1, 3.0)]

    result = best_fit_datasets_chi2(chi2s, jnp.zeros(1), {"G": ["A", "ext"]})

    assert result.index.tolist() == ["A", "B", "ext"]
    assert result["group"].tolist() == ["G", UNGROUPED, "G"]
    assert result["ndat"].tolist() == [4, 5, 1]
    assert result["chi2"].tolist() == pytest.approx([2.0, 10.0, 3.0])


def test_best_fit_datasets_chi2_without_groups_puts_all_in_ungrouped() -> None:
    result = best_fit_datasets_chi2([_chi2("A", 4, 2.0)], jnp.zeros(1), None)
    assert result["group"].tolist() == [UNGROUPED]


# ---------------------------------------------------------------------------
# datasets_chi2_comparison_table
# ---------------------------------------------------------------------------


def _chi2_col(label: str) -> tuple[str, str]:
    return (_CHI2_PER_POINT, label)


_NDAT_COL = (_NDAT, "")
_BLANK = ("", "")


def _at(table: pd.DataFrame, row: tuple[str, str], column: tuple[str, str]) -> str:
    """The one cell of ``row``: the blank rows make the index non-unique."""
    [value] = table[column][[index == row for index in table.index]].tolist()
    return value


def test_comparison_table_is_one_block_per_group_then_the_overall_total() -> None:
    table = _table(
        [{"A": (UNGROUPED, 2, 1.0), "B": ("Top", 2, 1.0)}, {"C": ("Higgs", 1, 1.0)}]
    )
    assert table.index.tolist() == [
        ("Top", "B"),
        ("Top", _TOTAL),
        _BLANK,
        ("Higgs", "C"),
        ("Higgs", _TOTAL),
        _BLANK,
        (UNGROUPED, "A"),
        (UNGROUPED, _TOTAL),
        _BLANK,
        (_ALL, _TOTAL),
    ]


def test_comparison_table_separates_the_blocks_by_a_blank_row() -> None:
    table = _table([{"A": ("Top", 4, 2.0), "B": ("Higgs", 6, 9.0)}])
    assert table.index.tolist()[2] == _BLANK
    assert table.iloc[2].tolist() == [""] * len(table.columns)


def test_comparison_table_quotes_one_ndat_and_each_fit_chi2_per_point() -> None:
    table = _table(
        [
            {"A": ("Top", 4, 2.0), "B": ("Top", 6, 9.0)},
            {"A": ("Top", 4, 4.0), "B": ("Top", 6, 6.0)},
        ]
    )

    assert table.columns.tolist() == [_NDAT_COL, _chi2_col("fit0"), _chi2_col("fit1")]
    rows = table[[index != _BLANK for index in table.index]]
    assert rows[_NDAT_COL].tolist() == ["4", "6", "10", "10"]
    # the total is the summed chi2 over the summed points, not a mean of ratios
    assert rows[_chi2_col("fit0")].tolist() == ["0.500", "1.500", "1.100", "1.100"]
    assert rows[_chi2_col("fit1")].tolist() == ["1.000"] * 4


def test_comparison_table_leaves_a_dataset_a_fit_lacks_empty() -> None:
    table = _table(
        [
            {"A": ("Top", 4, 2.0), "B": ("Top", 2, 2.0)},
            {"A": ("Top", 4, 4.0)},
        ]
    )

    assert _at(table, ("Top", "B"), _NDAT_COL) == "2"
    assert _at(table, ("Top", "B"), _chi2_col("fit1")) == ""
    # each fit's total is over the datasets it has
    assert _at(table, ("Top", _TOTAL), _NDAT_COL) == "6"
    assert _at(table, ("Top", _TOTAL), _chi2_col("fit0")) == "0.667"
    assert _at(table, ("Top", _TOTAL), _chi2_col("fit1")) == "1.000"


def test_comparison_table_leaves_the_total_of_a_fit_without_the_group_empty() -> None:
    table = _table([{"A": ("Top", 4, 2.0)}, {"B": ("Higgs", 4, 2.0)}])
    assert _at(table, ("Top", _TOTAL), _chi2_col("fit1")) == ""
    assert _at(table, ("Higgs", _TOTAL), _chi2_col("fit0")) == ""


def test_comparison_table_overall_total_is_over_every_dataset_of_each_fit() -> None:
    table = _table(
        [
            {"A": ("Top", 4, 2.0), "B": ("Higgs", 6, 9.0)},
            {"A": ("Top", 4, 4.0), "C": (UNGROUPED, 10, 5.0)},
        ]
    )
    total = (_ALL, _TOTAL)
    assert _at(table, total, _NDAT_COL) == "20"
    assert _at(table, total, _chi2_col("fit0")) == "1.100"
    assert _at(table, total, _chi2_col("fit1")) == "0.643"


def test_comparison_table_overall_total_counts_a_dataset_grouped_twice_once() -> None:
    table = _table([{"A": ("Top", 4, 2.0)}, {"A": ("Higgs", 4, 2.0)}])
    assert {("Top", "A"), ("Higgs", "A")} <= set(table.index.tolist())
    assert _at(table, (_ALL, _TOTAL), _NDAT_COL) == "4"
    assert _at(table, (_ALL, _TOTAL), _chi2_col("fit0")) == "0.500"


def test_comparison_table_keeps_rows_named_like_the_totals() -> None:
    table = _table([{"Total": ("All", 4, 2.0), "B": ("Top", 6, 3.0)}])
    assert table.index.tolist() == [
        ("All", "Total"),
        ("All", _TOTAL),
        _BLANK,
        ("Top", "B"),
        ("Top", _TOTAL),
        _BLANK,
        (_ALL, _TOTAL),
    ]
    assert table[_NDAT_COL].tolist() == ["4", "4", "", "6", "6", "", "10"]


def test_comparison_table_rejects_fits_disagreeing_on_ndat() -> None:
    with pytest.raises(ValueError, match=r"'A' \(4, 5\)"):
        _table([{"A": ("Top", 4, 2.0)}, {"A": ("Top", 5, 2.0)}])


def test_comparison_table_has_no_ratio_for_a_source_without_data_points() -> None:
    table = _table([{"ext": (UNGROUPED, 0, 2.0)}])
    assert _at(table, (UNGROUPED, "ext"), _NDAT_COL) == "0"
    assert _at(table, (UNGROUPED, "ext"), _chi2_col("fit0")) == ""


def test_comparison_table_rejects_fits_sharing_a_label() -> None:
    with pytest.raises(ValueError, match="share the column label"):
        _table([{"A": ("G", 1, 1.0)}, {"A": ("G", 1, 1.0)}], labels=["x", "x"])


# ---------------------------------------------------------------------------
# Through reportengine, on fits written to disk
# ---------------------------------------------------------------------------


def _write_fit(
    fit_dir: Path, data_path: Path, theory_path: Path, group: str | None
) -> dict[str, Any]:
    dataset: dict[str, Any] = {"name": "TESTDATA", "order": "NLO"}
    if group is not None:
        dataset["group"] = group
    runcard = {
        "data_path": str(data_path),
        "theory_path": str(theory_path),
        "use_theory_covmat": False,
        "use_t0": False,
        "use_quad": True,
        "datasets": [dataset],
        "coefficients": {
            name: {"free": True, "prior": {"dist": "uniform", "low": -5, "high": 5}}
            for name in _BEST_FIT
        },
        "actions_": ["run_analytic_fit"],
    }
    (fit_dir / "input").mkdir(parents=True)
    (fit_dir / "input" / "runcard.yaml").write_text(yaml.dump(runcard))
    FitResult(
        free_parameters=list(_BEST_FIT),
        best_fit_point=_BEST_FIT,
        max_loglikelihood=-1.0,
        num_data=3,
    ).write(fit_dir)
    return runcard


def test_comparison_table_resolves_from_fits_alone_and_uses_each_fit_setup(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _write_fit(
        tmp_path / "grouped", fixtures_data_path, fixtures_theory_path, "Top"
    )
    _write_fit(tmp_path / "plain", fixtures_data_path, fixtures_theory_path, None)
    fits = [
        {"name": "grouped", "path": str(tmp_path), "label": "G"},
        {"name": "plain", "path": str(tmp_path), "label": "P"},
    ]
    config = smefitConfig({"fits": fits}, environment=smefitAPI.loadedenv)
    builder = ResourceBuilder(
        config,
        smefitAPI.provider_loaded,
        [FuzzyTarget("datasets_chi2_comparison_table", (), (), ())],
        perform_final=False,
    )
    builder.rootns.update(smefitAPI.loadedenv.ns_dump())
    builder.resolve_fuzzytargets()
    builder.execute_sequential()
    table = builder.rootns["datasets_chi2_comparison_table"]

    [chi2] = smefitAPI.datasets_chi2(**runcard)
    expected = f"{float(chi2(jnp.array(list(_BEST_FIT.values())))) / chi2.num_data:.3f}"
    top = (r"$\mathrm{Top}$", _TOTAL)
    assert _at(table, top, _chi2_col("G")) == expected
    assert _at(table, top, _chi2_col("P")) == ""
    other = (rf"$\mathrm{{{UNGROUPED}}}$", _TOTAL)
    assert _at(table, other, _chi2_col("P")) == expected
    assert _at(table, other, _chi2_col("G")) == ""
    assert _at(table, (_ALL, _TOTAL), _NDAT_COL) == str(chi2.num_data)
    assert _at(table, (_ALL, _TOTAL), _chi2_col("G")) == expected
    assert _at(table, (_ALL, _TOTAL), _chi2_col("P")) == expected
