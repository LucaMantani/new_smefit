"""Tests of ``produce_fitcontext``: the setup of a fit already run, as a namespace.

Every target is resolved under ``fits::fitcontext`` by a real ResourceBuilder,
as a runcard's ``- fits::fitcontext <action>`` or ``{@with fits::fitcontext@}``
does, so the namespace lookup reportengine does is part of what is tested.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from reportengine import namespaces
from reportengine.configparser import ConfigError
from reportengine.resourcebuilder import FuzzyTarget, ResourceBuilder

from smefit.api import smefitAPI
from smefit.config import smefitConfig
from smefit.fit_result import Fit, FitResult, FitResultGroup

_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}
_BEST_FIT = {"OpA": 0.02, "OpB": -0.01}


def _fit_runcard(
    data_path: Path,
    theory_path: Path,
    group: str | None = None,
    coefficients: tuple[str, ...] = tuple(_BEST_FIT),
) -> dict[str, Any]:
    dataset: dict[str, Any] = {"name": "TESTDATA", "order": "NLO"}
    if group is not None:
        dataset["group"] = group
    return {
        "data_path": str(data_path),
        "theory_path": str(theory_path),
        "use_theory_covmat": False,
        "use_t0": False,
        "use_quad": True,
        "datasets": [dataset],
        "coefficients": {
            name: {"free": True, "prior": _PRIOR} for name in coefficients
        },
        "actions_": ["run_analytic_fit"],
    }


def _write_fit(
    fit_dir: Path, runcard: dict[str, Any], best_fit: dict[str, float] = _BEST_FIT
) -> dict[str, str]:
    """A fit directory as a smefit run leaves it; returns its ``fits:`` entry."""
    (fit_dir / "input").mkdir(parents=True)
    (fit_dir / "input" / "runcard.yaml").write_text(yaml.dump(runcard))
    FitResult(
        free_parameters=list(best_fit),
        best_fit_point=best_fit,
        max_loglikelihood=-1.0,
        num_data=3,
    ).write(fit_dir)
    return {"name": fit_dir.name, "path": str(fit_dir.parent)}


def _in_fitcontext(target: str, fits: list[dict[str, str]], **keys: Any) -> list[Any]:
    """The value of ``target`` under ``fits::fitcontext``, for each fit."""
    config = smefitConfig({"fits": fits, **keys}, environment=smefitAPI.loadedenv)
    builder = ResourceBuilder(
        config,
        smefitAPI.provider_loaded,
        [FuzzyTarget(target, ("fits", "fitcontext"), (), ())],
        perform_final=False,
    )
    builder.rootns.update(smefitAPI.loadedenv.ns_dump())
    builder.resolve_fuzzytargets()
    builder.execute_sequential()
    return [
        namespaces.resolve(builder.rootns, [("fits", index), "fitcontext"])[target]
        for index in range(len(fits))
    ]


def _hessian_at(chi2: Any, point: dict[str, float]) -> np.ndarray:
    return np.asarray(0.5 * jax.hessian(chi2)(jnp.array(list(point.values()))))


# ---------------------------------------------------------------------------
# The likelihood is the fit's, evaluated at its stored best fit
# ---------------------------------------------------------------------------


def test_fisher_is_the_hessian_of_the_fit_runcard_at_its_best_fit(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    [result] = _in_fitcontext("aggregate_fisher_information_matrices", fits)

    chi2 = smefitAPI.datasets_chi2(**runcard)[0]
    assert list(result) == ["TESTDATA"]
    assert result["TESTDATA"].index.tolist() == list(_BEST_FIT)
    np.testing.assert_allclose(result["TESTDATA"].values, _hessian_at(chi2, _BEST_FIT))
    # with quadratics the point matters: the test would pass on the SM otherwise
    at_sm = _hessian_at(chi2, dict.fromkeys(_BEST_FIT, 0.0))
    assert not np.allclose(result["TESTDATA"].values, at_sm)


def test_total_fisher_is_the_hessian_of_the_fit_runcard_at_its_best_fit(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    [result] = _in_fitcontext("total_fisher_information_matrix", fits)

    expected = _hessian_at(smefitAPI.chi2(**runcard), _BEST_FIT)
    assert result.index.tolist() == list(_BEST_FIT)
    np.testing.assert_allclose(result.values, expected)


def test_groups_come_from_the_fit_runcard(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path, group="Top")
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    [result] = _in_fitcontext("aggregate_fisher_information_matrices", fits)

    assert list(result) == ["Top"]


def test_each_fit_gets_its_own_setup(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """Data, coefficients and best fit all differ between the two fits.

    The heavy nodes are cached on the config: with one value per resource the
    second fit would be handed the first one's. With quadratics the Hessian
    depends on the data, through the residual.
    """
    shifted = tmp_path / "shifted_commondata"
    shifted.mkdir()
    dataset = yaml.safe_load((fixtures_data_path / "TESTDATA.yaml").read_text())
    dataset["data_central"] = [3.0, 1.0, 5.0]
    (shifted / "TESTDATA.yaml").write_text(yaml.dump(dataset))

    first = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    second = _fit_runcard(shifted, fixtures_theory_path, coefficients=("OpA",))
    points = [_BEST_FIT, {"OpA": -0.03}]
    fits = [
        _write_fit(tmp_path / "first", first, points[0]),
        _write_fit(tmp_path / "second", second, points[1]),
    ]

    results = _in_fitcontext("total_fisher_information_matrix", fits)

    for result, runcard, point in zip(results, (first, second), points):
        assert result.index.tolist() == list(point)
        expected = _hessian_at(smefitAPI.chi2(**runcard), point)
        np.testing.assert_allclose(result.values, expected)


def test_the_fit_runcard_record_is_untouched(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    entry = _write_fit(tmp_path / "my_fit", runcard)

    _in_fitcontext("total_fisher_information_matrix", [entry])

    assert Fit.from_folder(tmp_path / "my_fit").fit_runcard == runcard


def test_best_fit_point_is_the_stored_one(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    [result] = _in_fitcontext("best_fit_point", fits)

    np.testing.assert_allclose(result, list(_BEST_FIT.values()))


def test_gd_best_fit_is_left_to_the_gradient_descent(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """Nodes that need an actual descent are not handed the stored point."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    [result] = _in_fitcontext(
        "gd_best_fit", fits, gradient_descent_settings={"sm_solution": True}
    )

    np.testing.assert_allclose(result, [0.0, 0.0])


def test_without_fitcontext_best_fit_point_is_gd_best_fit(
    fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    settings = {"gradient_descent_settings": {}}

    np.testing.assert_allclose(
        smefitAPI.best_fit_point(**runcard, **settings),
        smefitAPI.gd_best_fit(**runcard, **settings),
    )


# ---------------------------------------------------------------------------
# The runcard's own keys
# ---------------------------------------------------------------------------


def test_a_setup_key_the_fit_lacks_is_refused_from_the_runcard(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """It would otherwise stand in for the fit's own, which is the default."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    del runcard["use_t0"]
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    with pytest.raises(ConfigError, match="use_t0"):
        _in_fitcontext("total_fisher_information_matrix", fits, use_t0=True)


def test_report_keys_stay_the_runcard_own(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """Only setup keys are taken from the fit: how to report is the runcard's."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    runcard["params_to_plot"] = ["OpA"]
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    [result] = _in_fitcontext("params_to_plot", fits, params_to_plot=["OpB"])

    assert result == ["OpB"]


# ---------------------------------------------------------------------------
# External likelihoods
# ---------------------------------------------------------------------------

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


def _external_entry(tmp_path: Path, group: str | None = None) -> dict[str, Any]:
    module = tmp_path / "ext_chi2_for_fitcontext.py"
    module.write_text(_EXTERNAL_CHI2)
    entry: dict[str, Any] = {"path": str(module), "curvature": 2.0}
    if group is not None:
        entry["group"] = group
    return {"ExtChi2": entry}


def test_an_external_chi2_merges_into_its_group(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """A group shared by a dataset and an external chi2 is one source."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path, group="Top")
    runcard["external_chi2"] = _external_entry(tmp_path, group="Top")
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    [result] = _in_fitcontext("aggregate_fisher_information_matrices", fits)

    assert list(result) == ["Top"]
    data_chi2 = smefitAPI.datasets_chi2(**runcard)[0]
    # F = 0.5 * H of curvature * c^2 is curvature on the diagonal
    expected = _hessian_at(data_chi2, _BEST_FIT) + 2.0 * np.eye(2)
    np.testing.assert_allclose(result["Top"].values, expected)


def test_an_external_chi2_keeps_its_own_group(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path, group="Top")
    runcard["external_chi2"] = _external_entry(tmp_path, group="DY")
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    [result] = _in_fitcontext("aggregate_fisher_information_matrices", fits)

    assert sorted(result) == ["DY", "Top"]


def test_an_external_only_fit_uses_its_groups(tmp_path: Path) -> None:
    """No datasets: the groups come from the external chi2s alone."""
    runcard = {
        "external_chi2": _external_entry(tmp_path, group="DY"),
        "coefficients": {name: {"free": True, "prior": _PRIOR} for name in _BEST_FIT},
        "actions_": ["run_analytic_fit"],
    }
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    [result] = _in_fitcontext("aggregate_fisher_information_matrices", fits)

    assert list(result) == ["DY"]
    np.testing.assert_allclose(result["DY"].values, 2.0 * np.eye(2))


# ---------------------------------------------------------------------------
# Fits whose stored runcard does not describe the joint likelihood
# ---------------------------------------------------------------------------


def test_an_individual_fit_is_refused(
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
    fits = [{"name": "my_fit", "path": str(tmp_path)}]

    with pytest.raises(ConfigError, match="individual fits"):
        _in_fitcontext("total_fisher_information_matrix", fits)


def test_a_bayesian_update_warns_it_covers_only_the_update(
    tmp_path: Path,
    fixtures_data_path: Path,
    fixtures_theory_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    previous = _write_fit(tmp_path / "previous_fit", runcard)
    runcard["bayesian_update"] = previous
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        [result] = _in_fitcontext("aggregate_fisher_information_matrices", fits)

    assert "Bayesian update" in caplog.text
    assert list(result) == ["TESTDATA"]


def test_an_ordinary_fit_does_not_warn(
    tmp_path: Path,
    fixtures_data_path: Path,
    fixtures_theory_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        _in_fitcontext("total_fisher_information_matrix", fits)

    assert "Bayesian update" not in caplog.text


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def test_pca_needs_no_setup_of_its_own(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """Only ``fits`` and the required ``pca_settings``: the rest is the fit's."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    # at the minimum, as a real fit is: PCA refuses negative curvature
    minimum = smefitAPI.gd_best_fit(**runcard, gradient_descent_settings={})
    point = dict(zip(_BEST_FIT, map(float, minimum)))
    fits = [_write_fit(tmp_path / "my_fit", runcard, point)]

    [result] = _in_fitcontext("pca", fits, pca_settings={})

    fisher = _hessian_at(smefitAPI.chi2(**runcard), point)
    assert result.coeff_names == list(_BEST_FIT)
    np.testing.assert_allclose(np.sort(result.eigenvalues), np.linalg.eigvalsh(fisher))


def test_pca_off_the_minimum_explains_a_stored_best_fit(
    tmp_path: Path, fixtures_data_path: Path, fixtures_theory_path: Path
) -> None:
    """The stored best fit of a sampler fit need not be a minimum: say so."""
    runcard = _fit_runcard(fixtures_data_path, fixtures_theory_path)
    fits = [_write_fit(tmp_path / "my_fit", runcard)]

    with pytest.raises(ValueError, match="stored best fit is its best sample"):
        _in_fitcontext("pca", fits, pca_settings={})
