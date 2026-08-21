"""Unit tests for smefit.fit_result — FitResult, FitResultGroup and Fit."""

import json
import math

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from smefit.fit_result import Fit, FitResult, FitResultGroup


def _make_result(
    free=("OpA", "OpB"),
    best_fit=None,
    max_loglikelihood=-5.0,
    num_data=10,
    samples=None,
):
    if best_fit is None:
        best_fit = {name: 1.0 for name in free}
    return FitResult(
        free_parameters=list(free),
        best_fit_point=best_fit,
        max_loglikelihood=max_loglikelihood,
        num_data=num_data,
        samples=samples,
    )


# ---------------------------------------------------------------------------
# Derived scalar properties
# ---------------------------------------------------------------------------


def test_n_free():
    fr = _make_result(free=["OpA", "OpB"])
    assert fr.n_free == 2


def test_ndof():
    fr = _make_result(free=["OpA", "OpB"], num_data=10)
    assert fr.ndof == 8


def test_chi2_val():
    fr = _make_result(max_loglikelihood=-5.0)
    assert fr.chi2_val == pytest.approx(10.0)


def test_chi2_ndof():
    fr = _make_result(free=["OpA", "OpB"], num_data=10, max_loglikelihood=-5.0)
    # ndof = 8, chi2 = 10
    assert fr.chi2_ndof == pytest.approx(10.0 / 8.0)


def test_chi2_ndof_zero_dof():
    # n_free == num_data → ndof = 0 → should return nan
    fr = _make_result(free=["OpA", "OpB"], num_data=2, max_loglikelihood=-5.0)
    assert math.isnan(fr.chi2_ndof)


def test_bic():
    fr = _make_result(free=["OpA", "OpB"], num_data=10, max_loglikelihood=-5.0)
    expected = 2 * math.log(10) - 2 * (-5.0)
    assert fr.bic == pytest.approx(expected)


def test_aic():
    fr = _make_result(free=["OpA", "OpB"], max_loglikelihood=-5.0)
    expected = 2 * 2 - 2 * (-5.0)
    assert fr.aic == pytest.approx(expected)


# ---------------------------------------------------------------------------
# std property
# ---------------------------------------------------------------------------


def test_std_no_samples():
    fr = _make_result(samples=None)
    assert fr.std == {}


def test_std_with_samples():
    samples = {
        "OpA": jnp.array([1.0, 1.0, 1.0]),  # constant → std = 0
        "OpB": jnp.array([0.0, 1.0, 2.0]),  # std = sqrt(2/3)
    }
    fr = _make_result(free=["OpA", "OpB"], samples=samples)
    std = fr.std
    assert std["OpA"] == pytest.approx(0.0, abs=1e-6)
    assert std["OpB"] == pytest.approx(
        float(jnp.std(jnp.array([0.0, 1.0, 2.0]))), rel=1e-5
    )


# ---------------------------------------------------------------------------
# correlations property
# ---------------------------------------------------------------------------


def test_correlations_no_samples():
    fr = _make_result(samples=None)
    with pytest.raises(ValueError, match="no posterior samples"):
        fr.correlations


def test_correlations_of_perfectly_dependent_samples():
    """A parameter correlates perfectly with itself, and with a copy of itself
    either way round."""
    samples = {
        "OpA": jnp.array([0.0, 1.0, 2.0, 3.0]),
        "OpB": jnp.array([0.0, 1.0, 2.0, 3.0]),
        "OpC": jnp.array([0.0, -1.0, -2.0, -3.0]),
    }
    fr = _make_result(free=["OpA", "OpB", "OpC"], samples=samples)

    corr = fr.correlations
    assert corr.loc["OpA", "OpA"] == pytest.approx(1.0, abs=1e-5)
    assert corr.loc["OpA", "OpB"] == pytest.approx(1.0, abs=1e-5)
    assert corr.loc["OpA", "OpC"] == pytest.approx(-1.0, abs=1e-5)


def test_correlations_covers_the_free_parameters_in_order():
    """Derived coefficients are in `samples` too, and are left out: they are
    functions of the free ones."""
    samples = {
        "OpB": jnp.array([0.0, 1.0, 2.0]),
        "OpA": jnp.array([2.0, 0.0, 1.0]),
        "OpDerived": jnp.array([0.0, 1.0, 4.0]),
    }
    fr = _make_result(free=["OpB", "OpA"], samples=samples)

    corr = fr.correlations
    assert corr.index.tolist() == ["OpB", "OpA"]
    assert corr.columns.tolist() == ["OpB", "OpA"]


def test_correlations_is_symmetric():
    samples = {
        "OpA": jnp.array([0.0, 1.0, 2.0, 5.0]),
        "OpB": jnp.array([2.0, 0.0, 1.0, 1.5]),
    }
    fr = _make_result(free=["OpA", "OpB"], samples=samples)

    corr = fr.correlations
    assert corr.loc["OpA", "OpB"] == pytest.approx(corr.loc["OpB", "OpA"])


def test_correlations_of_a_single_free_parameter_is_one_by_one():
    """corrcoef gives a scalar for one variable; it is still a 1x1 matrix."""
    fr = _make_result(free=["OpA"], samples={"OpA": jnp.array([0.0, 1.0, 2.0])})

    corr = fr.correlations
    assert corr.shape == (1, 1)
    assert corr.loc["OpA", "OpA"] == pytest.approx(1.0, abs=1e-5)


def test_correlations_of_a_frozen_parameter_are_nan():
    """A parameter whose samples never moved correlates with nothing, and says
    so rather than failing the whole fit's correlations."""
    samples = {
        "OpA": jnp.array([0.0, 1.0, 2.0]),
        "OpB": jnp.array([1.0, 1.0, 1.0]),
    }
    fr = _make_result(free=["OpA", "OpB"], samples=samples)

    corr = fr.correlations
    assert corr.loc["OpA", "OpA"] == pytest.approx(1.0, abs=1e-5)
    assert math.isnan(corr.loc["OpA", "OpB"])
    assert math.isnan(corr.loc["OpB", "OpB"])


# ---------------------------------------------------------------------------
# write / JSON round-trip
# ---------------------------------------------------------------------------


def test_write_json_roundtrip(tmp_path):
    samples = {"OpA": jnp.array([1.0, 2.0, 3.0]), "OpB": jnp.array([-1.0, 0.0, 1.0])}
    fr = FitResult(
        free_parameters=["OpA", "OpB"],
        best_fit_point={"OpA": 1.5, "OpB": 0.0},
        max_loglikelihood=-3.0,
        num_data=8,
        samples=samples,
    )

    fr.write(tmp_path)

    out_file = tmp_path / "fit_results.json"
    assert out_file.exists()

    with out_file.open() as f:
        payload = json.load(f)

    assert payload["free_parameters"] == ["OpA", "OpB"]
    assert payload["num_data"] == 8
    assert payload["n_free"] == 2
    assert payload["ndof"] == 6
    assert payload["max_loglikelihood"] == pytest.approx(-3.0)
    assert payload["chi2"] == pytest.approx(6.0)
    assert payload["best_fit_point"]["OpA"] == pytest.approx(1.5)
    assert payload["samples"]["OpA"] == pytest.approx([1.0, 2.0, 3.0])


def test_from_json_names_an_unparsable_file(tmp_path):
    """from_json reads a fit_results.json on its own, without the rest of the
    fit directory, so it must fail as informatively as Fit.from_folder does."""
    (tmp_path / "fit_results.json").write_text("{not json")

    with pytest.raises(ValueError, match=r"fit_results\.json' is not valid JSON"):
        FitResult.from_json(tmp_path)


# ---------------------------------------------------------------------------
# FitResultGroup.write_summary
# ---------------------------------------------------------------------------


def _make_individual_result(
    name, best_val, samples_vals, max_loglikelihood=-5.0, num_data=10
):
    """Build a single-free-parameter result as produced by an individual fit."""
    return FitResult(
        free_parameters=[name],
        best_fit_point={name: best_val},
        max_loglikelihood=max_loglikelihood,
        num_data=num_data,
        samples={name: jnp.array(samples_vals)},
    )


def test_write_summary_no_overwrite(tmp_path):
    """Each coefficient's best_fit and std must come from its own fit, not be
    overwritten by later fits."""
    r1 = _make_individual_result("OpA", best_val=1.0, samples_vals=[0.8, 1.0, 1.2])
    r2 = _make_individual_result("OpB", best_val=2.0, samples_vals=[1.8, 2.0, 2.2])
    group = FitResultGroup([r1, r2])
    group.write_summary(tmp_path)

    with (tmp_path / "fit_results.json").open() as f:
        payload = json.load(f)

    assert payload["best_fit_point"]["OpA"] == pytest.approx(1.0)
    assert payload["best_fit_point"]["OpB"] == pytest.approx(2.0)
    assert payload["std"]["OpA"] == pytest.approx(
        float(jnp.std(jnp.array([0.8, 1.0, 1.2]))), rel=1e-5
    )
    assert payload["std"]["OpB"] == pytest.approx(
        float(jnp.std(jnp.array([1.8, 2.0, 2.2]))), rel=1e-5
    )
    assert payload["samples"]["OpA"] == pytest.approx([0.8, 1.0, 1.2])
    assert payload["samples"]["OpB"] == pytest.approx([1.8, 2.0, 2.2])


def test_write_summary_metadata(tmp_path):
    """Summary JSON has correct aggregated metadata."""
    r1 = _make_individual_result(
        "OpA", best_val=1.0, samples_vals=[1.0], max_loglikelihood=-3.0, num_data=10
    )
    r2 = _make_individual_result(
        "OpB", best_val=2.0, samples_vals=[2.0], max_loglikelihood=-7.0, num_data=10
    )
    group = FitResultGroup([r1, r2])
    group.write_summary(tmp_path)

    with (tmp_path / "fit_results.json").open() as f:
        payload = json.load(f)

    assert payload["free_parameters"] == ["OpA", "OpB"]
    assert payload["n_free"] == 2
    assert payload["num_data"] == 10
    assert payload["chi2"]["OpA"] == pytest.approx(6.0)
    assert payload["chi2"]["OpB"] == pytest.approx(14.0)


def test_write_summary_round_trips_through_from_payload(tmp_path):
    """The group reads back the schema it writes, without going through Fit."""
    r1 = _make_individual_result(
        "OpA", best_val=1.0, samples_vals=[0.8, 1.0, 1.2], max_loglikelihood=-3.0
    )
    r2 = _make_individual_result(
        "OpB", best_val=2.0, samples_vals=[1.8, 2.0, 2.2], max_loglikelihood=-7.0
    )
    FitResultGroup([r1, r2]).write_summary(tmp_path)

    with (tmp_path / "fit_results.json").open() as f:
        payload = json.load(f)
    recovered = FitResultGroup.from_payload(payload)

    assert isinstance(recovered, FitResultGroup)
    assert [r.free_parameters[0] for r in recovered.results] == ["OpA", "OpB"]
    assert [r.chi2_val for r in recovered.results] == pytest.approx([6.0, 14.0])
    assert recovered.results[0].best_fit_point["OpA"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fit: the FitResult of a fit on disk, plus the metadata describing it
# ---------------------------------------------------------------------------


def _make_written_result(**kwargs):
    """The FitResult a fitting routine produces, ready to be written."""
    defaults = dict(
        free_parameters=["OpA", "OpB"],
        best_fit_point={"OpA": 1.5, "OpB": 0.0},
        max_loglikelihood=-3.0,
        num_data=8,
    )
    defaults.update(kwargs)
    return FitResult(**defaults)


def test_fit_holds_its_fit_result():
    """A Fit is not a FitResult: it has one."""
    result = _make_written_result()
    fit = Fit(fit_results=result, fit_name="my_fit")

    assert not isinstance(fit, FitResult)
    assert fit.fit_results is result
    assert fit.fit_results.ndof == 6
    assert fit.fit_results.chi2_val == pytest.approx(6.0)


def test_fit_is_named_by_its_fit_name():
    """str(fit) is the fit name: reportengine builds the file name of a per-fit
    figure or table out of it, and the dataclass repr would be unusable there.
    """
    fit = Fit(fit_results=_make_written_result(), fit_name="my_fit")
    assert str(fit) == "my_fit"


def test_plot_label_prefers_the_label_and_falls_back_to_the_name():
    """A plot has to say which fit it is drawn from; the runcard's label is how
    the fit is presented, and the name is the fallback when it gave none."""
    result = _make_written_result()
    assert Fit(fit_results=result, fit_name="my_fit").plot_label == "my_fit"
    assert (
        Fit(fit_results=result, fit_name="my_fit", label=r"$\mathrm{Mine}$").plot_label
        == r"$\mathrm{Mine}$"
    )


def test_fit_metadata_defaults():
    fit = Fit(fit_results=_make_written_result(), fit_name="my_fit")
    assert fit.fit_name == "my_fit"
    assert fit.fit_type is None
    assert fit.use_quad is False
    assert fit.individual_fit is False


def test_label_is_given_by_the_caller_not_the_fit_directory(tmp_path):
    """A label is how a runcard presents a fit; nothing on disk records it."""
    out = _write_runcard(tmp_path / "my_fit")
    _make_written_result().write(out)

    assert Fit.from_folder(out).label is None
    assert (
        Fit.from_folder(out, label=r"$\mathrm{My\ fit}$").label == r"$\mathrm{My\ fit}$"
    )


def _write_summary(path):
    """The ``fit_results.json`` a ``run_individual_*_fits`` run leaves behind."""
    r1 = _make_individual_result("OpA", best_val=1.0, samples_vals=[0.8, 1.0, 1.2])
    r2 = _make_individual_result("OpB", best_val=2.0, samples_vals=[1.8, 2.0, 2.2])
    FitResultGroup([r1, r2]).write_summary(path)
    return path


def _write_payload_of(action, path):
    """The payload *action* would have written: a summary when it ran individually.

    A fit directory whose runcard and results disagree is rejected, so a test
    that is not about that must write the payload its action implies.
    """
    if action.startswith("run_individual_"):
        return _write_summary(path)
    _make_written_result().write(path)
    return path


def _write_runcard(path, use_quad=False, action="run_analytic_fit"):
    """The runcard copy a smefit run leaves in the fit directory."""
    (path / "input").mkdir(parents=True, exist_ok=True)
    config = {"use_quad": use_quad}
    if action is not None:
        config["actions_"] = [action]
    (path / "input" / "runcard.yaml").write_text(yaml.dump(config))
    return path


def test_written_payload_carries_no_metadata(tmp_path):
    """A fit's json holds the numbers it produced, not how it was configured."""
    _make_written_result().write(tmp_path)

    with (tmp_path / "fit_results.json").open() as f:
        payload = json.load(f)

    assert "use_quad" not in payload
    assert "fit_type" not in payload
    assert "individual_fit" not in payload
    assert payload["max_loglikelihood"] == pytest.approx(-3.0)


def test_fit_from_folder_roundtrip(tmp_path):
    out = _write_runcard(tmp_path / "my_fit", use_quad=True)
    _make_written_result(
        samples={"OpA": jnp.array([1.0, 2.0]), "OpB": jnp.array([0.0, 1.0])}
    ).write(out)

    recovered = Fit.from_folder(out)

    assert isinstance(recovered, Fit)
    # the directory name is the identity of a fit on disk
    assert recovered.fit_name == "my_fit"
    assert isinstance(recovered.fit_results, FitResult)
    assert recovered.fit_results.max_loglikelihood == pytest.approx(-3.0)
    assert list(recovered.fit_results.samples["OpA"]) == pytest.approx([1.0, 2.0])


# ---------------------------------------------------------------------------
# Fit metadata: read from the runcard, never from the payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_quad", [True, False])
def test_use_quad_comes_from_the_runcard(tmp_path, use_quad):
    out = _write_runcard(tmp_path / "my_fit", use_quad=use_quad)
    _make_written_result().write(out)

    assert Fit.from_folder(out).use_quad is use_quad


def test_use_quad_ignores_a_stale_payload_entry(tmp_path):
    """Fits written when use_quad was serialised must not be believed."""
    out = _write_runcard(tmp_path / "my_fit", use_quad=True)
    _make_written_result().write(out)
    with (out / "fit_results.json").open() as f:
        payload = json.load(f)
    payload["use_quad"] = False
    (out / "fit_results.json").write_text(json.dumps(payload))

    assert Fit.from_folder(out).use_quad is True


@pytest.mark.parametrize(
    "action, fit_type",
    [
        ("run_analytic_fit", "analytic"),
        ("run_hessian_fit", "hessian"),
        ("run_ultranest_fit", "ultranest"),
        ("run_blackjax_fit", "blackjax"),
        ("run_individual_analytic_fits", "analytic"),
        ("run_individual_blackjax_fits", "blackjax"),
    ],
)
def test_fit_type_comes_from_the_runcard_action(tmp_path, action, fit_type):
    out = _write_runcard(tmp_path / "my_fit", action=action)
    _write_payload_of(action, out)

    assert Fit.from_folder(out).fit_type == fit_type


def test_fit_type_ignores_the_arguments_of_an_action(tmp_path):
    out = _write_runcard(tmp_path / "my_fit", action="run_analytic_fit(main=True)")
    _make_written_result().write(out)

    assert Fit.from_folder(out).fit_type == "analytic"


def test_fit_type_looks_past_a_nested_namespace_entry(tmp_path):
    """An ``actions_`` entry that nests actions under a namespace is a mapping."""
    out = tmp_path / "my_fit"
    (out / "input").mkdir(parents=True)
    (out / "input" / "runcard.yaml").write_text(
        yaml.dump({"actions_": [{"scan": ["plot_chi2"]}, "run_hessian_fit"]})
    )
    _make_written_result().write(out)

    assert Fit.from_folder(out).fit_type == "hessian"


def test_fit_type_is_unset_when_no_fit_action_ran(tmp_path, caplog):
    out = _write_runcard(tmp_path / "my_fit", action="report")
    _make_written_result().write(out)

    assert Fit.from_folder(out).fit_type is None
    assert "no known fit action" in caplog.text


def test_individual_fit_comes_from_the_runcard_action(tmp_path):
    out = _write_runcard(tmp_path / "my_fit", action="run_individual_ultranest_fits")
    _write_summary(out)

    assert Fit.from_folder(out).individual_fit is True


def test_a_joint_action_is_not_an_individual_fit(tmp_path):
    out = _write_runcard(tmp_path / "my_fit", action="run_ultranest_fit")
    _make_written_result().write(out)

    assert Fit.from_folder(out).individual_fit is False


def test_a_fit_without_a_runcard_cannot_be_loaded(tmp_path):
    """Numbers alone are not a fit: how they were produced is required."""
    out = tmp_path / "no_runcard"
    _make_written_result().write(out)

    with pytest.raises(FileNotFoundError, match="runcard.yaml"):
        Fit.from_folder(out)


def test_a_fit_without_results_cannot_be_loaded(tmp_path):
    out = _write_runcard(tmp_path / "no_results")

    with pytest.raises(FileNotFoundError, match="fit_results.json"):
        Fit.from_folder(out)


def test_an_unparsable_payload_names_the_file_at_fault(tmp_path):
    """A decoder locates the fault within a file; from_folder names the file.

    Two are read, so a bare "line 1 column 2" would not say which.
    """
    out = _write_runcard(tmp_path / "bad_json")
    (out / "fit_results.json").write_text("{not json")

    with pytest.raises(ValueError, match=r"fit_results\.json' is not valid JSON"):
        Fit.from_folder(out)


def test_an_unparsable_runcard_names_the_file_at_fault(tmp_path):
    """A truncated runcard is a broken fit directory, not a PyYAML traceback."""
    out = _write_runcard(tmp_path / "bad_yaml")
    _make_written_result().write(out)
    (out / "input" / "runcard.yaml").write_text(
        "actions_: [run_analytic_fit\n bad: : :"
    )

    with pytest.raises(ValueError, match=r"runcard\.yaml' is not valid YAML"):
        Fit.from_folder(out)


def test_fit_from_folder_on_an_individual_fit_summary(tmp_path):
    """A summary is read back as the group of single-parameter fits it is."""
    _write_runcard(tmp_path, action="run_individual_analytic_fits")
    r1 = _make_individual_result(
        "OpA", best_val=1.0, samples_vals=[0.8, 1.0, 1.2], max_loglikelihood=-3.0
    )
    r2 = _make_individual_result(
        "OpB", best_val=2.0, samples_vals=[1.8, 2.0, 2.2], max_loglikelihood=-7.0
    )
    FitResultGroup([r1, r2]).write_summary(tmp_path)

    recovered = Fit.from_folder(tmp_path)

    assert recovered.individual_fit is True
    assert recovered.fit_type == "analytic"
    assert isinstance(recovered.fit_results, FitResultGroup)
    assert [r.free_parameters[0] for r in recovered.fit_results.results] == [
        "OpA",
        "OpB",
    ]


def test_an_individual_fit_keeps_every_coefficient_its_own_numbers(tmp_path):
    """Each result is a genuine one-parameter fit, not a slice of a joint one."""
    _write_runcard(tmp_path, action="run_individual_ultranest_fits")
    r1 = _make_individual_result(
        "OpA", best_val=1.0, samples_vals=[0.8, 1.0, 1.2], max_loglikelihood=-3.0
    )
    r1.logz = -8.0
    r2 = _make_individual_result(
        "OpB", best_val=2.0, samples_vals=[1.8, 2.0, 2.2], max_loglikelihood=-7.0
    )
    r2.logz = -9.0
    FitResultGroup([r1, r2]).write_summary(tmp_path)

    first, second = Fit.from_folder(tmp_path).fit_results.results

    assert first.chi2_val == pytest.approx(6.0)
    assert second.chi2_val == pytest.approx(14.0)
    assert (first.logz, second.logz) == (-8.0, -9.0)
    assert first.best_fit_point["OpA"] == pytest.approx(1.0)
    assert first.samples["OpA"] == pytest.approx([0.8, 1.0, 1.2])
    # one free parameter each: the ndof of a one-at-a-time fit, not of a joint one
    assert first.n_free == 1
    assert first.ndof == r1.ndof


def test_a_summary_payload_under_a_joint_action_is_rejected(tmp_path):
    """The action says how to read the payload; a payload that disagrees is an error.

    Nothing about how a fit was run is written to ``fit_results.json``, so its
    shape is never what decides — it is only checked against the action.
    """
    _write_runcard(tmp_path, action="run_analytic_fit")
    _write_summary(tmp_path)

    with pytest.raises(ValueError, match="inconsistent"):
        Fit.from_folder(tmp_path)


def test_a_joint_payload_under_an_individual_action_is_rejected(tmp_path):
    _write_runcard(tmp_path, action="run_individual_analytic_fits")
    _make_written_result().write(tmp_path)

    with pytest.raises(ValueError, match="inconsistent"):
        Fit.from_folder(tmp_path)


# ---------------------------------------------------------------------------
# FitResultGroup — the per-coefficient fields read under the FitResult names
# ---------------------------------------------------------------------------


def test_group_free_parameters_lists_every_coefficient_fitted():
    """Under the FitResult name, so that a routine reading one coefficient at
    a time — the bounds — reads both kinds of fit the same way."""
    r1 = _make_individual_result("OpA", best_val=1.0, samples_vals=[0.8, 1.2])
    r2 = _make_individual_result("OpB", best_val=2.0, samples_vals=[1.8, 2.2])

    assert FitResultGroup([r1, r2]).free_parameters == ["OpA", "OpB"]


def test_group_samples_merges_the_individual_posteriors():
    """Each coefficient's samples come from its own single-parameter fit."""
    r1 = _make_individual_result("OpA", best_val=1.0, samples_vals=[0.8, 1.2])
    r2 = _make_individual_result("OpB", best_val=2.0, samples_vals=[1.8, 2.2])

    samples = FitResultGroup([r1, r2]).samples

    assert list(samples) == ["OpA", "OpB"]
    assert samples["OpA"] == pytest.approx([0.8, 1.2])
    assert samples["OpB"] == pytest.approx([1.8, 2.2])


def test_group_samples_is_none_when_no_fit_kept_any():
    """Same absent-samples signal as FitResult, so consumers test it once."""
    result = _make_result(free=("OpA",), samples=None)

    assert FitResultGroup([result]).samples is None


# ---------------------------------------------------------------------------
# Fit.confidence_bounds / Fit.bounds
# ---------------------------------------------------------------------------


def _joint_fit(name, samples):
    """A joint fit as the bounds see one: samples for every free coefficient,
    drawn together."""
    return Fit(
        fit_results=_make_result(
            free=tuple(samples),
            samples={key: jnp.array(vals) for key, vals in samples.items()},
        ),
        fit_name=name,
    )


def _individual_fit(name, samples):
    """A one-coefficient-at-a-time fit: one single-parameter FitResult each,
    which is what a `fits:` entry pointing at an individual_fits output loads
    as."""
    return Fit(
        fit_results=FitResultGroup(
            [
                _make_individual_result(key, best_val=0.0, samples_vals=vals)
                for key, vals in samples.items()
            ]
        ),
        fit_name=name,
    )


# percentile p of 0..1000 is 10 * p, so the 68% interval is [160, 840], the
# 95% one [25, 975], around a mean of 500
_RAMP = list(range(0, 1001))


@pytest.fixture
def joint_fit():
    return _joint_fit("fit_a", {"OtG": _RAMP, "OpQM": [1.0, 2.0, 3.0]})


def test_confidence_bounds_are_equal_tailed_percentiles(joint_fit):
    """68% means 16/84 — the convention the old report pipeline used, and the
    one the tables have always quoted."""
    assert joint_fit.confidence_bounds(68)["OtG"] == pytest.approx(
        (160.0, 500.0, 840.0)
    )


def test_confidence_bounds_at_95_percent(joint_fit):
    assert joint_fit.confidence_bounds(95)["OtG"] == pytest.approx((25.0, 500.0, 975.0))


def test_confidence_bounds_central_value_is_the_mean():
    """Not the median: an asymmetric posterior has to report the same central
    value everywhere it is quoted."""
    fit = _joint_fit("fit", {"OtG": [0.0, 0.0, 0.0, 4.0]})

    assert fit.confidence_bounds(68)["OtG"][1] == pytest.approx(1.0)


def test_confidence_bounds_cover_every_free_coefficient_in_order(joint_fit):
    """Consumers iterate the result directly, so it keeps the fit's order."""
    assert list(joint_fit.confidence_bounds(68)) == ["OtG", "OpQM"]


def test_confidence_bounds_ignore_nans():
    """A sampler that wrote a NaN should shrink the sample, not the bounds to
    NaN."""
    with_nan = _joint_fit("fit", {"OtG": [0.0, 1.0, 2.0, math.nan]})
    without = _joint_fit("fit", {"OtG": [0.0, 1.0, 2.0]})

    assert with_nan.confidence_bounds(68)["OtG"] == pytest.approx(
        without.confidence_bounds(68)["OtG"]
    )


def test_confidence_bounds_reproduce_a_gaussian_sigma():
    """The percentile bounds of a Gaussian posterior are the familiar
    +-sigma."""
    rng = np.random.default_rng(0)
    fit = _joint_fit("fit", {"OtG": rng.normal(loc=2.0, scale=0.5, size=200_000)})

    low, mean, high = fit.confidence_bounds(68.27)["OtG"]

    assert mean == pytest.approx(2.0, abs=0.01)
    assert (high - low) / 2 == pytest.approx(0.5, rel=0.02)


@pytest.mark.parametrize("level", [0, 100, -5, 0.95, 120])
def test_confidence_bounds_reject_a_level_that_is_not_a_percentage(joint_fit, level):
    """`confidence_level: 0.95` is the plausible mistake, and it would
    silently give a 0.95% interval."""
    with pytest.raises(ValueError, match="between 1 and 100"):
        joint_fit.confidence_bounds(level)


def test_confidence_bounds_reject_a_fit_without_samples():
    """A Hessian fit that kept no posterior has nothing to take a percentile
    of, and the error has to name which fit that was."""
    fit = Fit(fit_results=_make_result(free=("OtG",)), fit_name="sample_less")

    with pytest.raises(ValueError, match="sample_less.*no posterior samples"):
        fit.confidence_bounds(68)


def test_confidence_bounds_read_an_individual_fit():
    """The whole point of going through Fit: individual bounds are a fits:
    entry pointing at an individual_fits output, not a mode switch."""
    fit = _individual_fit("individual", {"OtG": _RAMP, "OpQM": [1.0, 2.0, 3.0]})

    assert fit.confidence_bounds(68)["OtG"] == pytest.approx((160.0, 500.0, 840.0))


def test_confidence_bounds_of_an_individual_fit_match_the_joint_reading():
    """Same samples, same numbers, whichever container they arrived in."""
    samples = {"OtG": [0.0, 1.0, 2.0, 3.0], "OpQM": [-1.0, 0.0, 1.0]}

    assert _joint_fit("j", samples).confidence_bounds(95) == _individual_fit(
        "i", samples
    ).confidence_bounds(95)


def test_bounds_property_quotes_the_two_report_levels(joint_fit):
    """68 and 95, keyed by level, each exactly what confidence_bounds gives."""
    bounds = joint_fit.bounds

    assert list(bounds) == [68.0, 95.0]
    assert bounds[68.0] == joint_fit.confidence_bounds(68)
    assert bounds[95.0] == joint_fit.confidence_bounds(95)
