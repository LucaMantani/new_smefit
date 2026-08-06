"""Unit tests for smefit.fit_result — FitResult and Fit dataclasses."""

import json
import math

import jax.numpy as jnp
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

    assert Fit.from_json(out).label is None
    assert (
        Fit.from_json(out, label=r"$\mathrm{My\ fit}$").label == r"$\mathrm{My\ fit}$"
    )


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


def test_fit_from_json_roundtrip(tmp_path):
    out = _write_runcard(tmp_path / "my_fit", use_quad=True)
    _make_written_result(
        samples={"OpA": jnp.array([1.0, 2.0]), "OpB": jnp.array([0.0, 1.0])}
    ).write(out)

    recovered = Fit.from_json(out)

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

    assert Fit.from_json(out).use_quad is use_quad


def test_use_quad_ignores_a_stale_payload_entry(tmp_path):
    """Fits written when use_quad was serialised must not be believed."""
    out = _write_runcard(tmp_path / "my_fit", use_quad=True)
    _make_written_result().write(out)
    with (out / "fit_results.json").open() as f:
        payload = json.load(f)
    payload["use_quad"] = False
    (out / "fit_results.json").write_text(json.dumps(payload))

    assert Fit.from_json(out).use_quad is True


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
    _make_written_result().write(out)

    assert Fit.from_json(out).fit_type == fit_type


def test_fit_type_ignores_the_arguments_of_an_action(tmp_path):
    out = _write_runcard(tmp_path / "my_fit", action="run_analytic_fit(main=True)")
    _make_written_result().write(out)

    assert Fit.from_json(out).fit_type == "analytic"


def test_fit_type_looks_past_a_nested_namespace_entry(tmp_path):
    """An ``actions_`` entry that nests actions under a namespace is a mapping."""
    out = tmp_path / "my_fit"
    (out / "input").mkdir(parents=True)
    (out / "input" / "runcard.yaml").write_text(
        yaml.dump({"actions_": [{"scan": ["plot_chi2"]}, "run_hessian_fit"]})
    )
    _make_written_result().write(out)

    assert Fit.from_json(out).fit_type == "hessian"


def test_fit_type_is_unset_when_no_fit_action_ran(tmp_path, caplog):
    out = _write_runcard(tmp_path / "my_fit", action="report")
    _make_written_result().write(out)

    assert Fit.from_json(out).fit_type is None
    assert "no known fit action" in caplog.text


def test_individual_fit_comes_from_the_runcard_action(tmp_path):
    out = _write_runcard(tmp_path / "my_fit", action="run_individual_ultranest_fits")
    _make_written_result().write(out)

    assert Fit.from_json(out).individual_fit is True


def test_a_joint_action_is_not_an_individual_fit(tmp_path):
    out = _write_runcard(tmp_path / "my_fit", action="run_ultranest_fit")
    _make_written_result().write(out)

    assert Fit.from_json(out).individual_fit is False


def test_metadata_without_a_runcard_warns_and_falls_back(tmp_path, caplog):
    out = tmp_path / "no_runcard"
    _make_written_result().write(out)

    recovered = Fit.from_json(out)

    assert recovered.use_quad is False
    assert recovered.fit_type is None
    assert recovered.individual_fit is False
    assert "No input/runcard.yaml" in caplog.text


def test_fit_from_json_on_an_individual_fit_summary(tmp_path):
    """The summary of individual fits has no joint likelihood or evidence."""
    _write_runcard(tmp_path, action="run_individual_analytic_fits")
    r1 = _make_individual_result("OpA", best_val=1.0, samples_vals=[0.8, 1.0, 1.2])
    r2 = _make_individual_result("OpB", best_val=2.0, samples_vals=[1.8, 2.0, 2.2])
    FitResultGroup([r1, r2]).write_summary(tmp_path)

    recovered = Fit.from_json(tmp_path)

    assert recovered.individual_fit is True
    assert recovered.fit_type == "analytic"
    assert math.isnan(recovered.fit_results.max_loglikelihood)
    assert recovered.fit_results.logz is None
    assert recovered.fit_results.free_parameters == ["OpA", "OpB"]


def test_a_summary_without_a_runcard_is_not_known_to_be_individual(tmp_path):
    """The runcard action is the only thing that says a fit ran individually.

    The payload of a summary is never asked: nothing about how a fit was run is
    written to or read from ``fit_results.json``.
    """
    r1 = _make_individual_result("OpA", best_val=1.0, samples_vals=[0.8, 1.0, 1.2])
    r2 = _make_individual_result("OpB", best_val=2.0, samples_vals=[1.8, 2.0, 2.2])
    FitResultGroup([r1, r2]).write_summary(tmp_path)

    assert Fit.from_json(tmp_path).individual_fit is False
