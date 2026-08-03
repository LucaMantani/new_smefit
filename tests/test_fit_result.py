"""Unit tests for smefit.fit_result — FitResult dataclass."""

import json
import logging
import math

import jax.numpy as jnp
import pytest

from smefit.fit_result import FitResult, FitResultGroup


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
# The RGE matrix companion artefact
# ---------------------------------------------------------------------------


def _make_rge_matrix(scales=(91.2,)):
    """A small but real RGEMatrix, so the round trip goes through real pickling."""
    from smefit.rge import RGEMatrix, RGESettings

    return RGEMatrix(
        stacked_mats=jnp.array([[[1.0], [2.0]] for _ in scales]),
        obs_operators=["OpBox", "OpD"],
        init_operators=["OpBox"],
        scales=list(scales),
        settings=RGESettings(init_scale=1000.0),
    )


def test_write_json_payload_keys_unaffected_by_rge_matrix(tmp_path):
    """The payload key set must not depend on whether a matrix is attached.

    `scales` is written unconditionally (null when there is no matrix), like
    every other optional field, so `from_json` never has to guess.
    """
    without = _make_result()
    without.write(tmp_path / "plain")

    with_rge = _make_result()
    with_rge.rge_matrix = _make_rge_matrix()
    with_rge.write(tmp_path / "with_rge")

    with (tmp_path / "plain" / "fit_results.json").open() as f:
        plain_payload = json.load(f)
    with (tmp_path / "with_rge" / "fit_results.json").open() as f:
        rge_payload = json.load(f)

    assert plain_payload.keys() == rge_payload.keys()
    assert "rge_matrix" not in rge_payload  # the matrix itself stays out of JSON
    assert plain_payload["scales"] is None
    assert rge_payload["scales"] == [91.2]


def test_write_emits_rge_matrix_alongside_result(tmp_path):
    fr = _make_result()
    fr.rge_matrix = _make_rge_matrix()
    fr.write(tmp_path)

    assert (tmp_path / "rge_matrix.pkl").exists()


def test_write_without_rge_matrix_writes_no_pickle(tmp_path):
    _make_result().write(tmp_path)
    assert not (tmp_path / "rge_matrix.pkl").exists()


def test_from_json_restores_the_rge_matrix(tmp_path):
    """The round trip is exact, including the per-data-point stacking."""
    fr = _make_result()
    fr.rge_matrix = _make_rge_matrix(scales=(91.2, 91.2, 200.0))
    fr.write(tmp_path)

    loaded = FitResult.from_json(tmp_path)

    assert loaded.rge_matrix is not None
    assert loaded.rge_matrix.scales == [91.2, 91.2, 200.0]
    assert loaded.rge_matrix.obs_operators == ["OpBox", "OpD"]
    assert loaded.rge_matrix.init_operators == ["OpBox"]
    assert loaded.rge_matrix.stacked_mats.shape == fr.rge_matrix.stacked_mats.shape
    assert loaded.rge_matrix.settings == fr.rge_matrix.settings


def test_from_json_without_rge_matrix_gives_none(tmp_path):
    _make_result().write(tmp_path)
    assert FitResult.from_json(tmp_path).rge_matrix is None


def test_from_json_tolerates_results_written_before_rge_scales_existed(tmp_path):
    """Legacy fit_results.json has no scales key at all."""
    _make_result().write(tmp_path)
    payload_file = tmp_path / "fit_results.json"
    with payload_file.open() as f:
        payload = json.load(f)
    del payload["scales"]
    with payload_file.open("w") as f:
        json.dump(payload, f)

    assert FitResult.from_json(tmp_path).rge_matrix is None


def test_from_json_warns_when_the_pickle_is_missing(tmp_path, caplog):
    """Losing the companion file must be visible, not a silent None."""
    fr = _make_result()
    fr.rge_matrix = _make_rge_matrix()
    fr.write(tmp_path)
    (tmp_path / "rge_matrix.pkl").unlink()

    with caplog.at_level(logging.WARNING):
        loaded = FitResult.from_json(tmp_path)

    assert loaded.rge_matrix is None
    assert "rge_matrix.pkl" in caplog.text


def test_group_takes_rge_matrix_over_from_its_results(tmp_path):
    """Each sub-result carries the shared matrix; the group writes it once.

    Individual fits all run against the same EFT model, so writing per-result
    would put an identical pickle in every coefficient's subdirectory.
    """
    rge_matrix = _make_rge_matrix()
    results = [
        _make_individual_result("OpA", 1.0, [1.0, 2.0]),
        _make_individual_result("OpB", 2.0, [2.0, 3.0]),
    ]
    for r in results:
        r.rge_matrix = rge_matrix

    group = FitResultGroup(results)
    assert group.rge_matrix is rge_matrix

    group.write_results(tmp_path)

    assert (tmp_path / "rge_matrix.pkl").exists()
    assert not list((tmp_path / "individual_fits").glob("*/rge_matrix.pkl"))


def test_group_without_rge_matrix_writes_no_pickle(tmp_path):
    results = [_make_individual_result("OpA", 1.0, [1.0, 2.0])]
    FitResultGroup(results).write_results(tmp_path)
    assert not (tmp_path / "rge_matrix.pkl").exists()


def test_group_warns_if_results_carry_different_rge_matrices(caplog):
    """They share one matrix by construction; if that breaks, say so out loud.

    All of them target the same rge_matrix.pkl, so the extras would otherwise be
    dropped silently.
    """
    results = [
        _make_individual_result("OpA", 1.0, [1.0, 2.0]),
        _make_individual_result("OpB", 2.0, [2.0, 3.0]),
    ]
    results[0].rge_matrix = _make_rge_matrix()
    results[1].rge_matrix = _make_rge_matrix()

    with caplog.at_level(logging.WARNING):
        group = FitResultGroup(results)

    assert "different RGE matrices" in caplog.text
    assert group.rge_matrix is results[0].rge_matrix


# ---------------------------------------------------------------------------
# FitResultGroup.write_summary
# ---------------------------------------------------------------------------


def _make_individual_result(
    name, best_val, samples_vals, max_loglikelihood=-5.0, num_data=10
):
    """Build a single-free-parameter FitResult as produced by an individual fit."""
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
