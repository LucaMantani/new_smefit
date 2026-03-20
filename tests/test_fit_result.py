"""Unit tests for smefit.fit_result — FitResult dataclass."""

import json
import math

import jax.numpy as jnp
import pytest

from smefit.fit_result import FitResult


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
