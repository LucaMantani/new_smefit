"""Unit tests for smefit.chi2_scan and run_chi2_scan in utils_actions."""

import json
import logging

import jax.numpy as jnp
import pytest

from smefit.chi2_scan import individual_chi2_scan
from smefit.core import Coefficient, CoefficientGroup

_UNIFORM_PRIOR = {"dist": "uniform", "low": -2.0, "high": 2.0}


def _free(name, prior=None):
    return Coefficient(name=name, free=True, prior=prior or _UNIFORM_PRIOR)


@pytest.fixture
def coeff_with_uniform_prior():
    return CoefficientGroup([_free("OpA")])


# ---------------------------------------------------------------------------
# individual_chi2_scan — result structure
# ---------------------------------------------------------------------------


def test_individual_chi2_scan_result_keys(coeff_with_uniform_prior):
    result = individual_chi2_scan(
        individual_chi2=lambda c: jnp.sum(c**2),
        individual_coefficients=coeff_with_uniform_prior,
        individual_fit_coefficient="OpA",
        chi2_scan_settings={"n_points": 5},
    )
    assert set(result.keys()) == {"OpA"}
    assert set(result["OpA"].keys()) == {"points", "chi2"}


def test_individual_chi2_scan_n_points_respected(coeff_with_uniform_prior):
    result = individual_chi2_scan(
        individual_chi2=lambda c: jnp.sum(c**2),
        individual_coefficients=coeff_with_uniform_prior,
        individual_fit_coefficient="OpA",
        chi2_scan_settings={"n_points": 11},
    )
    assert len(result["OpA"]["points"]) == 11
    assert len(result["OpA"]["chi2"]) == 11


# ---------------------------------------------------------------------------
# individual_chi2_scan — scan range from uniform prior
# ---------------------------------------------------------------------------


def test_individual_chi2_scan_uses_prior_bounds(coeff_with_uniform_prior):
    result = individual_chi2_scan(
        individual_chi2=lambda c: jnp.sum(c**2),
        individual_coefficients=coeff_with_uniform_prior,
        individual_fit_coefficient="OpA",
        chi2_scan_settings={"n_points": 5},
    )
    points = result["OpA"]["points"]
    assert points[0] == pytest.approx(-2.0)
    assert points[-1] == pytest.approx(2.0)


def test_individual_chi2_scan_fallback_range_and_warning(caplog):
    cg = CoefficientGroup([_free("OpB", {"dist": "normal", "loc": 0.0, "scale": 1.0})])
    with caplog.at_level(logging.WARNING, logger="smefit.chi2_scan"):
        result = individual_chi2_scan(
            individual_chi2=lambda c: jnp.sum(c**2),
            individual_coefficients=cg,
            individual_fit_coefficient="OpB",
            chi2_scan_settings={"n_points": 5},
        )
    points = result["OpB"]["points"]
    assert points[0] == pytest.approx(-1.0)
    assert points[-1] == pytest.approx(1.0)
    assert any("default scan range" in r.message for r in caplog.records)


def test_individual_chi2_scan_missing_coeff_fallback(caplog):
    """Coefficient not in the group's prior_specs → fallback to [-1, 1]."""
    cg = CoefficientGroup([_free("OpA")])
    with caplog.at_level(logging.WARNING, logger="smefit.chi2_scan"):
        result = individual_chi2_scan(
            individual_chi2=lambda c: jnp.sum(c**2),
            individual_coefficients=cg,
            individual_fit_coefficient="OpZ",
            chi2_scan_settings={"n_points": 3},
        )
    points = result["OpZ"]["points"]
    assert points[0] == pytest.approx(-1.0)
    assert points[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# individual_chi2_scan — chi2 values
# ---------------------------------------------------------------------------


def test_individual_chi2_scan_chi2_values_correct(coeff_with_uniform_prior):
    # chi2(c) = c[0]^2; scan over [-2, 2] with 3 points → [4, 0, 4]
    result = individual_chi2_scan(
        individual_chi2=lambda c: c[0] ** 2,
        individual_coefficients=coeff_with_uniform_prior,
        individual_fit_coefficient="OpA",
        chi2_scan_settings={"n_points": 3},
    )
    chi2_vals = result["OpA"]["chi2"]
    assert chi2_vals[0] == pytest.approx(4.0, abs=1e-5)
    assert chi2_vals[1] == pytest.approx(0.0, abs=1e-5)
    assert chi2_vals[2] == pytest.approx(4.0, abs=1e-5)


def test_individual_chi2_scan_calls_chi2_at_each_point(coeff_with_uniform_prior):
    calls = []

    def chi2_fn(c):
        calls.append(float(c[0]))
        return jnp.array(0.0)

    individual_chi2_scan(
        individual_chi2=chi2_fn,
        individual_coefficients=coeff_with_uniform_prior,
        individual_fit_coefficient="OpA",
        chi2_scan_settings={"n_points": 7},
    )
    assert len(calls) == 7


def test_individual_chi2_scan_returns_python_floats(coeff_with_uniform_prior):
    result = individual_chi2_scan(
        individual_chi2=lambda c: jnp.sum(c**2),
        individual_coefficients=coeff_with_uniform_prior,
        individual_fit_coefficient="OpA",
        chi2_scan_settings={"n_points": 4},
    )
    for v in result["OpA"]["chi2"]:
        assert isinstance(v, float)
