"""Unit tests for smefit.wcxf."""

import logging

import numpy as np
import pytest

from smefit.constants import cw, sw
from smefit.wcxf import (
    SMEFIT_TO_WARSAW,
    WarsawMap,
    warsaw_map,
)

GS = 1.2


# ---------------------------------------------------------------------------
# Forward map
# ---------------------------------------------------------------------------


def test_to_warsaw_unit_entry():
    assert warsaw_map(GS).to_warsaw("OpBox") == {"phiBox": 1.0}


def test_to_warsaw_OWWW_sign():
    assert warsaw_map(GS).to_warsaw("OWWW") == {"W": -1.0}


def test_to_warsaw_OtW_two_entries():
    assert warsaw_map(GS).to_warsaw("OtW") == pytest.approx(
        {"uB_33": -cw / sw, "uW_33": -1.0}
    )


def test_to_warsaw_OtG_depends_on_gs():
    assert warsaw_map(GS).to_warsaw("OtG") == {"uG_33": -GS}
    assert warsaw_map(2 * GS).to_warsaw("OtG") == {"uG_33": -2 * GS}


def test_every_entry_is_number_or_callable():
    for op, entry in SMEFIT_TO_WARSAW.items():
        assert entry, op
        for coeff in entry.values():
            assert callable(coeff) or isinstance(coeff, float), op


# ---------------------------------------------------------------------------
# Derived inverse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gs", [GS, 0.7])
def test_inverse_is_left_inverse(gs):
    m = warsaw_map(gs)
    np.testing.assert_allclose(m.inverse @ m.matrix, np.eye(len(m.ops)), atol=1e-12)


def test_inverse_reads_generation_one():
    m = warsaw_map(GS)
    row = m.inverse[m.ops.index("O3pq")]
    assert {m.warsaw[i]: row[i] for i in np.flatnonzero(row)} == {"phiq3_11": 1.0}


# Rows of the hand-written inverse table this map replaced: the derived inverse
# must keep reading the same Warsaw components.
@pytest.mark.parametrize(
    "op, expected",
    [
        ("OtZ", {"uB_33": sw, "uW_33": -cw}),
        ("OpQM", {"phiq1_33": 1.0, "phiq3_33": -1.0}),
        ("O11qq", {"qq1_1133": 1.0, "qq1_1331": 1 / 6, "qq3_1331": 1 / 2}),
        ("OQQ1", {"qq1_3333": 2.0, "qq3_3333": -2 / 3}),
        ("Oee1122", {"ee_1122": 1 / 4}),
    ],
)
def test_inverse_matches_previous_convention(op, expected):
    m = warsaw_map(GS)
    row = m.inverse[m.ops.index(op)]
    assert {m.warsaw[i]: row[i] for i in np.flatnonzero(row)} == pytest.approx(expected)


def test_inverse_OtG_is_minus_one_over_gs():
    assert warsaw_map(GS).to_smefit({"uG_33": 1.0}) == pytest.approx({"OtG": -1 / GS})


def test_dependent_operators_raise():
    table = {"OpBox": {"phiBox": 1.0}, "OpBox2": {"phiBox": 2.0}}
    with pytest.raises(ValueError, match="linearly dependent"):
        WarsawMap.from_table(table, GS)


# ---------------------------------------------------------------------------
# to_smefit
# ---------------------------------------------------------------------------


def test_round_trip_on_symmetric_point(caplog):
    m = warsaw_map(GS)
    rng = np.random.default_rng(0)
    coeffs = dict(zip(m.ops, rng.normal(size=len(m.ops))))
    point = dict(zip(m.warsaw, m.matrix @ np.array(list(coeffs.values()))))
    with caplog.at_level(logging.WARNING, logger="smefit.wcxf"):
        assert m.to_smefit(point) == pytest.approx(coeffs)
    assert not caplog.records


def test_unrelated_operators_are_exactly_zero():
    assert warsaw_map(GS).to_smefit({"phiBox": 1.0}) == {"OpBox": 1.0}


def test_coefficients_outside_the_map_are_dropped():
    assert warsaw_map(GS).to_smefit({"phiBox": 1.0, "qq1_1111": 3.0}) == {"OpBox": 1.0}


def test_broken_symmetry_warns_every_time_and_keeps_generation_one(caplog):
    m = warsaw_map(GS)
    point = {"phid_11": 1.0, "phid_22": 1.0, "phid_33": 5.0}
    with caplog.at_level(logging.WARNING, logger="smefit.wcxf"):
        assert m.to_smefit(point, origin="Running Obb") == {"Opdi": 1.0}
        m.to_smefit(point, origin="Running Obb")
    assert len(caplog.records) == 2
    # phid_33 misses the image by 4, of a largest component of 5
    assert "Running Obb" in caplog.text and "Opdi (80.0%)" in caplog.text


def test_breaking_small_next_to_the_point_still_warns(caplog):
    # the breaking is 4e-6 of the point but all of what Opdi is read from
    point = {"phiBox": 1.0, "phid_11": 1e-6, "phid_22": 1e-6, "phid_33": 5e-6}
    with caplog.at_level(logging.WARNING, logger="smefit.wcxf"):
        warsaw_map(GS).to_smefit(point, origin="Running Oeb")
    assert "Running Oeb" in caplog.text and "Opdi (80.0%)" in caplog.text
    assert "OpBox" not in caplog.text


def test_breaking_within_rtol_of_the_operator_is_silent(caplog):
    point = {"phid_11": 1.0, "phid_22": 1.0, "phid_33": 1.0 + 1e-5}
    with caplog.at_level(logging.WARNING, logger="smefit.wcxf"):
        warsaw_map(GS).to_smefit(point)
    assert not caplog.records


def test_breaking_within_atol_is_silent(caplog):
    # a symmetric pair cut at 1e-14: one component kept, the others dropped
    point = {"phiBox": 1e-6, "phid_33": 1.1e-14}
    with caplog.at_level(logging.WARNING, logger="smefit.wcxf"):
        warsaw_map(GS).to_smefit(point, atol=2e-14)
    assert not caplog.records
