"""Unit tests for smefit.constants."""

import math

import pytest

import smefit.constants as C


def test_alpha_s():
    assert C.alpha_s == pytest.approx(0.118)


def test_mw():
    assert C.mw == pytest.approx(80.387)


def test_mz():
    assert C.mz == pytest.approx(91.1876)


def test_gs_formula():
    expected = math.sqrt(4 * math.pi * C.alpha_s)
    assert C.gs == pytest.approx(expected)


def test_sw_formula():
    expected = math.sqrt(1 - C.mw**2 / C.mz**2)
    assert C.sw == pytest.approx(expected)


def test_cw_formula():
    expected = math.sqrt(1 - C.sw**2)
    assert C.cw == pytest.approx(expected)


def test_sw2_plus_cw2_equals_one():
    assert C.sw**2 + C.cw**2 == pytest.approx(1.0)


def test_cw_equals_mw_over_mz():
    assert C.cw == pytest.approx(C.mw / C.mz)
