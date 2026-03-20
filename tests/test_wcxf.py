"""Unit tests for smefit.wcxf."""

import pytest

from smefit.wcxf import inverse_wcxf_translate, wcxf_translate

# ---------------------------------------------------------------------------
# Structural invariants for wcxf_translate
# ---------------------------------------------------------------------------


def test_wcxf_translate_non_empty():
    assert len(wcxf_translate) > 0


def test_wcxf_translate_every_entry_has_wc_key():
    for name, entry in wcxf_translate.items():
        assert "wc" in entry, f"Entry '{name}' missing 'wc' key"


def test_wcxf_translate_wc_lists_non_empty():
    for name, entry in wcxf_translate.items():
        assert len(entry["wc"]) > 0, f"Entry '{name}' has empty 'wc' list"


def test_wcxf_translate_value_length_matches_wc():
    for name, entry in wcxf_translate.items():
        if "value" in entry:
            assert len(entry["value"]) == len(entry["wc"]), (
                f"Entry '{name}': len(value)={len(entry['value'])} "
                f"!= len(wc)={len(entry['wc'])}"
            )


# ---------------------------------------------------------------------------
# Spot checks for wcxf_translate
# ---------------------------------------------------------------------------


def test_wcxf_translate_OpBox():
    assert wcxf_translate["OpBox"] == {"wc": ["phiBox"]}


def test_wcxf_translate_OWWW():
    assert wcxf_translate["OWWW"] == {"wc": ["W"]}


def test_wcxf_translate_OtG_has_value():
    entry = wcxf_translate["OtG"]
    assert entry["wc"] == ["uG_33"]
    assert "value" in entry
    assert len(entry["value"]) == 1


def test_wcxf_translate_OtW_two_wcs():
    entry = wcxf_translate["OtW"]
    assert len(entry["wc"]) == 2
    assert len(entry["value"]) == 2


# ---------------------------------------------------------------------------
# Structural invariants for inverse_wcxf_translate
# ---------------------------------------------------------------------------


def test_inverse_wcxf_translate_non_empty():
    assert len(inverse_wcxf_translate) > 0


def test_inverse_wcxf_translate_every_entry_has_wc_key():
    for name, entry in inverse_wcxf_translate.items():
        assert "wc" in entry, f"Entry '{name}' missing 'wc' key"


def test_inverse_wcxf_translate_wc_lists_non_empty():
    for name, entry in inverse_wcxf_translate.items():
        assert len(entry["wc"]) > 0, f"Entry '{name}' has empty 'wc' list"


def test_inverse_wcxf_translate_coeff_length_matches_wc():
    for name, entry in inverse_wcxf_translate.items():
        if "coeff" in entry:
            assert len(entry["coeff"]) == len(entry["wc"]), (
                f"Entry '{name}': len(coeff)={len(entry['coeff'])} "
                f"!= len(wc)={len(entry['wc'])}"
            )


# ---------------------------------------------------------------------------
# Spot checks for inverse_wcxf_translate
# ---------------------------------------------------------------------------


def test_inverse_wcxf_translate_OpBox():
    assert inverse_wcxf_translate["OpBox"] == {"wc": ["phiBox"]}


def test_inverse_wcxf_translate_OWWW():
    assert inverse_wcxf_translate["OWWW"] == {"wc": ["W"]}


def test_inverse_wcxf_translate_OpD():
    assert inverse_wcxf_translate["OpD"] == {"wc": ["phiD"]}
