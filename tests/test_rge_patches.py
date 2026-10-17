"""Tests for smefit/rge/_patches.py — the wilson/ckmutil monkey patches.

These patches are process-global and applied at import of ``smefit.rge``, so the
tests here check that the rebinding happened and that the wrappers forward
correctly. They never run real RGE evolution.
"""

import numpy as np
import pytest
import wilson

from smefit.rge import _patches

# ---------------------------------------------------------------------------
# The patches are in place after importing smefit.rge
# ---------------------------------------------------------------------------


def test_beta_is_patched():
    assert wilson.run.smeft.beta.beta is _patches.beta_wrapper


def test_smeftpar_is_patched():
    assert wilson.run.smeft.smpar.smeftpar is _patches.patched_smeftpar


def test_to_wcxf_is_patched():
    assert wilson.run.smeft.classes.SMEFT._to_wcxf is _patches._to_wcxf_no_rotation


def test_ckm_tree_is_patched_to_zeroth_order():
    """The CP phase is fixed to gamma rather than computed."""
    import ckmutil.ckm

    assert ckmutil.ckm.ckm_tree.keywords == {"delta_expansion_order": 0}


# ---------------------------------------------------------------------------
# beta_wrapper — forces HIGHSCALE=inf so SM-EFT mixing is switched off
# ---------------------------------------------------------------------------


def test_beta_wrapper_defaults_highscale_to_infinity(monkeypatch):
    captured = {}

    def fake_beta(C, HIGHSCALE, *args, **kwargs):
        captured["C"] = C
        captured["HIGHSCALE"] = HIGHSCALE
        return "beta"

    monkeypatch.setattr(_patches, "original_beta", fake_beta)

    assert _patches.beta_wrapper({"phi": 1.0}) == "beta"
    assert captured["C"] == {"phi": 1.0}
    assert captured["HIGHSCALE"] == np.inf


def test_beta_wrapper_forwards_an_explicit_highscale(monkeypatch):
    captured = {}

    def fake_beta(C, HIGHSCALE, *args, **kwargs):
        captured["HIGHSCALE"] = HIGHSCALE
        return "beta"

    monkeypatch.setattr(_patches, "original_beta", fake_beta)

    _patches.beta_wrapper({"phi": 1.0}, 1e16)
    assert captured["HIGHSCALE"] == 1e16


# ---------------------------------------------------------------------------
# patched_smeftpar — blanks out the SMEFT contribution to SM parameters
# ---------------------------------------------------------------------------


def test_patched_smeftpar_replaces_C_passed_as_keyword(monkeypatch):
    captured = {}

    def fake_smeftpar(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "smpar"

    monkeypatch.setattr(_patches, "original_smeftpar", fake_smeftpar)

    result = _patches.patched_smeftpar(160.0, C={"phi": 7.0}, basis="Warsaw")

    assert result == "smpar"
    assert captured["kwargs"]["C"] == _patches.C_patch
    assert captured["kwargs"]["basis"] == "Warsaw"
    assert captured["args"] == (160.0,)


def test_patched_smeftpar_replaces_C_passed_positionally(monkeypatch):
    captured = {}

    def fake_smeftpar(*args, **kwargs):
        captured["args"] = args
        return "smpar"

    monkeypatch.setattr(_patches, "original_smeftpar", fake_smeftpar)

    _patches.patched_smeftpar(160.0, {"phi": 7.0}, "Warsaw")

    # the second positional argument is the one that gets blanked
    assert captured["args"][0] == 160.0
    assert captured["args"][1] == _patches.C_patch
    assert captured["args"][2] == "Warsaw"


def test_C_patch_zeroes_every_entry():
    """Every Higgs-sector Wilson coefficient the SM parameters could see is zero."""
    assert set(_patches.C_patch) == {
        "phi",
        "phiBox",
        "phiD",
        "phiWB",
        "phiG",
        "phiW",
        "phiB",
        "dphi",
        "uphi",
        "ephi",
    }
    assert all(value == pytest.approx(0.0) for value in _patches.C_patch.values())
