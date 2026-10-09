"""Unit tests for smefit.latex_labels — the label lookup the reports share."""

from __future__ import annotations

from smefit.latex_labels import default_latex_labels, latex_label


def test_latex_label_uses_the_builtin_coefficient_label() -> None:
    assert latex_label("OQQ1") == default_latex_labels["OQQ1"]


def test_latex_label_falls_back_to_the_plain_name() -> None:
    assert latex_label("NotARealOp") == "NotARealOp"


def test_latex_label_override_beats_the_builtin_label() -> None:
    assert latex_label("OQQ1", {"OQQ1": "$c_1$"}) == "$c_1$"


def test_latex_label_override_names_what_has_no_builtin_label() -> None:
    """A group with no default label can only be named by the runcard."""
    assert latex_label("MyGroup", {"MyGroup": r"$t\bar{t}$"}) == r"$t\bar{t}$"


def test_latex_label_uses_the_builtin_data_group_label() -> None:
    assert latex_label("FCCee_240") == default_latex_labels["FCCee_240"]


def test_latex_label_override_beats_a_builtin_data_group_label() -> None:
    assert latex_label("FCCee_240", {"FCCee_240": "ZH run"}) == "ZH run"


def test_latex_label_override_for_another_name_is_ignored() -> None:
    assert latex_label("OQQ1", {"LHC-top": "x"}) == default_latex_labels["OQQ1"]


def test_latex_label_accepts_no_overrides() -> None:
    assert latex_label("OQQ1", None) == latex_label("OQQ1", {})
