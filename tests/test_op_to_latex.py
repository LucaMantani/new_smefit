"""Unit tests for smefit.op_to_latex — the label lookup the reports share."""

from __future__ import annotations

from smefit.op_to_latex import coeff_info_latex, latex_label


def test_latex_label_uses_the_builtin_coefficient_label() -> None:
    assert latex_label("OQQ1") == coeff_info_latex["OQQ1"]


def test_latex_label_falls_back_to_the_plain_name() -> None:
    assert latex_label("NotARealOp") == "NotARealOp"


def test_latex_label_override_beats_the_builtin_label() -> None:
    assert latex_label("OQQ1", {"OQQ1": "$c_1$"}) == "$c_1$"


def test_latex_label_override_names_what_has_no_builtin_label() -> None:
    """A data group has no default label, so only the runcard can give one."""
    assert latex_label("LHC-top", {"LHC-top": r"$t\bar{t}$"}) == r"$t\bar{t}$"


def test_latex_label_override_for_another_name_is_ignored() -> None:
    assert latex_label("OQQ1", {"LHC-top": "x"}) == coeff_info_latex["OQQ1"]


def test_latex_label_accepts_no_overrides() -> None:
    assert latex_label("OQQ1", None) == latex_label("OQQ1", {})
