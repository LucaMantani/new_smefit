"""Unit tests for smefit.latex_labels — how a report writes a name in LaTeX."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

from smefit.latex_labels import default_latex_labels, make_latex_label

if TYPE_CHECKING:
    from pathlib import Path


def test_uses_the_builtin_coefficient_label() -> None:
    assert make_latex_label()("OQQ1") == default_latex_labels["OQQ1"]


def test_override_beats_the_builtin_label() -> None:
    assert make_latex_label({"OQQ1": "$c_1$"})("OQQ1") == "$c_1$"


def test_override_names_what_has_no_builtin_label() -> None:
    label = make_latex_label({"MyGroup": r"$t\bar{t}$"})
    assert label("MyGroup") == r"$t\bar{t}$"


def test_override_for_another_name_is_ignored() -> None:
    assert make_latex_label({"LHC-top": "x"})("OQQ1") == default_latex_labels["OQQ1"]


def test_no_overrides_and_empty_overrides_agree() -> None:
    assert make_latex_label(None)("OQQ1") == make_latex_label({})("OQQ1")


def test_overrides_do_not_leak_into_the_defaults() -> None:
    make_latex_label({"OQQ1": "$c_1$"})
    assert make_latex_label()("OQQ1") == default_latex_labels["OQQ1"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("LEP", r"$\mathrm{LEP}$"),
        ("FCCee_91", r"$\mathrm{FCCee\_91}$"),
        ("HL-LHC", r"$\mathrm{HL\mbox{-}LHC}$"),
        ("LHC-top", r"$\mathrm{LHC\mbox{-}top}$"),
        ("My group", r"$\mathrm{My\ group}$"),
        ("a&b%c#d", r"$\mathrm{a\&b\%c\#d}$"),
        ("x$y", r"$\mathrm{x\$y}$"),
        ("{x}", r"$\mathrm{\{x\}}$"),
    ],
)
def test_a_name_without_a_label_is_typeset_as_written(name: str, expected: str) -> None:
    """Group and dataset names come from the runcard, so the fallback must not
    leave anything that would stop usetex compiling."""
    assert make_latex_label()(name) == expected


@pytest.mark.parametrize("name", ["tilde~", "caret^", "back\\slash"])
def test_the_fallback_escapes_each_character_on_its_own(name: str) -> None:
    """One pass over the name: the backslash an escape adds is not re-escaped."""
    label = make_latex_label()(name)
    assert label.count(r"\mbox{") == 1
    assert r"\mbox{\mbox" not in label


@pytest.mark.slow
def test_the_fallback_compiles_under_usetex(tmp_path: Path) -> None:
    """The point of the escaping, against a real LaTeX: a raw dataset name with an
    underscore used to crash the figure."""
    if not (shutil.which("latex") and shutil.which("dvipng")):
        pytest.skip("no LaTeX installation")
    names = ["ATLAS_tttt_13TeV", "HL-LHC", "a b", "a&b%c#d", "x$y", "{x}", "~^\\"]
    with plt.rc_context({"text.usetex": True}):
        fig, ax = plt.subplots()
        ax.set_yticks(range(len(names)), [make_latex_label()(n) for n in names])
        fig.savefig(tmp_path / "labels.png")
    plt.close(fig)
    assert (tmp_path / "labels.png").stat().st_size > 0
