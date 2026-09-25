"""Unit tests for smefit.plot_utils — select_params and set_plot_style."""

import json
import logging
import subprocess
import sys

import matplotlib
import pytest

from smefit.plot_utils import select_params, set_plot_style


def test_select_params_without_a_list_keeps_everything():
    """No params_to_plot in the runcard means the routine plots all it has,
    in its own order."""
    assert select_params(["OpZZ", "OpA"], None) == ["OpZZ", "OpA"]


def test_select_params_follows_the_requested_order():
    """The list is a layout as much as a filter: operators are written in the
    order they should be read, which is rarely the alphabetical one."""
    assert select_params(["OpA", "OpB", "OpC"], ["OpC", "OpA"]) == ["OpC", "OpA"]


def test_select_params_skips_names_that_are_not_there():
    """One runcard-wide list serves several fits, which need not have fitted
    the same coefficients: each keeps the largest subset it can."""
    assert select_params(["OpA", "OpB"], ["OpA", "OpMissing", "OpB"]) == [
        "OpA",
        "OpB",
    ]


def test_select_params_says_which_names_it_skipped(caplog):
    """Expected with several fits, so it is reported rather than warned — but
    it still has to be reported, or a typo shrinks a plot in silence."""
    with caplog.at_level(logging.INFO):
        select_params(["OpA"], ["OpA", "OpMissing"], context="my_fit")

    assert "OpMissing" in caplog.text
    assert "my_fit" in caplog.text


def test_select_params_keeps_a_repeated_name_once():
    """A coefficient listed twice would otherwise be drawn twice."""
    assert select_params(["OpA", "OpB"], ["OpA", "OpB", "OpA"]) == ["OpA", "OpB"]


def test_select_params_accepts_a_bare_name():
    """`params_to_plot: OtG` is a plausible way to write a one-element list."""
    assert select_params(["OpA", "OtG"], "OtG") == ["OtG"]


def test_select_params_rejects_an_empty_selection():
    """Nothing left to plot is a misspelt list, not a subset anybody asked
    for, so it says what was available instead of drawing an empty figure."""
    with pytest.raises(ValueError, match="OpA"):
        select_params(["OpA", "OpB"], ["OpTypo"])


# ---------------------------------------------------------------------------
# set_plot_style
# ---------------------------------------------------------------------------


def test_set_plot_style_enables_latex_rendering():
    """The style is what turns usetex on — nothing else in smefit may."""
    with matplotlib.rc_context():
        matplotlib.rcParams["text.usetex"] = False
        set_plot_style()
        assert matplotlib.rcParams["text.usetex"] is True
        assert matplotlib.rcParams["font.family"] == ["sans-serif"]


def test_importing_smefit_does_not_touch_matplotlib_settings():
    """No smefit module may restyle matplotlib merely by being imported.

    op_to_latex used to do exactly that, which silently switched every plot in
    the importing program over to LaTeX and broke any machine without one.
    """
    script = (
        "import matplotlib, json;"
        "before = matplotlib.rcParams['text.usetex'];"
        "import smefit.op_to_latex, smefit.tables, smefit.figures, smefit.plot_utils;"
        "print(json.dumps([before, matplotlib.rcParams['text.usetex']]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    before, after = json.loads(out.stdout)
    assert before is False and after is False
