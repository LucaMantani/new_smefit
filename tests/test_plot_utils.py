"""Unit tests for smefit.plot_utils — select_params, rescale_exponents and
set_plot_style."""

import json
import logging
import subprocess
import sys

import matplotlib
import pytest

from smefit.plot_utils import rescale_exponents, select_params, set_plot_style


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
# rescale_exponents
# ---------------------------------------------------------------------------


def test_rescale_exponents_brings_every_row_within_a_decade_of_the_widest():
    """The widest row is the reference; each other one gets the largest power
    of ten that keeps it inside the range the reference sets."""
    extents = [1.0, 0.3, 0.03, 0.0002]

    exponents = rescale_exponents(extents)

    assert exponents == [0, 0, 1, 3]
    rescaled = [extent * 10**k for extent, k in zip(extents, exponents)]
    assert all(0.1 < value <= 1.0 for value in rescaled)


def test_rescale_exponents_does_not_depend_on_the_order_of_the_rows():
    """The reference is the widest row wherever it sits, so reordering the
    rows with params_to_plot does not change their factors."""
    assert rescale_exponents([0.0002, 1.0, 0.03, 0.3]) == [3, 0, 1, 0]


def test_rescale_exponents_leaves_comparable_rows_alone():
    """Rows within a decade of each other are readable as they are: the plot
    is the one drawn without any rescaling."""
    assert rescale_exponents([0.5, 0.2, 0.06]) == [0, 0, 0]


def test_rescale_exponents_only_depends_on_the_ratios():
    """Nothing is special about bounds of order one."""
    assert rescale_exponents([3.0e-4, 6.0e-7]) == rescale_exponents([3.0e4, 6.0e1])
    assert rescale_exponents([3.0e-4, 6.0e-7]) == [0, 2]


def test_rescale_exponents_leaves_a_numerically_zero_row_alone():
    """A row that is zero up to rounding is not a narrow row: magnifying it
    would draw the noise. Neither is an exact zero, whose ratio is infinite."""
    assert rescale_exponents([1.0, 1.0e-13, 0.0, 0.002]) == [0, 0, 0, 2]


def test_rescale_exponents_floor_sets_what_counts_as_zero():
    assert rescale_exponents([1.0, 2.0e-5]) == [0, 4]
    assert rescale_exponents([1.0, 2.0e-5], floor=1.0e-3) == [0, 0]


@pytest.mark.parametrize("extents", [[], [0.0], [0.0, 0.0]])
def test_rescale_exponents_with_nothing_to_compare_rescales_nothing(extents):
    """No rows, or rows that are all zero: there is no reference to scale to."""
    assert rescale_exponents(extents) == [0] * len(extents)


def test_rescale_exponents_returns_plain_ints():
    """The exponents are written into labels, where a numpy integer would do
    but a float would read 10^{2.0}."""
    exponents = rescale_exponents(iter([1.0, 0.004]))

    assert exponents == [0, 2]
    assert all(type(k) is int for k in exponents)


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

    latex_labels used to do exactly that, which silently switched every plot in
    the importing program over to LaTeX and broke any machine without one.
    """
    script = (
        "import matplotlib, json;"
        "before = matplotlib.rcParams['text.usetex'];"
        "import smefit.latex_labels, smefit.tables, smefit.figures, smefit.plot_utils;"
        "print(json.dumps([before, matplotlib.rcParams['text.usetex']]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    before, after = json.loads(out.stdout)
    assert before is False and after is False
