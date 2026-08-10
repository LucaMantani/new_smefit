"""Unit tests for smefit.figures — _plot_heatmap and plot_fisher_diagonals_heatmap.

_plot_heatmap enables matplotlib's usetex globally, which would require a
LaTeX installation to actually render text. Tests here never draw/save the
figure (get_text() etc. only read stored attributes), and the module-level
`rc` call is neutralised so running the suite never depends on LaTeX being
installed (e.g. on CI runners).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from smefit import figures as figures_mod
from smefit.figures import (
    _apply_group_latex_labels,
    _plot_heatmap,
    plot_fisher_diagonals_heatmap,
)


@pytest.fixture(autouse=True)
def _no_matplotlib_rc(monkeypatch):
    monkeypatch.setattr(figures_mod, "rc", lambda *args, **kwargs: None)
    yield
    plt.close("all")


def test_plot_heatmap_axis_labels():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    fig = _plot_heatmap(matrix, ["OpA", "OpZZ"], ["DS_A", "DS_B"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["DS_A", "DS_B"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]


def test_plot_heatmap_uses_latex_label_when_known():
    """Coefficient names present in coeff_info_latex are relabelled on the y-axis."""
    matrix = np.array([[1.0]])
    fig = _plot_heatmap(matrix, ["OQQ1"], ["DS_A"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert ax.get_yticklabels()[0].get_text() == r"$c_{QQ}^{\scriptscriptstyle 1}$"


def test_plot_heatmap_skips_zero_cells():
    """Cells equal to 0.0 get no text annotation."""
    matrix = np.array([[0.0, 5.0]])
    fig = _plot_heatmap(matrix, ["OpA"], ["DS_A", "DS_B"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["5.0"]


def test_plot_heatmap_text_color_threshold():
    """Cell text is white above 60% of vmax, black otherwise."""
    matrix = np.array([[8.0, 2.0]])
    fig = _plot_heatmap(matrix, ["OpA"], ["DS_A", "DS_B"], vmin=0, vmax=10)

    ax = fig.axes[0]
    colors = {t.get_text(): t.get_color() for t in ax.texts}
    assert colors["8.0"] == "white"
    assert colors["2.0"] == "black"


def test_plot_heatmap_figsize_scales_with_shape():
    small = _plot_heatmap(np.zeros((1, 1)), ["OpA"], ["DS_A"], vmin=0, vmax=1)
    large = _plot_heatmap(
        np.zeros((8, 10)),
        [f"Op{i}" for i in range(8)],
        [f"DS_{i}" for i in range(10)],
        vmin=0,
        vmax=1,
    )

    assert large.get_size_inches()[0] > small.get_size_inches()[0]
    assert large.get_size_inches()[1] > small.get_size_inches()[1]


def test_plot_fisher_diagonals_heatmap_scales_values_to_percent():
    fd = pd.DataFrame(
        {"DS_A": [0.3, 0.7], "DS_B": [0.6, 0.4]},
        index=["OpA", "OpZZ"],
    )

    fig = plot_fisher_diagonals_heatmap(fd)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["DS_A", "DS_B"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]
    texts = {t.get_text() for t in ax.texts}
    assert texts == {"30.0", "70.0", "60.0", "40.0"}


def test_apply_group_latex_labels_replaces_declared_groups():
    labels = {"LHC-top": r"$\mathrm{LHC}\ t\bar{t}$"}

    result = _apply_group_latex_labels(["LHC-top", "LEP"], labels)

    assert result == [r"$\mathrm{LHC}\ t\bar{t}$", "LEP"]


def test_apply_group_latex_labels_none_or_empty_is_identity():
    assert _apply_group_latex_labels(["DS_A", "DS_B"], None) == ["DS_A", "DS_B"]
    assert _apply_group_latex_labels(["DS_A", "DS_B"], {}) == ["DS_A", "DS_B"]


def test_apply_group_latex_labels_warns_on_unmatched_group(caplog):
    with caplog.at_level("WARNING"):
        result = _apply_group_latex_labels(["LEP"], {"LHC-top": r"$t\bar{t}$"})

    assert result == ["LEP"]
    assert "LHC-top" in caplog.text


def test_plot_fisher_diagonals_heatmap_uses_group_latex_labels():
    fd = pd.DataFrame(
        {"LHC-top": [0.3, 0.7], "LEP": [0.7, 0.3]},
        index=["OpA", "OpZZ"],
    )

    fig = plot_fisher_diagonals_heatmap(
        fd, group_latex_labels={"LHC-top": r"$\mathrm{LHC}\ t\bar{t}$"}
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        r"$\mathrm{LHC}\ t\bar{t}$",
        "LEP",
    ]
