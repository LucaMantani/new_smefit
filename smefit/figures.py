"""
smefit.figures.py

Functions for creating figures and plots.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from reportengine.figure import figure

from smefit.op_to_latex import coeff_info_latex

log = logging.getLogger(__name__)


def _plot_heatmap(matrix, coeff_names, source_names, vmin=None, vmax=None):
    """Render a (n_coeffs, n_sources) matrix as a heatmap and save to PDF.

    Parameters
    ----------
    matrix : array-like, shape (n_coeffs, n_sources)
    coeff_names : list of str
    source_names : list of str
    save_path : pathlib.Path
    vmin, vmax : float, optional
        Colour scale limits passed to imshow.
    """
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
    rc("text", usetex=True)
    rc("text.latex", preamble=r"\usepackage{amssymb}")

    matrix = np.array(matrix)
    n_coeffs, n_sources = matrix.shape

    fig_w = max(6, n_sources * 0.7)
    fig_h = max(3, n_coeffs * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(n_sources))
    ax.set_xticklabels(source_names, rotation=90, fontsize=14)

    coeff_labels = [coeff_info_latex.get(name, name) for name in coeff_names]
    ax.set_yticks(range(n_coeffs))
    ax.set_yticklabels(coeff_labels, fontsize=14)

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    ax.imshow(
        np.ma.masked_equal(matrix, 0.0), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )

    ax.set_xticks(np.arange(-0.5, n_sources, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_coeffs, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    color_threshold = vmax * 0.6
    for i in range(n_coeffs):
        for j in range(n_sources):
            val = matrix[i, j]
            if val == 0.0:
                continue
            color = "white" if val > color_threshold else "black"
            ax.text(
                j, i, f"{val:.1f}", ha="center", va="center", fontsize=10, color=color
            )

    return fig


@figure
def plot_fisher_diagonals_heatmap(fisher_diagonals_normalised):
    """Plot the Fisher diagonals matrix as a heatmap.

    Parameters
    ----------
    fisher_diagonals_normalised : pd.DataFrame
        Index = coeff_names, columns = source_names.
    """
    fd = fisher_diagonals_normalised
    return _plot_heatmap(
        fd.values * 100,
        fd.index.tolist(),
        fd.columns.tolist(),
        vmin=0,
        vmax=100,
    )
