"""
smefit.figures.py

Functions for creating figures and plots.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from reportengine.figure import figure, figuregen

from smefit.plot_utils import select_params, set_plot_style

log = logging.getLogger(__name__)


def _plot_heatmap(
    matrix,
    coeff_names,
    source_names,
    vmin=None,
    vmax=None,
    cmap="Blues",
    mask_zeros=True,
    value_fmt="{:.1f}",
    aspect="auto",
    colorbar=False,
    title=None,
):
    """Render a (n_coeffs, n_sources) matrix as a heatmap and save to PDF.

    Parameters
    ----------
    matrix : array-like, shape (n_coeffs, n_sources)
    coeff_names : list of str
    source_names : list of str
        Row and column labels, drawn as given: the callers have already
        turned names into labels.
    vmin, vmax : float, optional
        Colour scale limits passed to imshow.
    cmap : str, optional
        Colormap name. The default suits a matrix of one sign; a matrix that
        spans zero wants a diverging one.
    mask_zeros : bool, optional
        Whether an exactly zero cell is left blank. True where a zero means
        "this source does not enter" (the default), False where zero is a
        value like any other. Cells that are not finite are always blank —
        they are the absence of a number, whatever the matrix means.
    value_fmt : str, optional
        Format string for the per-cell annotation.
    aspect : str, optional
        imshow aspect. ``"equal"`` keeps cells square, which a square matrix
        wants and a (coeffs x sources) one does not.
    colorbar : bool, optional
        Whether to draw the colour scale alongside.
    title : str, optional
        Heading for the plot, drawn above the column labels. Passed to
        matplotlib verbatim, so it may be raw LaTeX.
    """
    set_plot_style()

    matrix = np.array(matrix, dtype=float)
    n_coeffs, n_sources = matrix.shape

    fig_w = max(6, n_sources * 0.7)
    fig_h = max(3, n_coeffs * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(n_sources))
    ax.set_xticklabels(source_names, rotation=90, fontsize=14)
    if title is not None:
        ax.set_xlabel(title, fontsize=18, labelpad=12)

    ax.set_yticks(range(n_coeffs))
    ax.set_yticklabels(coeff_names, fontsize=14)

    masked = np.ma.masked_invalid(matrix)
    if mask_zeros:
        masked = np.ma.masked_equal(masked, 0.0)

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("black", 0.7)
    im = ax.imshow(masked, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(-0.5, n_sources, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_coeffs, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    blank = np.ma.getmaskarray(masked)
    for i in range(n_coeffs):
        for j in range(n_sources):
            if blank[i, j]:
                continue
            val = matrix[i, j]
            red, green, blue, _ = im.cmap(im.norm(val))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            color = "white" if luminance < 0.5 else "black"
            ax.text(
                j,
                i,
                value_fmt.format(val),
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )

    return fig


@figuregen
def plot_chi2_scan(individual_chi2_scans, latex_label):
    """Plot the 1D chi2 scan for each free coefficient.

    Yields one ``(figure, coeff_name)`` per scanned coefficient, with the
    coefficient value on the x axis and chi2 on the y axis.

    Parameters
    ----------
    individual_chi2_scans : list[dict]
        Each entry maps ``{coeff_name: {"points": [...], "chi2": [...]}}``.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    """
    set_plot_style()

    results = {k: v for d in individual_chi2_scans for k, v in d.items()}
    for name, data in results.items():
        label = latex_label(name)
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        ax.plot(data["points"], data["chi2"], "-o", color="C0", markersize=4)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\chi^2$")
        ax.grid(alpha=0.3)
        yield fig, name


@figure
def plot_fisher_diagonals_heatmap(
    fisher_diagonals_normalised,
    cmap="Blues",
    value_fmt="{:.1f}",
    colorbar=False,
):
    """Plot the Fisher diagonals matrix as a heatmap.

    Parameters
    ----------
    fisher_diagonals_normalised : pd.DataFrame
        Index = coeff_names, columns = source_names.
    cmap : str, optional
        Colormap. Sequential by default: the values are percentage shares, so
        they run one way from zero and a diverging map would invent a midpoint.
    value_fmt : str, optional
        Format of the per-cell annotation, in percent.
    colorbar : bool, optional
        Whether to draw the colour scale alongside. Off by default: rows sum to
        100%, so the annotations already say what a cell is worth.
    """
    fd = fisher_diagonals_normalised
    return _plot_heatmap(
        fd.values * 100,
        fd.index.tolist(),
        fd.columns.tolist(),
        vmin=0,
        vmax=100,
        cmap=cmap,
        value_fmt=value_fmt,
        colorbar=colorbar,
    )


@figure
def plot_pca_components_heatmap(
    pca_components,
    cmap="RdBu_r",
    value_fmt="{:.2f}",
    colorbar=True,
):
    """Plot the principal-direction weights as a heatmap.

    Reading it: a column is one principal direction, and the coloured cells say
    which coefficients it is made of. The leftmost columns are the directions
    the data pin down; the rightmost are the flat ones, and the coefficients
    lit up there are the combinations the fit cannot resolve.

    Parameters
    ----------
    pca_components : pd.DataFrame
        Index = coeff_names, columns = PC1..PCn.
    cmap : str, optional
        Colormap. Diverging by default: eigenvector components run either way
        about zero, and the relative sign of two entries is what says whether a
        direction is a sum or a difference of them.
    value_fmt : str, optional
        Format of the per-cell annotation.
    colorbar : bool, optional
        Whether to draw the colour scale alongside.
    """
    comps = pca_components
    return _plot_heatmap(
        comps.values,
        comps.index.tolist(),
        comps.columns.tolist(),
        vmin=-1,
        vmax=1,
        mask_zeros=False,
        cmap=cmap,
        value_fmt=value_fmt,
        colorbar=colorbar,
    )


@figure
def plot_pca_spectrum(pca, latex_label):
    """Plot the eigenvalue spectrum, with the flat-direction threshold marked.

    The one picture of the whole analysis: bars falling below the dashed line
    are the directions the data do not constrain.

    Parameters
    ----------
    pca : smefit.pca.PCA
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    """
    set_plot_style()

    ratios = np.array(pca.eigenvalue_ratios, dtype=float)
    index = np.arange(1, pca.n_components + 1)
    flat = np.array(pca.flat_mask)

    # A log axis has no zero to grow the bars from, so the bottom of the axis
    # plays that role: every bar starts on it, and its length is how many
    # decades the direction stands above it. That floor follows the spectrum
    # rather than the threshold — the threshold is far below any physical scale
    # by design, and anchoring the axis to it would spend a dozen empty decades
    # on a plot whose bars all sit at the top. Only when something is actually
    # flagged flat does the floor drop below it, so that the dashed line and the
    # bars it condemns are in view together.
    positive = ratios[ratios > 0]
    lowest = positive.min() if positive.size else pca.threshold
    if flat.any():
        lowest = min(lowest, pca.threshold)
    floor = lowest / 10
    shown = np.clip(ratios, floor, None)
    heights = shown - floor

    fig, ax = plt.subplots(figsize=(max(6, pca.n_components * 0.4), 5))
    ax.set_axisbelow(True)
    ax.bar(
        index[~flat],
        heights[~flat],
        bottom=floor,
        width=0.7,
        color="C0",
        label="constrained",
    )
    if flat.any():
        ax.bar(
            index[flat],
            heights[flat],
            bottom=floor,
            width=0.7,
            color="C3",
            label="flat",
        )
        # A direction with no curvature at all sits on the floor, leaving a bar
        # of zero length; mark it with a downward triangle just above the
        # baseline so it is still read as present and off the bottom of the
        # scale.
        off_scale = flat & (ratios <= floor)
        if off_scale.any():
            ax.scatter(
                index[off_scale],
                np.full(int(off_scale.sum()), floor * 2.0),
                s=60,
                color="C3",
                marker="v",
                zorder=3,
            )
    # Below the bottom of the view when nothing is flat, which is the right
    # reading: there is then no direction for the threshold to separate.
    ax.axhline(pca.threshold, color="black", linestyle="--", linewidth=1)

    ax.set_yscale("log")
    # exactly the floor, so the bars sit on the bottom axis rather than hanging
    # above it
    ax.set_ylim(floor, 2.0)
    ax.set_xticks(index)
    ax.set_xticklabels(
        [latex_label(name) for name in pca.component_names], rotation=90, fontsize=12
    )
    ax.set_ylabel(r"$\lambda_i / \lambda_{\rm max}$", fontsize=16)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig


@figure
def plot_posterior_correlations(
    fit,
    latex_label,
    params_to_plot=None,
    cmap="RdBu_r",
    value_fmt="{:.2f}",
    colorbar=True,
):
    """Plot the posterior correlations of one fit's free coefficients.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one heatmap per fit, each headed with the fit it is drawn from.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    params_to_plot : list of str, optional
        Restrict the heatmap to these coefficients, in this order.
        All of them by default.
    cmap : str, optional
        Colormap. Diverging by default, as correlations run either way about
        zero; a sequential one misreads anti-correlation as "little".
    value_fmt : str, optional
        Format of the per-cell annotation. Widen it to read small
        correlations, or pass a format that renders nothing to drop the
        numbers on a matrix too large to label.
    colorbar : bool, optional
        Whether to draw the colour scale alongside.

    Raises
    ------
    ValueError
        If the fit was run one coefficient at a time: its coefficients were
        never sampled together, so there is no joint posterior to correlate.
    """
    if fit.individual_fit:
        raise ValueError(
            f"Fit '{fit.fit_name}' was run one coefficient at a time, so its "
            "coefficients were never sampled together and there is no joint "
            "posterior to correlate."
        )

    corr = fit.fit_results.correlations
    selected = select_params(corr.index, params_to_plot, context=fit.fit_name)
    corr = corr.loc[selected, selected]
    labels = [latex_label(name) for name in selected]
    return _plot_heatmap(
        corr.values,
        labels,
        labels,
        vmin=-1,
        vmax=1,
        mask_zeros=False,
        aspect="equal",
        cmap=cmap,
        value_fmt=value_fmt,
        colorbar=colorbar,
        title=fit.plot_label,
    )


@figure
def plot_posterior_histograms(
    fits,
    latex_label,
    params_to_plot=None,
    n_bins=30,
    n_cols=4,
):
    """Plot the marginal posterior of every coefficient, fits overlaid.

    One panel per coefficient, on a grid ``n_cols`` wide; each fit listed under
    ``fits:`` is drawn in its own colour on every panel, so a single fit gives
    its marginals and several give their comparison. A coefficient one fit did
    not sample is simply missing from that fit's histograms. The histograms
    are normalised, so fits with different numbers of samples compare
    directly.

    Free and ``expr``-constrained coefficients both get a panel: a constrained
    one has a posterior of its own, induced by the free ones it depends on.
    Coefficients fixed to a ``value`` do not, and are left out.

    Fits run one coefficient at a time are fine here: their marginals are what
    each individual fit sampled.

    Parameters
    ----------
    fits : list of smefit.fit_result.Fit
        Previously run fits, loaded from disk.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    params_to_plot : list of str, optional
        Restrict the panels to these coefficients, in this order. All the free
        and constrained coefficients of all the fits by default, in order of
        first appearance.
    n_bins : int, optional
        Number of histogram bins per panel, shared by every fit on that panel
        so their bars line up.
    n_cols : int, optional
        Panels per row.

    Raises
    ------
    ValueError
        If ``fits`` is empty, a fit stored no samples, or ``params_to_plot``
        selects nothing any fit has.
    """
    set_plot_style()

    if not fits:
        raise ValueError("fits is empty: list at least one fit under 'fits:'.")

    per_fit = []
    for fit in fits:
        samples = fit.fit_results.samples
        if not samples:
            raise ValueError(
                f"The fit '{fit.fit_name}' stored no posterior samples, so it "
                "has no posterior to plot."
            )
        # fixed coefficients are stored too, as one constant repeated
        fixed = set(fit.fixed_coefficients)
        per_fit.append(
            {
                name: np.asarray(values, dtype=float)
                for name, values in samples.items()
                if name not in fixed
            }
        )

    # union, in order of first appearance; selected once on it, since
    # params_to_plot is global and fits need not share coefficients
    names = list(dict.fromkeys(name for samples in per_fit for name in samples))
    context = ", ".join(fit.fit_name for fit in fits)
    selected = select_params(names, params_to_plot, context=context)

    n_cols = max(1, min(int(n_cols), len(selected)))
    n_rows = -(-len(selected) // n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()
    for ax in axes[len(selected) :]:
        ax.set_visible(False)

    colors = [f"C{i}" for i in range(len(fits))]
    for ax, name in zip(axes, selected):
        drawn = [
            (samples[name], color)
            for samples, color in zip(per_fit, colors)
            if name in samples
        ]
        lowest = min(values.min() for values, _ in drawn)
        highest = max(values.max() for values, _ in drawn)
        edges = np.linspace(lowest, highest, int(n_bins) + 1)

        for values, color in drawn:
            if np.ptp(values) == 0:
                # a point mass has no width to bin
                ax.axvline(values[0], color=color, linewidth=1.5)
                continue
            ax.hist(values, bins=edges, density=True, color=color, alpha=0.3)
            ax.hist(
                values,
                bins=edges,
                density=True,
                histtype="step",
                color=color,
                linewidth=1.5,
            )

        ax.set_xlabel(latex_label(name))
        # few ticks: at the report font size, more run into the next panel's
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        # the height of a normalised density has no meaning of its own to read
        ax.set_yticks([])
        ax.set_ylim(bottom=0)

    handles = [
        Patch(facecolor=color, edgecolor=color, alpha=0.5, label=fit.plot_label)
        for color, fit in zip(colors, fits)
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncols=min(len(handles), 4),
        frameon=False,
    )
    return fig


@figure
def plot_global_information_heatmap(
    global_information_table,
    cmap="Blues",
    value_fmt="{:.1f}",
    colorbar=False,
):
    """Plot the global information matrix as a heatmap.

    Parameters
    ----------
    global_information_table : pd.DataFrame
        Index = coeff_names, columns = source_names.
    cmap : str, optional
        Colormap. Sequential by default, as for
        :func:`plot_fisher_diagonals_heatmap`: the values are percentage shares.
    value_fmt : str, optional
        Format of the per-cell annotation, in percent.
    colorbar : bool, optional
        Whether to draw the colour scale alongside. Off by default: rows sum to
        100%, so the annotations already say what a cell is worth.
    """
    shares = global_information_table.values * 100
    # 0-100% unless a share falls outside it (see global_information_matrix):
    # clipping would hide exactly the cells the warning is about
    return _plot_heatmap(
        shares,
        global_information_table.index.tolist(),
        global_information_table.columns.tolist(),
        vmin=min(0.0, shares.min()),
        vmax=max(100.0, shares.max()),
        cmap=cmap,
        value_fmt=value_fmt,
        colorbar=colorbar,
    )
