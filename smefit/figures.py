"""
smefit.figures.py

Functions for creating figures and plots.
"""

import logging

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from reportengine.figure import figure, figuregen

from smefit.op_to_latex import coeff_info_latex
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

    coeff_labels = [coeff_info_latex.get(name, name) for name in coeff_names]
    ax.set_yticks(range(n_coeffs))
    ax.set_yticklabels(coeff_labels, fontsize=14)

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
def plot_chi2_scan(individual_chi2_scans):
    """Plot the 1D chi2 scan for each free coefficient.

    Yields one ``(figure, coeff_name)`` per scanned coefficient, with the
    coefficient value on the x axis and chi2 on the y axis.

    Parameters
    ----------
    individual_chi2_scans : list[dict]
        Each entry maps ``{coeff_name: {"points": [...], "chi2": [...]}}``.
    """
    set_plot_style()

    results = {k: v for d in individual_chi2_scans for k, v in d.items()}
    for name, data in results.items():
        label = coeff_info_latex.get(name, name)
        # Not plt.subplots, here or in any @figuregen: reportengine materialises
        # the whole generator before saving and never closes what it saved, so
        # figures registered with pyplot pile up (and warn past 20). A bare
        # Figure is not tracked by pyplot and is freed once written.
        fig = Figure(figsize=(6, 5), layout="constrained")
        ax = fig.subplots()
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
def plot_pca_spectrum(pca):
    """Plot the eigenvalue spectrum, with the flat-direction threshold marked.

    The one picture of the whole analysis: bars falling below the dashed line
    are the directions the data do not constrain.

    Parameters
    ----------
    pca : smefit.pca.PCA
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
    ax.set_xticklabels(pca.component_names, rotation=90, fontsize=12)
    ax.set_ylabel(r"$\lambda_i / \lambda_{\rm max}$", fontsize=16)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig


@figure
def plot_posterior_correlations(
    fit, params_to_plot=None, cmap="RdBu_r", value_fmt="{:.2f}", colorbar=True
):
    """Plot the posterior correlations of one fit's free coefficients.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one heatmap per fit, each headed with the fit it is drawn from.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
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
    labels = [coeff_info_latex.get(name, name) for name in selected]
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


def _tex_escape(text):
    """Escape the characters LaTeX treats specially in text mode.

    Dataset names are plain identifiers full of underscores, and the figures
    render text through usetex, where a bare ``_`` is a math-mode error.
    """
    for char in ("\\", "_", "%", "&", "#", "$"):
        text = text.replace(char, "\\" + char)
    return text


def _chi2(residuals, covmat):
    """``r^T C^{-1} r``, by a solve rather than an explicit inverse."""
    return float(residuals @ np.linalg.solve(covmat, residuals))


def _stairs_band(axis, low, high, edges, **kwargs):
    """Fill between *low* and *high*, constant across each bin."""
    return axis.stairs(high, edges, baseline=low, fill=True, linewidth=0, **kwargs)


# Fixed so the EFT band of a report is the same every time it is rebuilt.
_EFT_BAND_SEED = 0


def _gaussian_draws(covmat, n_draws, rng):
    """*n_draws* draws of :math:`N(0, C)`, shape ``(n_draws, len(C))``.

    Negative eigenvalues of *covmat* are set to zero, which makes it the
    nearest positive-semidefinite matrix. Theory covmats in the database are
    not all exactly PSD — rounding at 1e-15 is common, and a few carry
    negative eigenvalues near 1e-4 of the largest, from rounded correlations —
    and a Gaussian with a negative variance has no draws to give.
    """
    eigvals, eigvecs = np.linalg.eigh(np.asarray(covmat))
    root = eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))
    return rng.standard_normal((n_draws, len(eigvals))) @ root.T


def _eft_predictions(fit, eft_model, coefficients, theory_covmat=None):
    """EFT predictions at the fit's best-fit point and over its posterior.

    The predictions are those of *this* runcard's ``eft_model``, evaluated at
    the values the fit found for the runcard's free coefficients — so the
    datasets plotted need not be the ones that were fitted.

    Each posterior sample :math:`c_i` gives one prediction,
    :math:`\\sigma_i = f(c_i) + \\theta_i` with
    :math:`\\theta_i \\sim N(0, C_{\\rm th})`: the full model ``forward_map``,
    quadratic corrections and RGE running included, plus a draw of the theory
    uncertainty, correlated across data points as in ``theory_covmat``. That is
    the predictive distribution the likelihood assumes, so its quantiles are
    exact whatever its shape; for a linear model and a Gaussian posterior they
    reduce to the best fit :math:`\\pm \\sqrt{L\\Sigma L^T + C_{\\rm th}}`.

    Parameters
    ----------
    theory_covmat : array-like, shape (n_data, n_data), optional
        Covariance of the theory uncertainty added to every sample. None adds
        none, and the samples are the posterior's spread alone.

    Returns
    -------
    best : np.ndarray, shape (n_data,)
    samples : np.ndarray, shape (n_samples, n_data)
    """
    results = fit.fit_results
    free = coefficients.free_names
    missing = [
        name
        for name in free
        if name not in results.best_fit_point or name not in (results.samples or {})
    ]
    if missing:
        raise ValueError(
            f"Fit '{fit.fit_name}' has no best-fit value or posterior samples "
            f"for {missing}, which are free in this runcard. Declare as free "
            "only coefficients the fit determined."
        )
    if fit.use_quad != eft_model.use_quad:
        log.warning(
            "Fit '%s' was run with use_quad=%s, but the predictions are drawn "
            "with use_quad=%s.",
            fit.fit_name,
            fit.use_quad,
            eft_model.use_quad,
        )

    best_point = jnp.array([results.best_fit_point[name] for name in free])
    posterior = jnp.stack([jnp.asarray(results.samples[name]) for name in free], 1)
    # Batched rather than vmapped over every sample at once: with quadratic
    # corrections the intermediate is (n_samples, n_data, n_ops, n_ops).
    samples = np.asarray(jax.lax.map(eft_model.forward_map, posterior, batch_size=256))
    if theory_covmat is not None:
        rng = np.random.default_rng(_EFT_BAND_SEED)
        samples = samples + _gaussian_draws(theory_covmat, len(samples), rng)
    return np.asarray(eft_model.forward_map(best_point)), samples


@figuregen
def plot_data_theory_comparison(
    fit, data, theory, eft_model, coefficients, fit_covmat, use_theory_covmat=False
):
    """Compare the data with the SM and with the EFT at a fit's posterior.

    Yields one ``(figure, dataset_name)`` per dataset of the runcard. The upper
    panel shows

    - the measurements, with the total experimental uncertainty as error bar
      (statistical and every systematic, correlated or not, in quadrature —
      the diagonal of the experimental covmat);
    - the SM prediction, banded by its theory uncertainty (the diagonal of the
      theory covmat), whether or not ``use_theory_covmat`` puts it in the fit;
    - the EFT prediction at the fit's best-fit point, banded by the 16th to
      84th percentiles of the predictions over the posterior, with the theory
      uncertainty sampled in when ``use_theory_covmat`` is set (see
      :func:`_eft_predictions`). With quadratic corrections the band need not
      be symmetric about the best fit, nor even contain it.

    The predictions are drawn constant across each bin. The lower panel shows
    the same, divided by the SM. The legend gives the :math:`\\chi^2` per
    data point of the SM and of the EFT best fit.

    The data and the theory come from this runcard — its ``datasets``,
    ``coefficients``, ``use_quad`` and ``rge`` — and only the coefficient values
    from ``fit``, so a runcard listing several ``fits`` overlays nothing: it
    gets one set of figures per fit, named after it.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run joint fit, loaded from disk. Every free coefficient of
        this runcard must be one it determined.
    data : smefit.core.DataGroup
    theory : smefit.core.TheoryGroup
    eft_model : smefit.model.EFTModel
    coefficients : smefit.core.CoefficientGroup
    fit_covmat : jnp.ndarray
        The covariance of the likelihood, so the :math:`\\chi^2` follows
        ``use_t0`` and ``use_theory_covmat`` as a fit would. Only the
        dataset's diagonal block enters, as in ``datasets_chi2``: correlations
        across datasets are ignored.
    use_theory_covmat : bool, optional
        Whether the theory uncertainty is part of the EFT band, as it is part
        of the likelihood. The runcard's own key, so the band follows what the
        fit assumed.

    Raises
    ------
    ValueError
        If the fit was run one coefficient at a time — its samples are
        independent 1D posteriors, not points in the joint space the
        predictions need — or lacks a best-fit value or posterior samples for
        a coefficient free in this runcard.
    """
    if fit.individual_fit:
        raise ValueError(
            f"Fit '{fit.fit_name}' was run one coefficient at a time, so it has "
            "no joint posterior to evaluate the predictions on."
        )
    if data.names != theory.names:
        raise ValueError(
            f"Data and theory contain different datasets.\n"
            f"  data:   {data.names}\n"
            f"  theory: {theory.names}"
        )

    set_plot_style()

    eft_best, eft_samples = _eft_predictions(
        fit,
        eft_model,
        coefficients,
        theory.sm_covmat if use_theory_covmat else None,
    )
    eft_low, eft_high = np.percentile(eft_samples, [16, 84], axis=0)
    sm = np.asarray(theory.sm_pred)
    sm_err = np.sqrt(np.diag(np.asarray(theory.sm_covmat)))
    cv = np.asarray(data.cv)
    exp_err = np.sqrt(np.diag(np.asarray(data.exp_covmat)))
    covmat = np.asarray(fit_covmat)

    offset = 0
    for name, ndata in zip(data.names, data.ndata_list):
        idx = slice(offset, offset + ndata)
        offset += ndata
        x = np.arange(1, ndata + 1)
        block = covmat[idx, idx]
        sm_chi2 = r"\chi^2/N_{{\rm dat}} = {:.2f}".format(
            _chi2(cv[idx] - sm[idx], block) / ndata
        )
        eft_chi2 = r"\chi^2/N_{{\rm dat}} = {:.2f}".format(
            _chi2(cv[idx] - eft_best[idx], block) / ndata
        )
        has_th_err = bool(sm_err[idx].any())

        # a bare Figure, not plt.subplots: see plot_chi2_scan
        fig = Figure(figsize=(max(7, ndata * 0.4), 7), layout="constrained")
        ax, ax_ratio = fig.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
        # The theory is drawn across its whole bin, a central line over its
        # band; the data are points on top. The gids name each artist, which is
        # how the tests find them.
        edges = np.arange(0.5, ndata + 1.5)
        for axis, norm in ((ax, 1.0), (ax_ratio, sm[idx])):
            sm_band = (
                _stairs_band(
                    axis,
                    (sm[idx] - sm_err[idx]) / norm,
                    (sm[idx] + sm_err[idx]) / norm,
                    edges,
                    color="C0",
                    alpha=0.3,
                    zorder=2,
                    gid="sm_band",
                )
                if has_th_err
                else None
            )
            # baseline=None: an unfilled stairs otherwise closes down to zero
            # at both ends of the axis
            sm_line = axis.stairs(
                sm[idx] / norm,
                edges,
                baseline=None,
                color="C0",
                linewidth=1.5,
                zorder=3,
                gid="sm",
            )
            eft_band = _stairs_band(
                axis,
                eft_low[idx] / norm,
                eft_high[idx] / norm,
                edges,
                color="C1",
                alpha=0.3,
                zorder=2.1,
                gid="eft_band",
            )
            eft_line = axis.stairs(
                eft_best[idx] / norm,
                edges,
                baseline=None,
                color="C1",
                linewidth=1.5,
                zorder=3.1,
                gid="eft",
            )
            data_points = axis.errorbar(
                x,
                cv[idx] / norm,
                # an error must stay positive when divided by a negative SM
                yerr=exp_err[idx] / np.abs(norm),
                fmt="o",
                color="black",
                markersize=4,
                capsize=3,
                elinewidth=1,
                zorder=4,
            )
            axis.grid(alpha=0.3)

        # Each theory entry pairs a line with its band.
        handles_labels = [
            (
                (sm_band, sm_line) if has_th_err else sm_line,
                rf"$\mathrm{{SM}},\ {sm_chi2}$",
            ),
            (
                (eft_band, eft_line),
                # the fit's label is the runcard's, already LaTeX or plain text
                rf"$\mathrm{{EFT}}$: {fit.plot_label}, ${eft_chi2}$",
            ),
            (data_points, r"$\mathrm{Data,\ stat.} + \mathrm{syst.}$"),
        ]
        ax.legend(*zip(*handles_labels), fontsize=12)

        ax.set_title(rf"$\mathrm{{{_tex_escape(name)}}}$", fontsize=18)
        ax.set_ylabel(r"$\mathrm{Observable}$", fontsize=16)
        ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax_ratio.set_ylabel(r"$\mathrm{Ratio\ to\ SM}$", fontsize=16)
        ax_ratio.set_xlabel(r"$\mathrm{Bin}$")
        ax_ratio.set_xticks(x)
        ax_ratio.set_xlim(edges[0], edges[-1])
        yield fig, name
