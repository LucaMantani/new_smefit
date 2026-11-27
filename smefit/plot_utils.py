"""
smefit.plot_utils.py

Helpers shared by the report figures and tables.

Deliberately not a provider module (it is absent from ``smefit_providers`` in
``smefit/app.py``): it holds helpers the report routines call, not nodes
reportengine resolves.
"""

import logging

import numpy as np
from matplotlib import rc

log = logging.getLogger(__name__)


def set_plot_style():
    """Apply the shared matplotlib style, LaTeX text rendering included.

    Call this before building a figure. It is deliberately *not* done at import
    time: these are process-wide matplotlib settings, so a module that applied
    them on import would silently switch every plot in the importing program —
    tests and API users included — over to LaTeX, and then fail wherever no
    LaTeX installation exists. Every figure in ``smefit.figures`` calls it, and
    a standalone script that wants the same look should call it too.
    """
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
    rc("text", usetex=True)
    rc("text.latex", preamble=r"\usepackage{amssymb}")


def select_params(names, params_to_plot, context=None):
    """Return the subset of *names* to show, in ``params_to_plot`` order.

    Parameters
    ----------
    names : iterable of str
        The coefficients the caller has, in its own order.
    params_to_plot : list of str or str or None
        The coefficients the runcard asked for. None selects everything, which
        leaves *names* untouched. A bare string is taken as a one-element list,
        so ``params_to_plot: OtG`` works in a runcard.
    context : str, optional
        What is being selected from — a fit name, say. Only used to say which
        plot a message is about.

    Returns
    -------
    list of str
        The requested names that *names* has, in requested order, deduplicated.
        Names it does not have are dropped: one global ``params_to_plot`` is
        meant to serve several fits, which need not have fitted the same
        coefficients, so each shows the largest subset it can.

    Raises
    ------
    ValueError
        If nothing at all was selected. An empty plot is a misspelt list rather
        than a subset anybody asked for.
    """
    names = list(names)
    if params_to_plot is None:
        return names

    if isinstance(params_to_plot, str):
        params_to_plot = [params_to_plot]
    requested = list(dict.fromkeys(params_to_plot))

    available = set(names)
    selected = [p for p in requested if p in available]
    dropped = [p for p in requested if p not in available]

    where = f" of '{context}'" if context else ""
    if dropped:
        log.info(
            "params_to_plot: %s not among the coefficients%s, skipping.",
            ", ".join(dropped),
            where,
        )
    if not selected:
        raise ValueError(
            f"params_to_plot selected none of the coefficients{where}: "
            f"asked for {requested}, available are {names}."
        )
    return selected


def rescale_exponents(extents, floor=1e-10):
    """Return the power of ten each row of a plot is multiplied by to be read
    next to the widest one.

    The widest row is the reference and is left alone. Every other row gets the
    largest power of ten that keeps it inside the range the reference sets, so
    all of them end up between a tenth of that range and the whole of it. Rows
    already larger than a tenth of the widest are left alone too, hence a plot
    whose rows are comparable is not rescaled at all.

    Parameters
    ----------
    extents : iterable of float
        How far each row reaches from zero — the largest absolute value drawn
        on it — in the order of the rows.
    floor : float, optional
        A row whose extent is below this fraction of the widest is numerical
        zero rather than a narrow row, and is left alone: rescaling it would
        only magnify rounding noise.

    Returns
    -------
    list of int
        One non-negative exponent ``k`` per row, in the order given: the row
        is to be multiplied by ``10**k``.
    """
    extents = np.asarray(list(extents), dtype=float)
    exponents = np.zeros(len(extents), dtype=int)
    reference = extents.max(initial=0.0)
    nonzero = extents > floor * reference
    exponents[nonzero] = np.floor(np.log10(reference / extents[nonzero]))
    return exponents.tolist()


def _tick_labels_crowded(axis):
    """Whether two neighbouring major tick labels of *axis* are less than half
    a font size apart, measured as they will be rendered.

    Only the labels that get drawn count: a locator may return ticks past the
    view limits, and an inner panel hides its labels altogether.
    """
    low, high = sorted(axis.get_view_interval())
    tolerance = 1e-9 * (high - low)
    # asking for the labels is what brings the ticks in line with the locator
    axis.get_majorticklabels()
    labels = [
        tick.label1
        for tick in axis.get_major_ticks()
        if low - tolerance <= tick.get_loc() <= high + tolerance
        and tick.label1.get_visible()
        and tick.label1.get_text()
    ]
    if len(labels) < 2:
        return False
    extents = sorted(
        tuple(
            label.get_window_extent().intervalx
            if axis.axis_name == "x"
            else label.get_window_extent().intervaly
        )
        for label in labels
    )
    gap = 0.5 * labels[0].get_fontsize() * axis.get_figure().dpi / 72
    return any(start - end < gap for (_, end), (start, _) in zip(extents, extents[1:]))


def thin_tick_labels(axes, which, max_nbins):
    """Give *axes* the most major ticks whose labels stay clear of each other.

    No fixed tick count suits every coefficient: how many labels fit along an
    axis depends on how wide they come out — ``-0.025`` takes twice the room
    of ``0.2`` — and that is only known once the limits are. So the labels
    are measured as rendered, starting from ``max_nbins`` intervals and going
    down one at a time until neighbouring labels are at least half a font size
    apart.

    Call it once the limits, the tick label size and the figure size are
    final, since all three enter the measurement.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        Panels showing the same quantity along ``which``, a column of a corner
        plot for instance. They all get the same ticks, so their grids line
        up, whichever of them carries the labels.
    which : {"x", "y"}
        The axis to set the ticks of.
    max_nbins : int
        The most intervals between ticks to settle for, as ``nbins`` of
        :class:`matplotlib.ticker.MaxNLocator`.
    """
    for nbins in range(max_nbins, 0, -1):
        for ax in axes:
            ax.locator_params(axis=which, nbins=nbins)
        if not any(_tick_labels_crowded(getattr(ax, f"{which}axis")) for ax in axes):
            break
