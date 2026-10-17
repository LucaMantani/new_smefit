"""
smefit.plot_utils.py

Helpers shared by the report figures and tables.

Deliberately not a provider module (it is absent from ``smefit_providers`` in
``smefit/app.py``): it holds helpers the report routines call, not nodes
reportengine resolves.
"""

import logging

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
