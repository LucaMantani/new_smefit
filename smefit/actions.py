"""
smefit.actions.py

Reportengine actions for smefit.
"""


def run_analytic_fit(analytic_fit):
    """Print the result of the analytic fit.

    Parameters
    ----------
    analytic_fit : FitResult
        Result produced by the ``analytic_fit`` provider node.
    """
    analytic_fit.print_summary()
