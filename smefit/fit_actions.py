"""
smefit.fit_actions.py

Reportengine fit actions for smefit.
"""

from smefit.utils import write_fit_result


def run_analytic_fit(analytic_fit, output_path):
    """Print and save the result of the analytic fit.

    Parameters
    ----------
    analytic_fit : FitResult
        Result produced by the ``analytic_fit`` provider node.
    output_path : pathlib.Path
        Path to the output directory for writing results to disk.
    """
    analytic_fit.print_summary()
    write_fit_result(analytic_fit, output_path)
