"""
smefit.fit_actions.py

Reportengine fit actions for smefit.
"""

from smefit.fit_result import FitResultGroup


def run_ultranest_fit(ultranest_fit, output_path):
    """Print and save the result of the UltraNest nested-sampling fit."""
    ultranest_fit.print_summary()
    ultranest_fit.write(output_path)


def run_analytic_fit(analytic_fit, output_path):
    """Print and save the result of the analytic fit."""
    analytic_fit.print_summary()
    analytic_fit.write(output_path)


def run_blackjax_fit(blackjax_fit, output_path):
    """Print and save the result of the BlackJAX nested-sampling fit."""

    blackjax_fit.print_summary()
    blackjax_fit.write(output_path)


def run_individual_analytic_fits(individual_analytic_fits, output_path):
    """Print and save individual analytic fit results."""
    group = FitResultGroup(individual_analytic_fits)
    group.print_summary()
    group.write_results(output_path)


def run_individual_ultranest_fits(individual_ultranest_fits, output_path):
    """Print and save individual ultranest fit results."""
    group = FitResultGroup(individual_ultranest_fits)
    group.print_summary()
    group.write_results(output_path)
