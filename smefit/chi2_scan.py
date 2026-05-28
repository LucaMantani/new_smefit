"""
smefit.chi2_scan

Per-coefficient 1D chi2 scan using reportengine's collect + NSList pattern.
Each free coefficient gets its own namespace via ``individual_fit_coefficients``.
"""

import logging

import jax.numpy as jnp
import numpy as np
from reportengine import collect

log = logging.getLogger(__name__)


def individual_chi2_scan(
    individual_chi2,
    individual_coefficients,
    individual_fit_coefficient,
    chi2_scan_settings=None,
):
    """Scan the chi2 along one free coefficient, holding all others at zero.

    Returns a dict ``{coeff_name: {"points": [...], "chi2": [...]}}`` to be
    collected by ``individual_chi2_scans`` and written by ``run_chi2_scan``.
    """
    n_points = chi2_scan_settings.get("n_points")
    spec = individual_coefficients.prior_specs().get(individual_fit_coefficient)

    if spec is not None and spec.get("dist") == "uniform":
        low, high = float(spec["low"]), float(spec["high"])
    else:
        log.warning(
            "Coefficient '%s' does not have a uniform prior with 'low'/'high'; "
            "using default scan range [-1, 1].",
            individual_fit_coefficient,
        )
        low, high = -1.0, 1.0

    scan_points = np.linspace(low, high, n_points)
    chi2_values = [float(individual_chi2(jnp.array([v]))) for v in scan_points]
    return {
        individual_fit_coefficient: {
            "points": scan_points.tolist(),
            "chi2": chi2_values,
        }
    }


individual_chi2_scans = collect(
    "individual_chi2_scan", ("individual_fit_coefficients",)
)
