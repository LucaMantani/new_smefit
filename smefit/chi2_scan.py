"""
smefit.chi2_scan

Per-coefficient 1D chi2 scan using reportengine's collect + NSList pattern.
Each free coefficient gets its own namespace via ``individual_fit_coefficients``.
"""

import logging
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from reportengine import collect

from smefit.chi2 import Chi2
from smefit.core import CoefficientGroup

log = logging.getLogger(__name__)


def individual_chi2_scan(
    individual_chi2: Chi2,
    individual_coefficients: CoefficientGroup,
    individual_fit_coefficient: str,
    chi2_scan_settings: Mapping | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Scan the chi2 along one free coefficient, holding all others at zero.

    Returns a dict ``{coeff_name: {"points": [...], "chi2": [...]}}`` to be
    collected by ``individual_chi2_scans`` and rendered by ``chi2_scan_table``
    / ``plot_chi2_scan``.
    """
    n_points = (chi2_scan_settings or {}).get("n_points", 50)
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


def individual_mass_scan_point(
    individual_mass_chi2: Chi2, individual_mass_scale: int | float
) -> float:
    """Evaluate chi2 at a single mass scan point."""
    return float(individual_mass_chi2(jnp.array([float(individual_mass_scale)])))


individual_mass_scan_points = collect(
    "individual_mass_scan_point", ("individual_mass_scales",)
)
