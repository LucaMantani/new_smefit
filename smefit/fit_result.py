"""
smefit.fit_result.py

FitResult dataclass shared across fitting routines.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax.numpy as jnp


@dataclass
class FitResult:
    """Container for the result of a fit.

    Attributes
    ----------
    free_parameters : list[str]
        Names of free (fitted) parameters, in CoefficientGroup order.
    best_fit_point : dict[str, float]
        Best-fit value for every coefficient (free and derived).
    max_loglikelihood : float
        Maximum log-likelihood value: ``-chi2_best / 2``.
    num_data : int
        Total number of data points used in the fit.
    logz : float or None
        Log evidence (only filled by nested-sampling routines).
    samples : dict[str, jnp.ndarray] or None
        Posterior samples for every coefficient,
        shape ``(n_samples,)`` per entry.
    """

    free_parameters: List[str]
    best_fit_point: Dict[str, float]
    max_loglikelihood: float
    num_data: int
    logz: Optional[float] = None
    samples: Optional[Dict[str, jnp.ndarray]] = None

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def n_free(self) -> int:
        return len(self.free_parameters)

    @property
    def ndof(self) -> int:
        return self.num_data - self.n_free

    @property
    def chi2_val(self) -> float:
        return float(-2.0 * self.max_loglikelihood)

    @property
    def chi2_ndof(self) -> float:
        return self.chi2_val / self.ndof if self.ndof > 0 else float("nan")

    @property
    def uncertainties(self) -> Dict[str, float]:
        """Standard deviation of samples per coefficient."""
        if self.samples is None:
            return {}
        return {name: float(jnp.std(vals)) for name, vals in self.samples.items()}

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("Fit Result")
        lines.append("=" * 60)
        lines.append(f"  n_data   = {self.num_data}")
        lines.append(f"  n_free   = {self.n_free}")
        lines.append(f"  ndof     = {self.ndof}")
        lines.append(f"  chi2     = {self.chi2_val:.4f}")
        lines.append(f"  chi2/dof = {self.chi2_ndof:.4f}")
        if self.logz is not None:
            lines.append(f"  log Z    = {self.logz:.4f}")
        lines.append("")
        lines.append(f"  {'Coefficient':<20} {'Best fit':>12} {'Uncertainty':>12}")
        lines.append("  " + "-" * 46)
        unc = self.uncertainties
        for name, val in self.best_fit_point.items():
            u = unc.get(name, float("nan"))
            lines.append(f"  {name:<20} {val:>12.6f} {u:>12.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)
