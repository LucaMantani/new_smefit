"""Gaussian constraints on the beta-decay nuisance parameters.

Each class adds a single term

    chi2 = ((param - central) / sigma)**2

to the likelihood. This is how a Gaussian prior reaches fits that only see the
chi2 (``run_hessian_fit``, gradient descent): those never read the runcard
priors. Samplers that do (BlackJAX, UltraNest) would count it twice, so pair a
constraint with a uniform prior there, or drop the constraint.

The runcard key selects the parameter, so there is one class per parameter::

    external_chi2:
      GaussConstraintDRV:
        path: new_smefit/external_chi2/low_energy/gaussian_constraints.py
        central: 0.02471
        sigma: 0.00025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp

    from smefit.core import CoefficientGroup


class _GaussConstraint:
    """Gaussian constraint on the coefficient named by ``_param_name``."""

    _param_name: str

    def __init__(
        self,
        coefficients: CoefficientGroup,
        central: float,
        sigma: float,
        rge_dict: dict | None = None,
    ) -> None:
        if self._param_name not in coefficients.coeff_index:
            raise ValueError(
                f"{type(self).__name__}: coefficient '{self._param_name}' "
                "is not declared in the runcard."
            )
        if sigma <= 0.0:
            raise ValueError(f"{type(self).__name__}: sigma must be positive.")
        self.coefficients = coefficients
        # Index into the *full* coefficient vector returned by resolve().
        self._idx = coefficients.coeff_index[self._param_name]
        self._central = float(central)
        self._sigma = float(sigma)
        self.num_data = 1

    def compute_chi2(self, coefficient_values: jnp.ndarray) -> jnp.ndarray:
        # coefficient_values holds the free parameters only.
        value = self.coefficients.resolve(coefficient_values)[self._idx]
        return ((value - self._central) / self._sigma) ** 2


class GaussConstraintDRV(_GaussConstraint):
    """Gaussian constraint on DRV, the universal radiative correction."""

    _param_name = "DRV"


class GaussConstraintEta1(_GaussConstraint):
    """Gaussian constraint on eta1."""

    _param_name = "eta1"


class GaussConstraintEta2(_GaussConstraint):
    """Gaussian constraint on eta2."""

    _param_name = "eta2"


class GaussConstraintEta3(_GaussConstraint):
    """Gaussian constraint on eta3."""

    _param_name = "eta3"


class GaussConstraintVud(_GaussConstraint):
    """Gaussian constraint on Vud."""

    _param_name = "Vud"
