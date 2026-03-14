"""
smefit.model.py

Model module of smefit, where the prediction model is defined.
"""

import logging
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for prediction models.

    A model maps a parameter vector to theory predictions.
    Concrete subclasses must implement :meth:`forward_map`.

    Contract for JAX JIT compatibility
    ------------------------------------
    If ``forward_map`` is decorated with ``@jax.jit(static_argnames=("self",))``,
    ``self`` must be hashable and all JAX arrays stored on it must be fixed at
    construction time (they will be treated as compile-time constants).
    """

    @abstractmethod
    def forward_map(self, params: jnp.ndarray) -> jnp.ndarray:
        """Map parameter values to theory predictions.

        Parameters
        ----------
        params : jnp.ndarray
            Free parameter values.

        Returns
        -------
        jnp.ndarray
            Theory predictions with shape (n_data,).
        """


class EFTModel(BaseModel):
    """EFT model mapping Wilson coefficients to theory predictions."""

    def __init__(self, theory, coefficients, use_quad=False):
        self.theory = theory
        self.coefficients = coefficients
        self.use_quad = use_quad

        # Coefficients that actively enter predictions (intersection of declared and theory)
        theory_params = set(theory.operators)
        declared = {c.name for c in coefficients.coefficients}
        vars_used = {var for c in coefficients.coefficients if c.vars for var in c.vars}

        active_names = sorted(theory_params & declared)

        inactive = [
            n for n in declared if n not in theory_params and n not in vars_used
        ]
        if inactive:
            log.warning(
                "The following coefficients are not present in any theory dataset "
                "and are not used to constrain other coefficients.\n"
                "They will have no effect on predictions: %s",
                inactive,
            )

        # Slice theory correction matrices down to active coefficients only
        theory_idx = {op: i for i, op in enumerate(theory.operators)}
        t_indices = [theory_idx[name] for name in active_names]

        self.lin_corr = theory.eft_lin_pred[:, t_indices]
        if use_quad:
            Q = theory.eft_quad_pred
            self.quad_corr = Q[:, t_indices, :][:, :, t_indices]

        # Indices into CoefficientGroup.resolve() output, used in forward_map
        self.active_coeff_indices = jnp.array(
            [coefficients.coeff_index[name] for name in active_names]
        )

    @jax.jit(static_argnames=("self",))
    def forward_map(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Predict theory values given free coefficient values."""
        active = self.coefficients.resolve(coeffs)[self.active_coeff_indices]
        predictions = self.theory.sm_pred + self.lin_corr @ active
        if self.use_quad:
            predictions += jnp.einsum("ijk,j,k->i", self.quad_corr, active, active)
        return predictions
