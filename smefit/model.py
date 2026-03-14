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
        # Keep coefficients that appear in theory operator list
        self.operators_to_keep = [
            c for c in self.coefficients.coefficients if c.name in self.theory.operators
        ]
        vars_used = {
            var for c in self.coefficients.coefficients if c.vars for var in c.vars
        }
        dropped = [
            c.name
            for c in self.coefficients.coefficients
            if c.name not in self.theory.operators and c.name not in vars_used
        ]
        if dropped:
            log.warning(
                "The following coefficients are not present in any theory dataset "
                "and are not used to constrain other coefficients.\n"
                "They will have no effect on predictions: %s",
                dropped,
            )

        # Indices in theory operator ordering (column/axis ordering)
        theory_index = {op: i for i, op in enumerate(self.theory.operators)}
        self.op_indices = [theory_index[c.name] for c in self.operators_to_keep]

        # Linear corrections: (n_data, n_keep)
        self.lin_corr = self.theory.eft_lin_pred[:, self.op_indices]

        if self.use_quad:
            # Quadratic corrections: (n_data, n_keep, n_keep)
            Q = self.theory.eft_quad_pred
            self.quad_corr = Q[:, self.op_indices, :][:, :, self.op_indices]

        # Indices into the CoefficientGroup.resolve output for operators_to_keep
        self.keep_coeff_indices = jnp.array(
            [self.coefficients.coeff_index[c.name] for c in self.operators_to_keep]
        )

    @jax.jit(static_argnames=("self",))
    def forward_map(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Predict theory values given free coefficient values."""
        all_coeffs = self.coefficients.resolve(coeffs)
        coeffs_derived = all_coeffs[self.keep_coeff_indices]

        predictions = self.theory.sm_pred + self.lin_corr @ coeffs_derived
        if self.use_quad:
            quad_contrib = jnp.einsum(
                "ijk,j,k->i", self.quad_corr, coeffs_derived, coeffs_derived
            )
            predictions += quad_contrib
        return predictions
