"""
smefit.model.py

Model module of smefit, where the prediction model is defined.
"""

import jax
import jax.numpy as jnp


class EFTModel:
    """EFT model mapping coefficients to theory predictions."""

    def __init__(self, theory, coefficients, use_quad=False):
        self.theory = theory
        self.coefficients = coefficients
        self.use_quad = use_quad
        # Keep coefficients that appear in theory operator list
        self.operators_to_keep = [
            c for c in self.coefficients.coefficients if c.name in self.theory.operators
        ]

        # Indices in theory operator ordering (column/axis ordering)
        theory_index = {op: i for i, op in enumerate(self.theory.operators)}
        self.op_indices = [theory_index[c.name] for c in self.operators_to_keep]

        # Linear corrections: (n_data, n_keep)
        self.lin_corr = self.theory.eft_lin_pred[:, self.op_indices]

        if self.use_quad:
            # Quadratic corrections: (n_data, n_keep, n_keep)
            Q = self.theory.eft_quad_pred
            self.quad_corr = Q[:, self.op_indices, :][:, :, self.op_indices]

    @jax.jit(static_argnames=("self",))
    def forward_map(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Predict theory values given coefficient values."""
        # Build coefficients entering theory from the free ones
        coeffs_derived = self.derive_coeffs(coeffs)

        # Compute predictions
        predictions = self.theory.sm_pred + self.lin_corr @ coeffs_derived
        if self.use_quad:
            # Add quadratic contributions
            quad_contrib = jnp.einsum(
                "ijk,j,k->i", self.quad_corr, coeffs_derived, coeffs_derived
            )
            predictions += quad_contrib
        return predictions

    def derive_coeffs(self, free_coeffs: jnp.ndarray) -> jnp.ndarray:
        """Derive full set of coefficient values from free coefficients.

        For each coefficient in the model:
        - If it's free: use the value from free_coeffs
        - If it's constrained: call its constrain method with free coefficient values

        Parameters
        ----------
        free_coeffs : jnp.ndarray
            Values of the free coefficients, ordered by self.coefficients.free_coeffs

        Returns
        -------
        jnp.ndarray
            Full set of coefficient values for operators in self.operators_to_keep
        """
        derived_coeffs = []

        free_coeff_dict = {
            fc.name: val for fc, val in zip(self.coefficients.free_coeffs, free_coeffs)
        }

        # Build the derived coefficients for operators_to_keep
        for coeff in self.operators_to_keep:
            if coeff.free:
                # Use the free coefficient value
                derived_coeffs.append(free_coeff_dict[coeff.name])
            else:
                # Use the constrain method
                # If the coefficient has vars, pass the corresponding free coefficient values
                if coeff.vars:
                    args = tuple(free_coeff_dict[var] for var in coeff.vars)
                    derived_coeffs.append(coeff.constrain(*args))
                else:
                    # Constant coefficient, no arguments needed
                    derived_coeffs.append(coeff.constrain())

        return jnp.array(derived_coeffs)
