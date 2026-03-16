"""
smefit.model.py

Model module of smefit, where the prediction model is defined.
"""

import logging
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

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

    def __init__(self, theory, coefficients, use_quad=False, rge_matrix=None):
        self.theory = theory
        self.coefficients = coefficients
        self.use_quad = use_quad

        if rge_matrix is not None:
            self._apply_rge(theory, *rge_matrix)
        else:
            self._setup_direct(theory)

    def _setup_direct(self, theory):
        """Set up lin/quad corrections by intersecting theory and declared coefficients."""
        coefficients = self.coefficients
        theory_params = set(theory.operators)
        declared = {c.name for c in coefficients.coefficients}
        vars_used = {var for c in coefficients.coefficients if c.vars for var in c.vars}

        active_names = sorted(theory_params & declared)

        if not active_names:
            raise ValueError(
                "None of the declared coefficients match any operator in the theory files.\n"
                f"  Coefficients declared      : {sorted(declared)}\n"
                f"  Operators found in datasets: {sorted(theory_params)}\n"
                "Check that coefficient names in the runcard match operator names in the theory files."
            )

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

        theory_idx = {c: i for i, c in enumerate(theory.operators)}
        t_indices = [theory_idx[name] for name in active_names]

        self.lin_corr = theory.eft_lin_pred[:, t_indices]
        if self.use_quad:
            Q = theory.eft_quad_pred
            self.quad_corr = Q[:, t_indices, :][:, :, t_indices]

        self.active_coeff_indices = jnp.array(
            [coefficients.coeff_index[name] for name in active_names]
        )

    def _apply_rge(self, theory, stacked_mats, operators_to_keep):
        """Set up lin/quad corrections by contracting theory tables with RGE matrices."""
        coeff_names = self.coefficients.names  # sorted by CoefficientGroup constructor
        rge_obs_ops = sorted(operators_to_keep.keys())
        n_obs, n_init = len(rge_obs_ops), len(coeff_names)

        R = stacked_mats
        if R.shape[0] == 1:
            R = jnp.broadcast_to(R, (theory.n_data, n_obs, n_init))

        theory_op_index = {op: i for i, op in enumerate(theory.operators)}

        matched_ops = [op for op in rge_obs_ops if op in theory_op_index]
        if not matched_ops:
            raise ValueError(
                "The likelihood does not depend on any of the declared free coefficients.\n"
                f"  RGE observable-basis operators: {sorted(rge_obs_ops)}\n"
                f"  Operators found in datasets   : {sorted(theory_op_index.keys())}\n"
                "Check that the RGE operator names match the operator names in the theory files."
            )

        lin_aligned = np.zeros((theory.n_data, n_obs))
        for ri, obs_op in enumerate(rge_obs_ops):
            if obs_op in theory_op_index:
                lin_aligned[:, ri] = np.array(
                    theory.eft_lin_pred[:, theory_op_index[obs_op]]
                )
        self.lin_corr = jnp.einsum("di,dik->dk", jnp.asarray(lin_aligned), R)

        if self.use_quad:
            quad_aligned = np.zeros((theory.n_data, n_obs, n_obs))
            for ri, op1 in enumerate(rge_obs_ops):
                if op1 not in theory_op_index:
                    continue
                for ci, op2 in enumerate(rge_obs_ops):
                    if ci < ri or op2 not in theory_op_index:
                        continue
                    gi, gj = theory_op_index[op1], theory_op_index[op2]
                    if gi > gj:
                        gi, gj = gj, gi
                    quad_aligned[:, ri, ci] = np.array(theory.eft_quad_pred[:, gi, gj])
            new_quad = jnp.einsum("dij,dil,djr->dlr", jnp.asarray(quad_aligned), R, R)
            i_idx, j_idx = np.tril_indices(n_init, k=-1)
            self.quad_corr = new_quad.at[:, i_idx, j_idx].set(0.0)

        self.active_coeff_indices = jnp.array(
            [self.coefficients.coeff_index[name] for name in coeff_names]
        )

    @jax.jit(static_argnames=("self",))
    def forward_map(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Predict theory values given free coefficient values."""
        active = self.coefficients.resolve(coeffs)[self.active_coeff_indices]
        predictions = self.theory.sm_pred + self.lin_corr @ active
        if self.use_quad:
            predictions += jnp.einsum("ijk,j,k->i", self.quad_corr, active, active)
        return predictions
