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
            self._apply_rge(
                theory,
                rge_matrix.stacked_mats,
                rge_matrix.obs_operators,
                rge_matrix.init_operators,
            )
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

    def _apply_rge(self, theory, stacked_mats, obs_operators, init_operators):
        """Set up lin/quad corrections by contracting theory tables with RGE matrices."""
        coeff_names = self.coefficients.names  # sorted by CoefficientGroup constructor
        rge_obs_ops = obs_operators  # already sorted
        n_obs = len(rge_obs_ops)

        # Slice R columns to only the declared coefficients (mirrors _setup_direct intersection)
        init_op_idx = {name: i for i, name in enumerate(init_operators)}
        active_init_names = [name for name in coeff_names if name in init_op_idx]
        col_indices = [init_op_idx[name] for name in active_init_names]
        n_init = len(active_init_names)

        R = stacked_mats[:, :, col_indices]
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
            self.quad_corr = jnp.einsum(
                "dij,dil,djr->dlr", jnp.asarray(quad_aligned), R, R
            )

        self.active_coeff_indices = jnp.array(
            [self.coefficients.coeff_index[name] for name in active_init_names]
        )

    @jax.jit(static_argnames=("self",))
    def forward_map(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Predict theory values given free coefficient values."""
        active = self.coefficients.resolve(coeffs)[self.active_coeff_indices]
        predictions = self.theory.sm_pred + self.lin_corr @ active
        if self.use_quad:
            predictions += jnp.einsum("ijk,j,k->i", self.quad_corr, active, active)
        return predictions
