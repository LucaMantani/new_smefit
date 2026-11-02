"""
smefit.model.py

Model module of smefit, where the prediction model is defined.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    """EFT model mapping Wilson coefficients to theory predictions.

    ``use_theory_covmat`` says whether the theory uncertainty
    (``theory.sm_covmat``) is part of the model, as it is part of the
    likelihood of a fit run with it. ``forward_map`` is the central
    prediction either way; :meth:`posterior_predictive` samples the
    uncertainty in when it is on.
    """

    def __init__(
        self,
        theory,
        coefficients,
        use_quad=False,
        rge_matrix=None,
        use_theory_covmat=False,
    ):
        self.theory = theory
        self.coefficients = coefficients
        self.use_quad = use_quad
        self.use_theory_covmat = use_theory_covmat

        if rge_matrix is not None:
            self._apply_rge(theory, rge_matrix)
        else:
            self._setup_direct(theory)

    def _setup_direct(self, theory):
        """Set up lin/quad corrections by intersecting theory and declared coefficients."""
        coefficients = self.coefficients
        theory_params = set(theory.operators)
        declared = {c.name for c in coefficients.coefficients}
        vars_used = {var for c in coefficients.coefficients if c.vars for var in c.vars}

        active_names = sorted(theory_params & declared)
        inactive = [
            n for n in declared if n not in theory_params and n not in vars_used
        ]

        # The two warnings below are exclusive: when nothing is active, every
        # declared coefficient is inactive and the first message already says so.
        if not active_names:
            log.warning(
                "None of the declared coefficients enter the predictions of the "
                "datasets %s: they have no effect there.\n"
                "  Coefficients declared      : %s\n"
                "  Operators found in datasets: %s\n"
                "Check that coefficient names in the runcard match operator names "
                "in the theory files if this is not intended.",
                theory.names,
                sorted(declared),
                sorted(theory_params),
            )
        elif inactive:
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
            [coefficients.coeff_index[name] for name in active_names], dtype=int
        )

    def _apply_rge(self, theory, rge_matrix):
        """Set up lin/quad corrections by contracting theory tables with RGE matrices."""
        coeff_names = self.coefficients.names
        rge_obs_ops = rge_matrix.obs_operators
        n_obs = len(rge_obs_ops)

        # Check that all declared coefficients are present in the RGE initial-basis operators
        init_op_idx = {name: i for i, name in enumerate(rge_matrix.init_operators)}
        missing = [name for name in coeff_names if name not in init_op_idx]
        if missing:
            raise ValueError(
                f"Coefficients {missing} are declared in the fit but not found in the "
                "RGE initial-basis operators. Check the RGE configuration covers all "
                "declared coefficients."
            )
        col_indices = [init_op_idx[name] for name in coeff_names]
        n_init = len(coeff_names)

        # Slice the RGE matrix to only the columns corresponding to the declared coefficients
        R = rge_matrix.stacked_mats[:, :, col_indices]
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

        # Dummy identity permutation for the forward map,
        # RGE already spans all declared coefficients.
        self.active_coeff_indices = jnp.arange(len(coeff_names))

    @jax.jit(static_argnames=("self",))
    def forward_map(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Predict theory values given free coefficient values."""
        active = self.coefficients.resolve(coeffs)[self.active_coeff_indices]
        predictions = self.theory.sm_pred + self.lin_corr @ active
        if self.use_quad:
            predictions += jnp.einsum("ijk,j,k->i", self.quad_corr, active, active)
        return predictions

    def posterior_predictive(
        self, posterior: jnp.ndarray, seed: int = 0, batch_size: int = 256
    ) -> jnp.ndarray:
        """The model's prediction at every posterior sample.

        Sample :math:`c_i` gives :math:`\\sigma_i = f(c_i) + \\theta_i`: the
        full ``forward_map``, quadratic corrections and RGE running included,
        plus, when ``use_theory_covmat`` is on, a draw
        :math:`\\theta_i \\sim N(0, C_{\\rm th})` of the theory uncertainty,
        correlated across data points as in ``theory.sm_covmat``. That is the
        predictive distribution the likelihood assumes, so quantiles taken
        over the result are exact whatever its shape; for a linear model and a
        Gaussian posterior they reduce to
        :math:`f(\\hat c) \\pm \\sqrt{L\\Sigma L^T + C_{\\rm th}}`.

        Parameters
        ----------
        posterior : jnp.ndarray, shape (n_samples, n_free)
            One sample of the free coefficients per row, in
            ``coefficients.free_names`` order, as
            :meth:`smefit.fit_result.FitResult.posterior_array` gives them.
        seed : int, optional
            Seeds the theory-uncertainty draws, so the same posterior always
            gives the same predictions.
        batch_size : int, optional
            Samples evaluated at once. Batched rather than vmapped over all of
            them: with quadratic corrections the intermediate is
            ``(batch_size, n_data, n_ops, n_ops)``.

        Returns
        -------
        jnp.ndarray, shape (n_samples, n_data)
        """
        predictions = jax.lax.map(self.forward_map, posterior, batch_size=batch_size)
        if self.use_theory_covmat:
            predictions = predictions + gaussian_draws(
                self.theory.sm_covmat, len(predictions), jax.random.PRNGKey(seed)
            )
        return predictions


def gaussian_draws(covmat, n_draws, key):
    """*n_draws* draws of :math:`N(0, C)`, shape ``(n_draws, len(C))``.

    Negative eigenvalues of *covmat* are set to zero, which makes it the
    nearest positive-semidefinite matrix. Theory covmats in the database are
    not all exactly PSD — rounding at 1e-15 is common, and a few carry
    negative eigenvalues near 1e-4 of the largest, from rounded correlations —
    and a Gaussian with a negative variance has no draws to give.
    """
    eigvals, eigvecs = jnp.linalg.eigh(jnp.asarray(covmat))
    root = eigvecs * jnp.sqrt(jnp.clip(eigvals, 0.0, None))
    return jax.random.normal(key, (n_draws, len(eigvals))) @ root.T


@dataclass(eq=False)
class FitModel:
    """The model a fit was run with, over the datasets to draw at it.

    Built by ``smefitConfig.produce_fit_model`` from the fit's own runcard, so
    ``use_quad``, ``rge``, ``use_t0``, ``use_theory_covmat``, the coefficients
    and their constraints are the fit's. The datasets are those the report
    lists, or all of the fit's when it lists none.

    Attributes
    ----------
    data : smefit.core.DataGroup
    eft_model : EFTModel
        Its ``theory`` is the datasets' theory, in ``data`` order.
    datasets_chi2 : list of smefit.chi2.Chi2
        One per dataset, in ``data`` order, from the dataset's diagonal block
        of the fit's covariance.
    """

    data: object
    eft_model: EFTModel
    datasets_chi2: list
