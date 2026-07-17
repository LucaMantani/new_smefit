import logging
import pathlib

import jax.numpy as jnp

from smefit import loader
from smefit.core import DataGroup, TheoryGroup
from smefit.model import EFTModel
from smefit.rge import load_rge_matrix

log = logging.getLogger(__name__)

_HERE = pathlib.Path(__file__).parent


class HLLHC_DYMee_13TeV:

    def __init__(
        self,
        coefficients,
        rge_dict=None,
        use_quad=True,
        order="LO",
        use_theory_covmat=False,
        use_t0=False,
        save_rge_path=None,
        rg_matrix=None,
    ):
        """
        Initialize the HLLHC_DYMee_13TeV class.

        Parameters
        ----------
        coefficients : CoefficientGroup
            The Wilson coefficients to be used in the analysis.
        rge_dict : dict, optional
            A dictionary containing the RGE information.
        use_quad : bool
            Whether to include quadratic EFT contributions.
        order : str
            Perturbative order of the EFT theory prediction (e.g., "LO", "NLO_QCD").
        use_theory_covmat : bool
            Whether to add the theory covariance matrix to the fit covariance.
        use_t0 : bool
            Whether to use the t0 prescription for multiplicative systematics.
        save_rge_path : str or pathlib.Path, optional
            If provided, the path where to save the computed RGE matrix for future reuse.
        rg_matrix : A pre-computed RGE matrix.
        """
        theory_path = _HERE / "theory"
        data_path = _HERE / "commondata_projections_L0"

        coeff_list = sorted(coefficients.names)

        dataset = loader.load_dataset(data_path, "HLLHC_DYMee_13TeV")
        theory = loader.load_theory(theory_path, "HLLHC_DYMee_13TeV", order)

        data = DataGroup([dataset])
        theory_group = TheoryGroup([theory])

        rge_matrix = None
        if rge_dict is not None:
            # If a pre-computed rge matrix is provided, add it to the rge_dict
            # If not, set it to False in case it was defined for the datasets
            if rg_matrix is not None:
                rge_dict["rg_matrix"] = rg_matrix
            else:
                rge_dict["rg_matrix"] = False

            rge_matrix = load_rge_matrix(
                rge_dict=rge_dict,
                coeff_list=coeff_list,
                theory_group=theory_group,
                save_path=save_rge_path,
            )

        self.model = EFTModel(
            theory_group, coefficients, use_quad, rge_matrix=rge_matrix
        )

        self.data_cv = data.cv
        self.num_data = data.num_data

        if use_t0:
            fit_covmat = data.t0_covmat(theory_group.sm_pred)
        else:
            fit_covmat = data.exp_covmat

        if use_theory_covmat:
            fit_covmat = fit_covmat + theory_group.sm_covmat

        self.inv_covmat = jnp.linalg.inv(fit_covmat)

    def compute_chi2(self, coeffs):
        theory = self.model.forward_map(coeffs)
        safe_theory = jnp.clip(theory, a_min=1e-6)

        data = self.data_cv
        # Standard JAX pattern: safe_data=1.0 on masked bins keeps log(1/t) finite
        # on the unselected branch, preventing nan gradients from 0*log(0/t)
        safe_data = jnp.where(data > 0, data, 1.0)

        log_term = jnp.where(
            data > 0, safe_data * jnp.log(safe_data / safe_theory), 0.0
        )

        return 2.0 * jnp.sum(safe_theory - data + log_term)
