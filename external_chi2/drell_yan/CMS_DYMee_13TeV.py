import pathlib

import jax.numpy as jnp

from smefit import loader
from smefit.core import DataGroup, TheoryGroup
from smefit.model import EFTModel

_HERE = pathlib.Path(__file__).parent


class CMS_DYMee_13TeV:
    def __init__(
        self,
        coefficients,
        rge_dict=None,
        use_quad=True,
        order="LO",
        use_theory_covmat=False,
        use_t0=False,
        use_projection_L0=False,
    ):
        """
        Initialize the CMS_DYMee_13TeV class.

        Parameters
        ----------
        coefficients : CoefficientGroup
            The Wilson coefficients to be used in the analysis.
        rge_dict : dict, optional
            Reserved for future RGE support; currently unused.
        use_quad : bool
            Whether to include quadratic EFT contributions.
        order : str
            Perturbative order of the EFT theory prediction (e.g., "LO", "NLO_QCD").
        use_theory_covmat : bool
            Whether to add the theory covariance matrix to the fit covariance.
        use_t0 : bool
            Whether to use the t0 prescription for multiplicative systematics.
        use_projection_L0 : bool
            Whether to use the L0 projection dataset instead of the default one.
        """
        theory_path = _HERE / "theory"
        if use_projection_L0:
            data_path = _HERE / "commondata_projections_L0"
        else:
            data_path = _HERE / "commondata"

        dataset = loader.load_dataset(data_path, "CMS_DYMee_13TeV")
        theory = loader.load_theory(theory_path, "CMS_DYMee_13TeV", order)

        data = DataGroup([dataset])
        theory_group = TheoryGroup([theory])

        self.model = EFTModel(theory_group, coefficients, use_quad)
        self.data_cv = data.cv
        self.num_data = data.num_data

        if use_t0:
            fit_covmat = data.t0_covmat(theory_group.sm_pred)
        else:
            fit_covmat = data.exp_covmat

        if use_theory_covmat:
            fit_covmat = fit_covmat + theory_group.sm_covmat

        inv_covmat = jnp.linalg.inv(fit_covmat)
        self.inv_covmat = inv_covmat

    def compute_chi2(self, coeffs):
        # Compute theory predictions
        theory = self.model.forward_map(coeffs)

        theory = jnp.where(theory > 0, theory, 1e-6)
        data = self.data_cv

        # Poisson log-likelihood chi2; if data == 0, the x*log(x) term vanishes
        log_term = jnp.where(data > 1e-6, data * jnp.log(data / theory), 0.0)

        return 2.0 * jnp.sum(theory - data + log_term)
