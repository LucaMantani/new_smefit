"""
smefit.config.py

Config module of smefit
"""

import logging
import pathlib

from reportengine.configparser import Config

from smefit.chi2 import Chi2, build_chi2
from smefit.core import Coefficient, CoefficientGroup, DataGroup, TheoryGroup
from smefit.external_chi2 import load_external_chi2
from smefit.loader import load_dataset, load_theory
from smefit.model import EFTModel
from smefit.priors import Prior, _build_dist

log = logging.getLogger(__name__)


class smefitConfig(Config):
    """smefit Config class."""

    def parse_data_path(self, data_path):
        """Parse data path."""
        data_path = pathlib.Path(data_path)
        # Verify it exists
        if not data_path.exists():
            log.error(f"data_path {data_path} does not exist.")
            raise ValueError(f"data_path {data_path} does not exist.")
        log.info(f"Using data path: {data_path}")
        return data_path

    def parse_theory_path(self, theory_path):
        """Parse theory path."""
        theory_path = pathlib.Path(theory_path)
        # Verify it exists
        if not theory_path.exists():
            log.error(f"theory_path {theory_path} does not exist.")
            raise ValueError(f"theory_path {theory_path} does not exist.")
        log.info(f"Using theory path: {theory_path}")
        return theory_path

    def produce_data(self, datasets, data_path):
        """Produce data group object."""
        parsed_datasets = []
        for ds in datasets:
            dataset = load_dataset(data_path, ds["name"])
            parsed_datasets.append(dataset)

        return DataGroup(parsed_datasets)

    def produce_theory(self, datasets, theory_path):
        """Produce theory group object."""
        parsed_theories = []
        for ds in datasets:
            theory = load_theory(theory_path, ds["name"], ds["order"])
            parsed_theories.append(theory)

        return TheoryGroup(parsed_theories)

    def produce_fit_covmat(self, data, theory, use_t0=False, use_theory_covmat=False):
        """Produce the covariance matrix to be used in the fit."""
        if data.names != theory.names:
            raise ValueError(
                f"DataGroup and TheoryGroup contain different datasets.\n"
                f"  data:   {data.names}\n"
                f"  theory: {theory.names}"
            )
        if data.num_data != theory.n_data:
            raise ValueError(
                f"DataGroup and TheoryGroup have different total number of data points: "
                f"{data.num_data} vs {theory.n_data}"
            )
        if use_t0:
            log.info("Using t0 covariance matrix.")
            # build t0 covmat using theory predictions
            theory_predictions = theory.sm_pred
            covmat = data.t0_covmat(theory_predictions)
        else:
            log.info("Using experimental covariance matrix.")
            covmat = data.exp_covmat

        if use_theory_covmat:
            log.info("Adding theory covariance matrix to data covariance matrix.")
            covmat += theory.sm_covmat

        return covmat

    def parse_coefficients(self, coefficients):
        """Parse coefficients configuration."""
        coeffs = []
        for coeff_name, coeff_info in coefficients.items():
            coeffs.append(Coefficient(name=coeff_name, **coeff_info))
        group = CoefficientGroup(coeffs)
        # Validate that all vars in constrained coefficients refer to free coefficients
        free_names = set(group.free_names)
        for coeff in group.fixed_coeffs:
            if coeff.vars:
                unknown = [v for v in coeff.vars if v not in free_names]
                if unknown:
                    raise ValueError(
                        f"Coefficient '{coeff.name}': vars {unknown} are not free coefficients."
                    )
        return group

    def produce_eft_model(self, theory, coefficients, use_quad=False):
        """Produce EFT model mapping coefficients to theory predictions."""
        return EFTModel(theory, coefficients, use_quad)

    def parse_external_chi2(self, external_chi2):
        """Pass-through parser so reportengine can resolve external_chi2 as a node."""
        return external_chi2

    def produce_chi2(self, eft_model, data, fit_covmat, external_chi2=None):
        """Produce the chi2 function for the fit, optionally combining with external chi2s."""

        base_chi2 = build_chi2(eft_model, data, fit_covmat)

        if not external_chi2:
            return Chi2(base_chi2)

        ext_modules = load_external_chi2(
            external_chi2, eft_model.coefficients, rge_dict=None
        )

        def total_fn(coeffs):
            return base_chi2(coeffs) + sum(ext(coeffs) for ext in ext_modules)

        return Chi2(total_fn, has_external=True)

    def produce_prior(self, coefficients):
        """Produce joint prior over all free coefficients."""
        prior_specs = coefficients.prior_specs()
        dists = []
        for name in coefficients.free_names:
            spec = prior_specs[name]
            if spec is None:
                raise ValueError(f"Free coefficient '{name}' has no prior defined.")
            dists.append(_build_dist(spec))
        return Prior(dists, coefficients.free_names)
