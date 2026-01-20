"""
smefit.config.py

Config module of smefit
"""

import logging
import pathlib

from reportengine.configparser import Config

from smefit.core import Coefficient, CoefficientGroup, DataGroup, TheoryGroup
from smefit.loader import load_dataset, load_theory
from smefit.model import EFTModel

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
        return CoefficientGroup(coeffs)

    def produce_eft_model(self, theory, coefficients, use_quad=False):
        """Produce EFT model mapping coefficients to theory predictions."""
        return EFTModel(theory, coefficients, use_quad)
