"""
smefit.config.py

Config module of smefit
"""

import logging
import pathlib

from reportengine.configparser import Config

from smefit.core import DataGroup, TheoryGroup
from smefit.loader import load_dataset, load_theory

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
