"""
smefit.config.py

Config module of smefit
"""

import logging
import pathlib

from reportengine.configparser import Config

from smefit.core import DataGroup
from smefit.loader import load_dataset

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

    def produce_data(self, datasets, data_path):
        """Produce list of data."""
        parsed_datasets = []
        for ds in datasets:
            dataset = load_dataset(data_path, ds["name"])
            parsed_datasets.append(dataset)

        return DataGroup(parsed_datasets)
