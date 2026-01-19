"""
smefit.core.py

Core module of smefit, containing the main data classes for the framework.
"""

from dataclasses import dataclass
from typing import List, Optional

import jax.numpy as jnp


@dataclass
class Dataset:
    """Class representing a dataset in smefit."""

    name: str
    num_data: int
    central_values: jnp.ndarray
    stat_err: jnp.ndarray
    syst_err: jnp.ndarray
    sys_names: List[str]
    sys_types: List[str]
    luminosity: jnp.ndarray


class DataGroup:
    """Class representing a group of datasets in smefit."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        # order datasets by name for consistency
        self.datasets.sort(key=lambda ds: ds.name)
        # concatenate central values
        self.cv = self._concatenate_central_values()
        # total number of data points
        self.num_data = sum(ds.num_data for ds in datasets)
        # concatenate luminosities
        self.lumi = self._concatenate_luminosities()
        # list of dataset names
        self.names = [ds.name for ds in datasets]
        # list of number of data points per dataset
        self.ndata_list = [ds.num_data for ds in datasets]

    def _concatenate_central_values(self) -> jnp.ndarray:
        """Concatenate central values from all datasets in the group."""
        return jnp.concatenate([ds.central_values for ds in self.datasets], axis=0)

    def _concatenate_luminosities(self) -> jnp.ndarray:
        """Concatenate luminosities from all datasets in the group."""
        return jnp.concatenate([ds.luminosity for ds in self.datasets], axis=0)
