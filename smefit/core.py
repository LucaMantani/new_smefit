"""
smefit.core.py

Core module of smefit, containing the main data classes for the framework.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import List

import jax.numpy as jnp
import pandas as pd

from smefit.data_utils import covmat_from_systematics


@dataclass
class Dataset:
    """Class representing a dataset in smefit.

    Attributes
    ----------
    name : str
        Name of the dataset.
    num_data : int
        Number of data points in the dataset.
    central_values : jnp.ndarray 1D array
        Central values of the data points.
    stat_err : jnp.ndarray 1D array
        Statistical uncertainties of the data points.
    syst_err : jnp.ndarray 2D array
        Systematic uncertainties of the data points, shape (n_sys, num_data).
    sys_names : List[str]
        Names of the systematic uncertainties.
    sys_types : List[str]
        Types of the systematic uncertainties.
    luminosity : jnp.ndarray 1D array
        Luminosity values associated with the data points.
    """

    name: str
    num_data: int
    central_values: jnp.ndarray
    stat_err: jnp.ndarray
    syst_err: jnp.ndarray
    sys_names: List[str]
    sys_types: List[str]
    luminosity: jnp.ndarray

    # compute syst_err as percentage of central values
    @cached_property
    def syst_err_mult(self) -> jnp.ndarray:
        """Systematic uncertainties as percentage of central values."""
        return self.syst_err / self.central_values[None, :]


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
        # build full exp covariance matrix
        self.exp_covmat = self._build_exp_covmat()

    def _concatenate_central_values(self) -> jnp.ndarray:
        """Concatenate central values from all datasets in the group."""
        return jnp.concatenate([ds.central_values for ds in self.datasets], axis=0)

    def _concatenate_luminosities(self) -> jnp.ndarray:
        """Concatenate luminosities from all datasets in the group."""
        return jnp.concatenate([ds.luminosity for ds in self.datasets], axis=0)

    def _build_exp_covmat(self) -> jnp.ndarray:
        """Build experimental covariance matrix from all datasets in the group.

        This combines statistical and systematic uncertainties from all datasets,
        accounting for correlations both within and across datasets.

        Returns
        -------
        covmat : jnp.ndarray
            Full covariance matrix of shape (num_data, num_data)
        """
        stat_errors = [ds.stat_err for ds in self.datasets]

        # Convert syst_err jnp arrays to pd.DataFrames with sys_names as columns
        # ds.syst_err has shape (n_sys, n_data), we need to transpose it
        sys_errors = [
            pd.DataFrame(ds.syst_err.T, columns=ds.sys_names) for ds in self.datasets
        ]
        return jnp.array(covmat_from_systematics(stat_errors, sys_errors))

    def t0_covmat(self, theory_predictions) -> jnp.ndarray:
        """Build t0 covariance matrix using theory predictions.
        This method constructs the t0 covariance matrix by replacing multiplicative
        systematic uncertainties with values derived from the provided theory predictions.

        Parameters
        ----------
        theory_predictions : jnp.ndarray
            Array of theory predictions corresponding to the concatenated data points.
        Returns
        -------
        covmat : jnp.ndarray
            Full t0 covariance matrix of shape (num_data, num_data)
        """
        stat_errors = [ds.stat_err for ds in self.datasets]
        sys_errors = []

        offset = 0
        for ds in self.datasets:
            n = ds.num_data
            # we take the theory predictions for this dataset
            t_ds = theory_predictions[offset : offset + n]
            offset += n

            sys_types = [t.upper() for t in ds.sys_types]
            is_add = jnp.array([t == "ADD" for t in sys_types])  # shape (n_sys,)

            mult_abs = ds.syst_err_mult * t_ds[None, :]  # shape (n_sys, n)

            syst_errs = jnp.where(is_add[:, None], ds.syst_err, mult_abs)

            sys_errors.append(
                pd.DataFrame(jnp.asarray(syst_errs).T, columns=ds.sys_names)
            )

        return jnp.array(covmat_from_systematics(stat_errors, sys_errors))
