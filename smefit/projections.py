"""
smefit.projections.py

Module for generating pseudodata projections.
"""

import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd

from smefit.core import DataGroup, Dataset
from smefit.data_utils import covmat_from_systematics

log = logging.getLogger(__name__)


class Projection:
    """Generates pseudodata by projecting central values and uncertainties."""

    def __init__(
        self, data, theory, pseudodata_settings, use_theory_covmat=False, eft_model=None
    ):
        self.data = data
        self.theory = theory
        self.lumi_new = pseudodata_settings["lumi_new"]
        self.noise = pseudodata_settings["noise"]
        self.seed = pseudodata_settings["seed"]
        self.fred_tot = pseudodata_settings["fred_tot"]
        self.fred_sys = pseudodata_settings["fred_sys"]
        self.use_theory_covmat = use_theory_covmat
        self.eft_model = eft_model

    def compute_cv_projection(self):
        """Return theory predictions (SM or SM+EFT) as central values.

        When eft_model is set, evaluates at the fixed coefficient values encoded
        in the model's CoefficientGroup (all fixed → empty free-params array).
        """
        if self.eft_model is None:
            return np.asarray(self.theory.sm_pred)
        return np.asarray(self.eft_model.forward_map(jnp.array([])))

    @staticmethod
    def rescale_sys(sys, fred):
        """Reduce systematic uncertainties by fred.

        For artificial (square, negative-entry) covariance decompositions,
        the covariance is reconstructed and only its diagonal is kept before scaling.
        """
        is_square = sys.shape[0] == sys.shape[1]
        if is_square and np.any(sys.values < 0):
            cov_diag = np.sqrt(np.diagonal(sys.values @ sys.values.T))
            return pd.DataFrame(
                np.diag(cov_diag * fred), index=sys.index, columns=sys.columns
            )
        return sys * fred

    @staticmethod
    def rescale_stat(stat, lumi_old, lumi_new):
        """Scale statistical uncertainties from lumi_old to lumi_new."""
        return stat * np.sqrt(lumi_old / lumi_new)

    def _apply_t0(self, ds, cv_theory):
        """Convert MULT systematics to absolute using the theory CV.

        Returns the (n_sys, n_data) array after t0 and the list of sys names.
        If the dataset has no systematics, injects a single zero UNCORR column.
        """
        if len(ds.sys_names) == 0:
            return np.zeros((1, ds.num_data)), ["UNCORR"]

        sys_add = np.asarray(ds.syst_err)  # (n_sys, n_data)
        is_mult = np.array([t.upper() == "MULT" for t in ds.sys_types])  # (n_sys,)
        # MULT rows: syst_err_mult * cv_theory (absolute); ADD rows: unchanged
        mult_scale = cv_theory / np.asarray(ds.central_values)  # (n_data,)
        scale = np.where(is_mult[:, None], mult_scale[None, :], 1.0)
        return sys_add * scale, list(ds.sys_names)

    def _rescale_uncertainties(self, ds, stat, sys_df):
        """Return (stat_red, sys_red, new_lumi) after applying lumi and fred factors."""
        if self.lumi_new is None:
            return stat, sys_df, ds.luminosity

        lumi_old = np.asarray(ds.luminosity)
        if np.isnan(lumi_old).any():
            raise ValueError(
                f"NaN luminosity for dataset {ds.name!r}. "
                "Specify a luminosity value in the dataset file."
            )
        new_lumi = jnp.full(ds.num_data, float(self.lumi_new))

        if not np.any(stat):
            return stat, self.rescale_sys(sys_df, self.fred_tot), new_lumi

        return (
            self.rescale_stat(stat, lumi_old, self.lumi_new),
            self.rescale_sys(sys_df, self.fred_sys),
            new_lumi,
        )

    def _project_dataset(self, ds, cv_theory, sm_covmat_block):
        """Build a single projected Dataset from one input dataset."""
        n = ds.num_data
        central_values = np.asarray(ds.central_values)

        # ratio of theory to experiment CV; clamp negative/invalid entries to 1
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = cv_theory / central_values
        ratio = np.where((ratio > 0) & np.isfinite(ratio), ratio, 1.0)

        stat = np.asarray(ds.stat_err) * np.sqrt(ratio)
        sys_t0, name_sys = self._apply_t0(ds, cv_theory)
        sys_df = pd.DataFrame(sys_t0.T, columns=name_sys)

        stat_red, sys_red, new_lumi = self._rescale_uncertainties(ds, stat, sys_df)

        newcov = covmat_from_systematics([stat_red], [sys_red])
        if sm_covmat_block is not None:
            newcov += sm_covmat_block

        cv_proj = cv_theory.copy()
        if self.noise == "L1":
            cv_proj = np.random.multivariate_normal(cv_theory, newcov)

        log.info("Building pseudodata for %s", ds.name)
        return Dataset(
            name=f"{ds.name}_proj" if self.lumi_new is not None else ds.name,
            num_data=n,
            central_values=jnp.asarray(cv_proj),
            stat_err=jnp.asarray(stat_red),
            syst_err=jnp.asarray(sys_red.T.values),  # (n_sys, n_data)
            sys_names=name_sys,
            sys_types=["ADD"] * len(name_sys),
            luminosity=new_lumi,
        )

    def build_data_group(self):
        """Build a DataGroup with projected central values and uncertainties."""
        if self.seed is not None:
            np.random.seed(self.seed)

        cv = self.compute_cv_projection()
        projected_datasets = []
        offset = 0

        for ds in self.data.datasets:
            n = ds.num_data
            sm_covmat_block = (
                np.asarray(
                    self.theory.sm_covmat[offset : offset + n, offset : offset + n]
                )
                if self.use_theory_covmat
                else None
            )
            projected_datasets.append(
                self._project_dataset(ds, cv[offset : offset + n], sm_covmat_block)
            )
            offset += n

        return DataGroup(projected_datasets)
