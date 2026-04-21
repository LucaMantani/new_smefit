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

    def rescale_sys(self, sys, fred_sys):
        """Reduce systematic uncertainties by fred_sys.

        For artificial (square, negative-entry) covariance decompositions,
        the covariance is reconstructed and only its diagonal is kept before scaling.
        """
        is_square = sys.shape[0] == sys.shape[1]
        is_artificial = is_square and np.any(sys.values < 0)

        if is_artificial:
            cov_tot = sys.values @ sys.values.T
            sys_diag = np.sqrt(np.diagonal(cov_tot))
            sys_rescaled = np.diag(sys_diag * fred_sys)
            return pd.DataFrame(sys_rescaled, index=sys.index, columns=sys.columns)

        return sys * fred_sys

    @staticmethod
    def rescale_stat(stat, lumi_old, lumi_new):
        """Scale statistical uncertainties from lumi_old to lumi_new."""
        fred_stat = np.sqrt(lumi_old / lumi_new)
        return stat * fred_stat

    def build_data_group(self):
        """Build a DataGroup with projected central values and uncertainties.

        Returns
        -------
        DataGroup
            Projected datasets, ready to be used as the pseudodata node.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        cv = self.compute_cv_projection()
        projected_datasets = []
        offset = 0

        for ds in self.data.datasets:
            n = ds.num_data
            cv_theory = cv[offset : offset + n]
            central_values = np.asarray(ds.central_values)

            # ratio of theory to experimental central value; clamp edge cases to 1
            ratio_sm_exp = cv_theory / central_values
            ratio_sm_exp = np.where(ratio_sm_exp < 0, 1.0, ratio_sm_exp)
            ratio_sm_exp = np.where(
                np.logical_or(np.isnan(ratio_sm_exp), np.isinf(ratio_sm_exp)),
                1.0,
                ratio_sm_exp,
            )

            # scale stat uncertainty to theory CV
            stat = np.asarray(ds.stat_err) * np.sqrt(ratio_sm_exp)

            # t0-project systematics: ADD rows kept absolute,
            # MULT rows converted to absolute using the theory CV
            n_sys = len(ds.sys_names)
            sys_add = np.asarray(ds.syst_err)  # (n_sys, n_data)

            if n_sys > 0:
                type_arr = np.array([t.upper() for t in ds.sys_types])
                mult_idx = np.flatnonzero(type_arr == "MULT")
                add_idx = np.flatnonzero(type_arr != "MULT")

                sys_t0 = np.zeros((n_sys, n))
                if len(add_idx) > 0:
                    sys_t0[add_idx] = sys_add[add_idx]
                if len(mult_idx) > 0:
                    # syst_err_mult = syst_err / cv_exp; multiply by cv_theory → absolute
                    sys_t0[mult_idx] = (
                        np.asarray(ds.syst_err_mult)[mult_idx] * cv_theory
                    )

                name_sys = list(ds.sys_names)
            else:
                # no systematics: inject a single zero UNCORR column
                name_sys = ["UNCORR"]
                sys_t0 = np.zeros((1, n))

            # DataFrame shape: (n_data, n_sys) — expected by covmat_from_systematics
            sys_df = pd.DataFrame(data=sys_t0.T, columns=name_sys)

            if self.lumi_new is not None:
                no_stats = not np.any(stat)
                if no_stats:
                    sys_red = self.rescale_sys(sys_df, self.fred_tot)
                    stat_red = stat
                else:
                    lumi_old = np.asarray(ds.luminosity)
                    if np.isnan(lumi_old).any():
                        raise ValueError(
                            f"NaN luminosity for dataset {ds.name!r}. "
                            "Specify a luminosity value in the dataset file."
                        )
                    stat_red = self.rescale_stat(stat, lumi_old, self.lumi_new)
                    sys_red = self.rescale_sys(sys_df, self.fred_sys)
                new_lumi = jnp.full(n, float(self.lumi_new))
            else:
                stat_red = stat
                sys_red = sys_df
                new_lumi = ds.luminosity

            # covariance used for L1 noise sampling
            newcov = covmat_from_systematics([stat_red], [sys_red])
            if self.use_theory_covmat:
                newcov += np.asarray(
                    self.theory.sm_covmat[offset : offset + n, offset : offset + n]
                )

            cv_proj = cv_theory.copy()
            if self.noise == "L1":
                cv_proj = np.random.multivariate_normal(cv_theory, newcov)

            new_name = f"{ds.name}_proj" if self.lumi_new is not None else ds.name
            # all systematics are now absolute
            new_sys_types = ["ADD"] * len(name_sys)

            log.info("Building pseudodata for %s", ds.name)
            projected_datasets.append(
                Dataset(
                    name=new_name,
                    num_data=n,
                    central_values=jnp.asarray(cv_proj),
                    stat_err=jnp.asarray(stat_red),
                    syst_err=jnp.asarray(sys_red.T.values),  # (n_sys, n_data)
                    sys_names=name_sys,
                    sys_types=new_sys_types,
                    luminosity=new_lumi,
                )
            )
            offset += n

        return DataGroup(projected_datasets)
