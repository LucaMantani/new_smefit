"""The :class:`RGE` runner: pure computation of RGE matrices via ``wilson``.

Relies on the monkey patches applied by :mod:`smefit.rge._patches` (imported
before this module by :mod:`smefit.rge`), but otherwise does no file I/O.
"""

import logging
from contextlib import contextmanager
from functools import cached_property

import numpy as np
import pandas as pd
import wilson

from smefit.constants import gs, mz
from smefit.wcxf import inverse_wcxf_translate, wcxf_translate

# Numerical threshold for filtering small Wilson coefficient values
_SMALL_VALUE_THRESHOLD = 1e-14

# Allowed values for the corresponding `rge:` runcard keys. Kept here, next to
# the RGE runner that gives them meaning, and imported by smefitConfig.parse_rge
# so a typo is caught at config time.
ALLOWED_YUKAWA = frozenset({"top", "none", "full"})
ALLOWED_SMEFT_ACCURACY = frozenset({"integrate", "leadinglog"})

# The observable-basis operator names an RGE matrix can have rows for. Fixed by
# the WCxf translation table.
ALL_OPS = tuple(sorted(wcxf_translate.keys()))

_logger = logging.getLogger(__name__)

###########################
# Input parameter options #
###########################

# copying so we could use the default parameters later
default_params = wilson.run.smeft.smpar.p.copy()

top_yukawa = {
    "Vus": 0.0,
    "Vub": 0.0,
    "Vcb": 0.0,
    "gamma": 0.0,
    "m_b": 0.0,
    "m_s": 0.0,
    "m_c": 0.0,
    "m_u": 0.0,
    "m_d": 0.0,
    "m_e": 0.0,
    "m_mu": 0.0,
    "m_tau": 0.0,
}

no_yukawa = {**top_yukawa, "m_t": 0.0}

QCD_only = {
    "alpha_e": 0.0,
    "m_W": 1e-20,
    "m_h": 1e-20,
}


def evolve_gs(scale):
    # evolve gs from MZ to scale using 1-loop running with nf=6
    beta0 = 11 - 2 / 3 * 6
    return gs / np.sqrt(1 + 2 * beta0 * gs**2 / (4 * np.pi) ** 2 * np.log(scale / mz))


@contextmanager
def _wilson_params(yukawa, adm_QCD):
    """Temporarily set wilson SM parameters, restoring originals on exit."""
    saved = wilson.run.smeft.smpar.p.copy()
    try:
        if yukawa == "top":
            wilson.run.smeft.smpar.p.update(**top_yukawa)
        elif yukawa == "none":
            wilson.run.smeft.smpar.p.update(**no_yukawa)
        elif yukawa == "full":
            wilson.run.smeft.smpar.p.update(**default_params)

        if adm_QCD:
            wilson.run.smeft.smpar.p.update(**QCD_only)

        yield
    finally:
        wilson.run.smeft.smpar.p.clear()
        wilson.run.smeft.smpar.p.update(saved)


class RGE:
    """
    Class to compute the RGE matrix for the SMEFT Wilson coefficients.
    The RGE matrix is computed at the initial scale `init_scale` and
    evolved to the scale of interest.

    Parameters
    ----------
    wc_names: list
        list of Wilson coefficient names to be included in the RGE matrix
    init_scale: float
        initial scale of the Wilson coefficients
    accuracy: str
        accuracy of the RGE integration. Options: "leadinglog" or "integrate".
        Default is 'integrate'. Inherited behaviour from wilson package.
    adm_QCD: bool
        if True, only the QCD anomalous dimension is used. Default is False.
    yukawa: str
        Yukawa parameterization to be used. Options: "top", "none" or "full".
        Default is "top".
    """

    def __init__(
        self,
        wc_names,
        init_scale,
        accuracy="integrate",
        adm_QCD=False,
        yukawa="top",
    ):
        # order the Wilson coefficients alphabetically
        self.wc_names = sorted(wc_names)
        self.init_scale = init_scale
        self.accuracy = accuracy
        self.adm_QCD = adm_QCD
        self.yukawa = yukawa

        if yukawa not in ALLOWED_YUKAWA:
            raise ValueError(f"Yukawa parameter not supported: {yukawa}")

        _logger.info(f"Using Yukawa parameterization: {yukawa}.")
        _logger.info(
            f"Using anomalous dimension order: {'QCD' if adm_QCD else 'full'}."
        )
        _logger.info(
            f"Initializing RGE runner with initial scale {init_scale} GeV and accuracy {accuracy}."
        )

    @classmethod
    def from_rge_dict(cls, rge_dict, wc_names):
        """Build a runner from a raw or parsed ``rge:`` dict."""
        return cls(
            wc_names,
            init_scale=float(rge_dict.get("init_scale", 1e3)),
            accuracy=str(rge_dict.get("smeft_accuracy", "integrate")),
            adm_QCD=bool(rge_dict.get("adm_QCD", False)),
            yukawa=str(rge_dict.get("yukawa", "top")),
        )

    @property
    def settings(self):
        """The physics settings that determine the RGE matrices this runner produces.

        These four values, and only these four, decide whether a stored
        ``rge_matrix.pkl`` may be reused: ``load_precomputed_rge_matrix``
        compares this dict against the ``rge_settings`` entry of the pickle with
        strict equality. Adding a key here — or renaming one — invalidates every
        RGE matrix ever written, so don't.

        ``obs_scale`` and ``scale_variation`` are deliberately absent: they
        select *which* scales are requested, not how the running is done, and a
        cached matrix keyed by scale is reusable across runcards that ask for
        different scales.
        """
        return {
            "init_scale": self.init_scale,
            "smeft_accuracy": self.accuracy,
            "adm_QCD": self.adm_QCD,
            "yukawa": self.yukawa,
        }

    def RGEmatrix_dict(self, scale):
        """
        Compute the RGE solution at the scale `scale` and return it as a dictionary.
        """
        rge_matrix_dict = {}
        with _wilson_params(self.yukawa, self.adm_QCD):
            for wc_name, wc_vals in self.RGEbasis.items():
                # External couplings map to the null vector: evolving them is a no-op,
                # so skip the (expensive) wilson call and return no entries.
                if not wc_vals:
                    rge_matrix_dict[wc_name] = {}
                    continue

                _logger.info(f"Computing RGE for {wc_name} at {scale} GeV.")
                wc_init = wilson.Wilson(
                    wc_vals, scale=self.init_scale, eft="SMEFT", basis="Warsaw"
                )
                wc_init.set_option("smeft_accuracy", self.accuracy)

                wc_final = wc_init.match_run(scale=scale, eft="SMEFT", basis="Warsaw")

                # Remove small values
                wc_final_vals = {
                    key: value
                    for key, value in wc_final.dict.items()
                    if abs(value) > _SMALL_VALUE_THRESHOLD
                }

                # check that imaginary values are small
                if any(abs(val.imag) > 1e-10 for val in wc_final_vals.values()):
                    raise ValueError(
                        f"Imaginary values in Wilson coefficient for operator {wc_name}."
                    )

                rge_matrix_dict[wc_name] = self.map_to_smefit(wc_final_vals, scale)

        return rge_matrix_dict

    def RGEmatrix(self, scale):
        """
        Compute the RGE solution at the scale `scale` and return it as a pandas DataFrame.
        """
        # compute the RGE matrix dict at the scale `scale`
        rge_matrix_dict = self.RGEmatrix_dict(scale)

        # create the RGE matrix as pandas dataframe
        rge_matrix = pd.DataFrame(
            columns=self.wc_names, index=self.all_ops, dtype=float
        )

        for wc_name, wc_dict in rge_matrix_dict.items():
            for op in self.all_ops:
                rge_matrix.loc[op, wc_name] = wc_dict.get(op, 0.0)
        # if there are rows with all zeros, remove them
        rge_matrix = rge_matrix.loc[(rge_matrix != 0).any(axis=1)]

        return rge_matrix

    @cached_property
    def RGEbasis(self):
        """
        Returns the RGE basis translated from smefit to Warsaw.
        """
        # computes the translation from the smefit basis to the Warsaw basis
        # as expected by the Wilson package
        wc_basis = {}
        for wc_name in self.wc_names:
            try:
                wcxf_dict = wcxf_translate[wc_name]
            except KeyError:
                _logger.warning(
                    f"Wilson coefficient {wc_name} not present in the WCxf translation dictionary."
                )
                _logger.warning(
                    "Assuming it is a external coupling and associating it to the null vector."
                )
                wc_basis[wc_name] = {}
                continue

            wc_warsaw_name = wcxf_dict["wc"]
            if "value" not in wcxf_dict:
                wc_warsaw_value = [1] * len(wcxf_dict["wc"])
            else:
                # check if value is gs
                # (this is a special case for OtG)
                if wcxf_dict["value"] == ["-gs"]:
                    wc_warsaw_value = [-evolve_gs(self.init_scale)]
                else:
                    wc_warsaw_value = wcxf_dict["value"]

            # 1e-6 is because the Warsaw basis is in GeV^-2
            wc_value = {
                wc: val * 1e-6 for wc, val in zip(wc_warsaw_name, wc_warsaw_value)
            }
            wc_basis[wc_name] = wc_value

        return wc_basis

    def map_to_smefit(self, wc_final_vals, scale):
        """
        Map the Wilson coefficients from the Warsaw basis to the SMEFiT basis.
        """
        wc_dict = {}
        wc_final_keys = set(wc_final_vals.keys())
        for wc_basis, wc_inv_dict in inverse_wcxf_translate.items():
            wc_warsaw_name = wc_inv_dict["wc"]
            # Skip operators with no overlap with the evolved WCs
            if not wc_final_keys.intersection(wc_warsaw_name):
                continue

            if "coeff" not in wc_inv_dict:
                wc_warsaw_coeff = [1] * len(wc_warsaw_name)
            else:
                # check if coeff is 1/gs
                # (this is a special case for OtG)
                if wc_inv_dict["coeff"] == ["-1/gs"]:
                    wc_warsaw_coeff = [-1 / evolve_gs(scale)]
                else:
                    wc_warsaw_coeff = wc_inv_dict["coeff"]

            value = 0.0
            for wc, coeff in zip(wc_warsaw_name, wc_warsaw_coeff):
                if wc in wc_final_keys:
                    # 1e6 is to transform from GeV^-2 to TeV^2
                    value += 1e6 * wc_final_vals[wc].real * coeff
            wc_dict[wc_basis] = value
        return wc_dict

    @property
    def all_ops(self):
        return ALL_OPS

    def RGEevolve(self, wcs, scale):
        """
        Evolve the Wilson coefficients from the initial scale to the scale of interest.
        """
        wc_wilson = {}
        for op, values in self.RGEbasis.items():
            for key in values:
                if key not in wc_wilson:
                    wc_wilson[key] = values[key] * wcs[op]
                else:
                    wc_wilson[key] += values[key] * wcs[op]

        with _wilson_params(self.yukawa, self.adm_QCD):
            wc_init = wilson.Wilson(
                wc_wilson, scale=self.init_scale, eft="SMEFT", basis="Warsaw"
            )
            wc_init.set_option("smeft_accuracy", self.accuracy)
            wc_final = wc_init.match_run(scale=scale, eft="SMEFT", basis="Warsaw").dict

        # remove small values
        wc_final = {
            key: value
            for key, value in wc_final.items()
            if abs(value) > _SMALL_VALUE_THRESHOLD
        }

        return self.map_to_smefit(wc_final, scale)

    def clone_runner(self, coeff_list):
        """
        Clone the RGE runner with a different coefficient list.

        Parameters
        ----------
        coeff_list: list
            list of Wilson coefficient names to be included in the RGE matrix

        Returns
        -------
        RGE
            cloned RGE runner
        """
        return RGE(
            coeff_list,
            self.init_scale,
            self.accuracy,
            self.adm_QCD,
            self.yukawa,
        )
