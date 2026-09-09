"""External chi2 for superallowed beta decays.

At initialisation the derivative dL/dc_i of the LEC VnueduLL_1111 with respect to
each SMEFT coefficient is computed once analytically via rgevolve's run_and_match
matrix:

    L(c) ~= L_SM + (dL/dc) . c

making compute_chi2 a pure JAX function, fully JIT-compatible with both BlackJAX and
UltraNest.

rgevolve is used deliberately for the SMEFT -> WET matching instead of the
``wilson``-based machinery in ``smefit.rge``: the latter is far too slow to run a
fit with.

Example runcard entry::

    external_chi2:
      SA_beta_decays:
        path: new_smefit/external_chi2/low_energy/superallowed_beta_decay.py
"""

from __future__ import annotations

import importlib.resources  # must precede rgevolve imports — Python 3.14 workaround
import logging
from typing import TYPE_CHECKING, Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from rgevolve.tools.functions import get_wc_basis, run_and_match

from smefit.rge import RGE

if TYPE_CHECKING:
    from smefit.core import CoefficientGroup

log = logging.getLogger(__name__)

# Experimental Ft values (10^-3 s, 2010.13797), uncertainties, and Q-values (MeV,
# https://journals.aps.org/prc/pdf/10.1103/PhysRevC.91.025501) per nucleus
_NUCLEI: dict[str, dict[str, float]] = {
    "10C": {"mean": 3075.7, "std": 4.4, "Q": 1.908, "delta_r": 8.999999999999999e-05},
    "14O": {"mean": 3070.2, "std": 1.9, "Q": 2.831, "delta_r": 7.666666666666667e-05},
    "22Mg": {"mean": 3076.2, "std": 7.0, "Q": 4.125, "delta_r": 6.666666666666667e-05},
    "26Al": {"mean": 3072.4, "std": 1.1, "Q": 4.233, "delta_r": 6.666666666666667e-05},
    "26Si": {"mean": 3075.4, "std": 5.7, "Q": 4.841, "delta_r": 6.333333333333333e-05},
    "34Cl": {"mean": 3071.6, "std": 1.8, "Q": 5.492, "delta_r": 5.9999999999999995e-05},
    "34Ar": {"mean": 3075.1, "std": 3.1, "Q": 6.062, "delta_r": 5.666666666666667e-05},
    "38K": {"mean": 3072.9, "std": 2.0, "Q": 6.044, "delta_r": 5.666666666666667e-05},
    "38Ca": {"mean": 3077.8, "std": 6.2, "Q": 6.612, "delta_r": 5.666666666666667e-05},
    "42Sc": {"mean": 3071.7, "std": 2.0, "Q": 6.426, "delta_r": 5.666666666666667e-05},
    "46V": {"mean": 3074.3, "std": 2.0, "Q": 7.052, "delta_r": 5.333333333333333e-05},
    "50Mn": {"mean": 3071.1, "std": 1.6, "Q": 7.634, "delta_r": 5e-05},
    "54Co": {"mean": 3070.4, "std": 2.5, "Q": 8.244, "delta_r": 5e-05},
    "62Ga": {"mean": 3072.4, "std": 6.7, "Q": 9.181, "delta_r": 4.666666666666667e-05},
    "74Rb": {
        "mean": 3077.0,
        "std": 11.0,
        "Q": 10.417,
        "delta_r": 4.3333333333333334e-05,
    },
}

_EXP_MEAN = jnp.array([v["mean"] for v in _NUCLEI.values()])
_EXP_STD = jnp.array([v["std"] for v in _NUCLEI.values()])
_Q = jnp.array([v["Q"] for v in _NUCLEI.values()])
_DELTA_R = jnp.array([v["delta_r"] for v in _NUCLEI.values()])

_CONV = 1.519267e24
_PREF = 4 * jnp.pi**3 * jnp.log(2.0) / (2 * (0.5109989e-3) ** 5)  # GeV^-1
_GF = 1.16637859e-5  # GeV^-2

# Nuisance parameters of the beta-decay likelihood. They are ordinary runcard
# coefficients but carry no SMEFT operator, so they are excluded from the RGE
# translation (which would otherwise warn about them being unknown WCs).
_BD_PARAM_DEFAULTS: dict[str, float] = {
    "DRV": 0.02467,
    "eta1": 0.0,
    "eta2": 0.0,
    "eta3": 0.0,
    "Vud": 0.9737,
}


@jax.jit
def _chi2_smeft(
    DRV: jnp.ndarray,
    eta1: jnp.ndarray,
    eta2: jnp.ndarray,
    eta3: jnp.ndarray,
    Vud: jnp.ndarray,
    L: jnp.ndarray,
) -> jnp.ndarray:
    """Beta-decay chi2 for a given LEC shift ``L``.

    ``L = 0`` reproduces the SM expression exactly.
    """
    mean = _EXP_MEAN * _CONV
    std = _EXP_STD * _CONV
    Q = _Q
    Lf = -2.0 * jnp.sqrt(2.0) * _GF + L
    CV = -0.5 * Vud * Lf * jnp.sqrt(1.0 + DRV)
    Ft = _PREF / CV**2
    Ftt = Ft - mean * (eta1 * _DELTA_R + eta2 * 3.3e-4 + eta3 * 8.0e-5 * Q)
    return jnp.sum((Ftt - mean) ** 2 / std**2)


class SA_beta_decays:
    """SMEFiT external chi2 for superallowed beta decays — rgevolve Jacobian.

    The Jacobian dL/dc_i is computed analytically at initialisation via
    rgevolve.tools.functions.run_and_match.
    The chi2 is then a pure JAX function of the resolved coefficient vector.
    """

    def __init__(
        self,
        coefficients: CoefficientGroup,
        rge_dict: Mapping[str, Any] | None = None,
        starting_scale: float | None = None,
    ) -> None:
        self.coefficients = coefficients

        # Indices into the *full* coefficient vector returned by resolve().
        self._bd_idx: dict[str, int] = {
            name: coefficients.coeff_index[name]
            for name in _BD_PARAM_DEFAULTS
            if name in coefficients.coeff_index
        }

        self.num_data = len(_EXP_MEAN)

        # The smefit -> Warsaw translation is built through the standard smefit
        # runner, so the rge: block is read with the defaults the runcard
        # documents. Only the translation comes from here; the SMEFT -> WET
        # matching below is rgevolve's.
        settings = dict(rge_dict) if rge_dict is not None else {}
        if starting_scale is not None:
            settings["init_scale"] = float(starting_scale)

        smeft_names = [n for n in coefficients.names if n not in _BD_PARAM_DEFAULTS]
        runner = RGE.from_rge_dict(settings, smeft_names)
        self._scale = float(runner.init_scale)
        translation = runner.RGEbasis if smeft_names else {}

        self._dL = self._compute_jacobian_rgevolve(translation)

    def _compute_jacobian_rgevolve(
        self, translation: Mapping[str, Mapping[str, float]]
    ) -> jnp.ndarray:
        """Compute dL/dc analytically from the rgevolve run_and_match matrix.

        Collects all Warsaw WC names referenced by the smefit -> Warsaw
        translation, queries rgevolve for the row vector
        d(VnueduLL_1111)/d(warsaw_wc_j), then contracts it with the translation
        factors.

        Returns
        -------
        jnp.ndarray
            Jacobian of shape ``(len(coefficients.names),)``, aligned with the
            full coefficient vector returned by ``CoefficientGroup.resolve``.
            Entries for the beta-decay nuisance parameters are zero.
        """
        n_coeff = len(self.coefficients.names)

        warsaw_names_needed: set[str] = set()
        for wc_dict in translation.values():
            warsaw_names_needed.update(wc_dict.keys())

        if not warsaw_names_needed:
            return jnp.zeros(n_coeff)

        # Filter to names that exist as real WCs in rgevolve's Warsaw basis.
        smeft_warsaw_wcs = {wc[0] for wc in get_wc_basis("SMEFT", "Warsaw")}
        unknown = warsaw_names_needed - smeft_warsaw_wcs
        if unknown:
            log.warning(
                "SA_beta_decays: the following Warsaw WC names from the RGE "
                "translation are not present in rgevolve's SMEFT Warsaw basis "
                "and will be ignored: %s",
                sorted(unknown),
            )
        warsaw_list = sorted(warsaw_names_needed & smeft_warsaw_wcs)

        if not warsaw_list:
            return jnp.zeros(n_coeff)

        wcs_in = tuple((name, "R") for name in warsaw_list)
        wcs_out = (("VnueduLL_1111", "R"),)

        # run_and_match returns shape (len(wcs_out), len(wcs_in)) = (1, N).
        # M[0, j] = d(VnueduLL_1111_R) / d(warsaw_list[j]_R) at the given scales.
        M = run_and_match(
            "SMEFT",
            "WET",
            "Warsaw",
            "JMS",
            self._scale,
            2.0,
            wcs_in=wcs_in,
            wcs_out=wcs_out,
        )
        m_row = M[0]  # shape (N,)

        dL = np.zeros(n_coeff)
        for name, wc_dict in translation.items():
            i = self.coefficients.coeff_index[name]
            for j, wc in enumerate(warsaw_list):
                factor = wc_dict.get(wc, 0.0)
                if factor != 0.0:
                    dL[i] += m_row[j] * factor

        return jnp.array(dL)

    def compute_chi2(self, coefficient_values: jnp.ndarray) -> jnp.ndarray:
        # resolve() maps free (possibly whitened) values to the full coefficient
        # vector, applying fixed values and expression constraints exactly.
        resolved = self.coefficients.resolve(coefficient_values)

        def _get(name: str) -> jnp.ndarray:
            idx = self._bd_idx.get(name)
            if idx is None:
                return jnp.asarray(_BD_PARAM_DEFAULTS[name])
            return resolved[idx]

        L = jnp.dot(self._dL, resolved)

        return _chi2_smeft(
            _get("DRV"),
            _get("eta1"),
            _get("eta2"),
            _get("eta3"),
            _get("Vud"),
            L,
        )
