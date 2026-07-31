import logging
import pathlib
import pickle
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property, partial, wraps

import ckmutil.ckm
import jax.numpy as jnp
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
### Patch of a CKM function, so that the CP violating
### phase is set to gamma and not computed explicitly
### See https://github.com/wilson-eft/wilson/issues/113#issuecomment-2179273979
### This needs to be done before the import of wilson

# copying so we keep the original function
ckm_tree = deepcopy(ckmutil.ckm.ckm_tree)
ckmutil.ckm.ckm_tree = partial(ckm_tree, delta_expansion_order=0)
### End of patch

##################### MONKEY PATCH
# switch off the SM - EFT mixing, since SMEFiT assumes that the
# RGE solution is linearised
# Keep a reference to the original beta function
original_beta = wilson.run.smeft.beta.beta


def beta_wrapper(C, HIGHSCALE=np.inf, *args, **kwargs):
    return original_beta(C, HIGHSCALE, *args, **kwargs)


wilson.run.smeft.beta.beta = beta_wrapper
wilson.run.smeft.beta.beta_array = partial(
    wilson.run.smeft.beta.beta_array, HIGHSCALE=np.inf
)
##################### END OF MONKEY PATCH

##################### MONKEY PATCH
# Patch smeftpar: we remove all dependence on SMEFT parameters
# in SM paramaters, otherwise the linear approximation is not valid
C_patch = {
    "phi": 0.0,
    "phiBox": 0.0,
    "phiD": 0.0,
    "phiWB": 0.0,
    "phiG": 0.0,
    "phiW": 0.0,
    "phiB": 0.0,
    "dphi": 0.0,
    "uphi": 0.0,
    "ephi": 0.0,
}
# Reference to the original smeftpar function
original_smeftpar = wilson.run.smeft.smpar.smeftpar


# Define the monkey-patched function
@wraps(original_smeftpar)
def patched_smeftpar(*args, **kwargs):
    # check if C is passed as a keyword argument
    if "C" in kwargs:
        kwargs["C"] = C_patch
    # otherwise, check if it is passed as a positional argument
    else:
        args = list(args)
        args[1] = C_patch

    return original_smeftpar(*args, **kwargs)


# Apply the monkey patch
wilson.run.smeft.smpar.smeftpar = patched_smeftpar
##################### END OF MONKEY PATCH


##################### MONKEY PATCH
# Monkey patch flavour rotation
# Define the new method
def _to_wcxf_no_rotation(self, C_out, scale_out):
    """Return the Wilson coefficients `C_out` as a wcxf.WC instance, without rotation."""
    # Skip the self._rotate_defaultbasis line
    d = wilson.util.smeftutil.arrays2wcxf_nonred(C_out)
    d = wilson.wcxf.WC.dict2values(d)
    wc = wilson.wcxf.WC("SMEFT", "Warsaw", scale_out, d)
    return wc


# Monkey-patch the method
wilson.run.smeft.classes.SMEFT._to_wcxf = _to_wcxf_no_rotation
##################### END OF MONKEY PATCH

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

no_yukawa = {
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
    "m_t": 0.0,
}

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

    def RGEmatrix_dict(self, scale):
        """
        Compute the RGE solution at the scale `scale` and return it as a dictionary.
        """
        rge_matrix_dict = {}
        with _wilson_params(self.yukawa, self.adm_QCD):
            for wc_name, wc_vals in self.RGEbasis.items():
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
                    "Assuming it is a UV coupling and associating it to the null vector."
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
        return sorted(wcxf_translate.keys())

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


@dataclass(frozen=True)
class RGESettings:
    """The physics settings that determine an RGE matrix.

    These four values, and only these four, decide whether a stored
    ``rge_matrix.pkl`` may be reused: :func:`load_precomputed_rge_matrix`
    compares :meth:`to_dict` against the ``rge_settings`` entry of the pickle
    with strict equality. Adding a field here — or changing a key name in
    :meth:`to_dict` — invalidates every RGE matrix ever written, so don't.

    ``obs_scale`` and ``scale_variation`` are deliberately absent: they select
    *which* scales are requested, not how the running is done, and a cached
    matrix keyed by scale is reusable across runcards that ask for different
    scales.
    """

    init_scale: float
    smeft_accuracy: str = "integrate"
    adm_QCD: bool = False
    yukawa: str = "top"

    @classmethod
    def from_dict(cls, rge_dict):
        """Build from a raw or parsed ``rge:`` dict.

        Tolerant of missing keys and of YAML wrapper types, so hand-built dicts
        (external chi2 modules, tests) work as well as the normalised dict
        returned by ``smefitConfig.parse_rge``. The casts also keep the pickled
        settings plain-Python and therefore comparable.
        """
        return cls(
            init_scale=float(rge_dict.get("init_scale", 1e3)),
            smeft_accuracy=str(rge_dict.get("smeft_accuracy", "integrate")),
            adm_QCD=bool(rge_dict.get("adm_QCD", False)),
            yukawa=str(rge_dict.get("yukawa", "top")),
        )

    def to_dict(self):
        """Plain-Python dict used as the on-disk compatibility key."""
        return {
            "init_scale": self.init_scale,
            "smeft_accuracy": self.smeft_accuracy,
            "adm_QCD": self.adm_QCD,
            "yukawa": self.yukawa,
        }

    def runner(self, coeff_list):
        """Return an :class:`RGE` runner configured with these settings."""
        return RGE(
            coeff_list,
            self.init_scale,
            self.smeft_accuracy,
            self.adm_QCD,
            self.yukawa,
        )


@dataclass
class RGEMatrix:
    """Container for the stacked RGE matrices and associated metadata.

    Attributes
    ----------
    stacked_mats : jnp.ndarray
        Shape ``(n_data, n_obs_ops, n_init_coeffs)`` — one matrix per data
        point (or ``(1, …)`` when a single fixed observable scale is used and
        the matrix is broadcast downstream by ``EFTModel._apply_rge``).
    obs_operators : list of str
        Observable-basis operator names (alphabetically sorted).
    init_operators : list of str
        Initial Wilson coefficient names (alphabetically sorted).
    scales : list of float
        One scale per data point for dynamic mode, or a single-element list for
        a fixed observable scale.
    settings : RGESettings
        The running configuration these matrices were computed with; stored in
        the pickle so a later run can check reusability.

    Notes
    -----
    Serialisation is deliberately one-way. :meth:`write` stores one frame per
    *unique* scale, so the per-data-point stacking of ``stacked_mats`` cannot be
    reconstructed from the file and there is no ``from_file``. Reading a stored
    matrix back is the job of :func:`load_precomputed_rge_matrix` followed by
    :func:`load_rge_mats_from_scales`, which need the new run's coefficient list
    and scales anyway.
    """

    stacked_mats: jnp.ndarray
    obs_operators: list
    init_operators: list
    scales: list
    settings: RGESettings

    def to_dump_dict(self):
        """Build the on-disk payload: ``{'rge_settings': {...}, <scale>: DataFrame}``.

        Duplicate scales collapse to a single entry, so a dynamic-scale fit over
        many data points sharing a scale stores one frame per unique scale.
        """
        to_dump = {"rge_settings": self.settings.to_dict()}
        for scale, matrix in zip(self.scales, self.stacked_mats):
            to_dump[scale] = pd.DataFrame(
                np.asarray(matrix, dtype=float),
                index=self.obs_operators,
                columns=self.init_operators,
            )
        return to_dump

    def write(self, output_path, name="rge_matrix"):
        """Pickle this matrix to ``<output_path>/<name>.pkl``.

        The file can be fed back to a later runcard through ``rge.rg_matrix``.
        """
        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{name}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(self.to_dump_dict(), f)
        _logger.info("RGE matrix written to %s.", out_file)


def load_precomputed_rge_matrix(path_to_rge_mat, rge_settings):
    """
    Load a precomputed RGE matrix pickle and validate its settings.

    Parameters
    ----------
    path_to_rge_mat : str or pathlib.Path
        Path to the pickle file containing the precomputed RGE matrices.
    rge_settings : dict
        Expected RGE settings to validate against the stored file.

    Returns
    -------
    dict
        A dictionary containing cached RGE matrices keyed by scale. The
        returned dictionary excludes the `'rge_settings'` entry.

    Raises
    ------
    ValueError
        If the settings in the precomputed file do not match `rge_settings`.
    """
    with open(path_to_rge_mat, "rb") as f:
        rgemats_precomp = pickle.load(f)
    if rge_settings != rgemats_precomp["rge_settings"]:
        raise ValueError("RGE settings do not match RGE matrix precomputed settings.")

    rge_cache = {k: v for k, v in rgemats_precomp.items() if k != "rge_settings"}
    _logger.info(f"Loaded precomputed RGE matrix from {path_to_rge_mat}.")
    return rge_cache


def _find_cached_scale(cache: dict, scale: float, rtol: float = 1e-5) -> float | None:
    """Return the matching key in cache if one exists within relative tolerance, else None."""
    for key in cache:
        if abs(key - scale) <= rtol * abs(key):
            return key
    return None


def load_rge_mats_from_scales(scales, coeff_list, rge_runner, rge_cache):
    """
    Load or compute RGE matrices for the given list of scales.

    For each scale in *scales* this function:
    - Returns a pandas DataFrame containing the RGE matrix for the requested
      coefficients in *coeff_list*.
    - Uses matrices from *rge_cache* when available.
    - If a cached matrix is missing some requested coefficients, computes only
      the missing coefficients with a cloned runner and merges them into the
      cached matrix. The cache is updated in-place.
    - If no cached matrix exists for a scale, computes the full matrix and
      stores it in the cache.

    Parameters
    ----------
    scales : iterable of float
        Energy scales at which to fetch/compute RGE matrices.
    coeff_list : list of str
        Requested Wilson coefficient names (columns) to return for each matrix.
    rge_runner : RGE
        An instance of the RGE runner
    rge_cache : dict
        Mapping scale -> pandas.DataFrame for previously computed RGE matrices.
        This dict will be updated in-place when new or extended RGE matrices are
        computed.

    Returns
    -------
    list of pandas.DataFrame
        List of RGE matrices (one per input scale)
    """
    # Deduplicate: compute only for unique scales, then map back
    unique_scales = sorted(set(scales))
    _logger.info(
        f"Computing RGE matrices for {len(unique_scales)} unique scales "
        f"(out of {len(scales)} total data points)."
    )

    unique_rgemats = {}
    for scale in unique_scales:
        # Check if the RGE matrix has already been computed
        cached_key = _find_cached_scale(rge_cache, scale)
        if cached_key is not None:
            rgemat_scale = rge_cache[cached_key]
            coeff_list_cache = rgemat_scale.columns.tolist()
            # Compute difference in coeff_list
            missing_coeffs = list(set(coeff_list) - set(coeff_list_cache))
            present_coeffs = list(set(coeff_list) & set(coeff_list_cache))
            if missing_coeffs:
                rge_runner_missing = rge_runner.clone_runner(missing_coeffs)
                _logger.warning(
                    f"The cached RGE matrix does not contain all the requested coefficients. "
                    f"Missing coefficients: {missing_coeffs}. "
                    f"These will be computed from scratch and added to the cached matrix."
                )
                rgemat_new = rge_runner_missing.RGEmatrix(scale)
                # concatenate the new matrix to the cached one, filling missing values with zeros
                rgemat_scale = pd.concat(
                    [rgemat_scale[present_coeffs], rgemat_new], axis=1
                ).fillna(0)
                # reorder columns
                rgemat_scale = rgemat_scale[sorted(rgemat_scale.columns.tolist())]
                # Update the cache under the key we matched, not under `scale`:
                # the two differ by up to `rtol` and writing to a new key would
                # leave the narrower matrix behind as a stale duplicate.
                rge_cache[cached_key] = rgemat_scale
            else:
                # take the columns corresponding to the requested coefficients
                rgemat_scale = rgemat_scale[coeff_list]
        else:
            rgemat_scale = rge_runner.RGEmatrix(scale)
            # cache the RGE matrix
            rge_cache[scale] = rgemat_scale

        unique_rgemats[scale] = rgemat_scale

    # Map back to original order, using copies to avoid shared references
    return [unique_rgemats[scale].copy() for scale in scales]


def _resolve_scales(rge_dict, theory_group):
    """
    Resolve observable scales from an rge_dict and a TheoryGroup.

    Returns a single-element list when ``obs_scale`` is a fixed number, or
    one entry per data point when it is ``'dynamic'`` (the default).
    """
    obs_scale = rge_dict.get("obs_scale", "dynamic")
    if isinstance(obs_scale, (float, int)):
        return [float(obs_scale)]
    scale_variation = rge_dict.get("scale_variation", 1.0)
    scales = theory_group.scales.tolist()
    if scale_variation != 1.0:
        _logger.info("Applying scale variation of %s.", scale_variation)
        scales = [s * scale_variation for s in scales]
    return scales


def load_rge_matrix(
    rge_dict,
    coeff_list,
    theory_group,
):
    """
    Load the RGE matrix for the SMEFT Wilson coefficients.

    Parameters
    ----------
    rge_dict: dict
        dictionary with the RGE input parameter options
    coeff_list: list
        list of Wilson coefficients to be included in the RGE matrix
    theory_group: TheoryGroup
        theory group providing per-data-point observable scales

    Returns
    -------
    RGEMatrix
        Dataclass with stacked matrices and operator/scale metadata.
    """
    # Sort the coefficient list alphabetically
    coeff_list = sorted(coeff_list)
    scales = _resolve_scales(rge_dict, theory_group)
    settings = RGESettings.from_dict(rge_dict)
    rge_cache = {}
    rge_runner = settings.runner(coeff_list)

    # load precomputed RGE matrix if it exists. The path is already resolved and
    # fetched by smefitConfig.parse_rge; callers outside reportengine that pass
    # a raw path should resolve it themselves.
    path_to_rge_mat = rge_dict.get("rg_matrix", None)
    if path_to_rge_mat:
        rge_cache = load_precomputed_rge_matrix(path_to_rge_mat, settings.to_dict())

    # compute or fetch the RGE matrix for each scale
    rgemats = load_rge_mats_from_scales(scales, coeff_list, rge_runner, rge_cache)

    # Collect the union of all observable operators across scales
    all_obs_ops = set()
    for df in rgemats:
        all_obs_ops.update(df.index)
    obs_operators = sorted(all_obs_ops)

    # Pad matrices so every scale has the same rows, then sort alphabetically
    for mat in rgemats:
        for op in obs_operators:
            if op not in mat.index:
                mat.loc[op] = np.zeros(len(mat.columns))
        mat.sort_index(inplace=True)

    # now stack the matrices in a 3D array
    stacked_mats = jnp.stack([mat.values for mat in rgemats])

    return RGEMatrix(
        stacked_mats=stacked_mats,
        obs_operators=obs_operators,
        init_operators=coeff_list,
        scales=scales,
        settings=settings,
    )
