"""Top-level ``build_rge_matrix`` entry point: scale resolution + caching.

Ties the :class:`~smefit.rge.runner.RGE` runner and the
:class:`~smefit.rge.matrix.RGEMatrix` data model together into the function
``smefitConfig.produce_rge_matrix`` calls.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .matrix import RGEMatrix
from .runner import RGE

if TYPE_CHECKING:
    from smefit.core import TheoryGroup

_logger = logging.getLogger(__name__)


def _find_cached_scale(
    cache: Mapping[float, pd.DataFrame], scale: float, rtol: float = 1e-5
) -> float | None:
    """Return the matching key in cache if one exists within relative tolerance, else None."""
    for key in cache:
        if abs(key - scale) <= rtol * abs(key):
            return key
    return None


def resolve_rge_matrices(
    scales: Sequence[float],
    coeff_list: Sequence[str],
    rge_runner: RGE,
    rge_cache: dict[float, pd.DataFrame],
) -> list[pd.DataFrame]:
    """
    Resolve the RGE matrices for the given list of scales, from cache or by computing them.

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
    scales : Sequence[float]
        Energy scales in GeV at which to fetch/compute RGE matrices.
    coeff_list : Sequence[str]
        Requested Wilson coefficient names (columns) to return for each matrix.
    rge_runner : RGE
        An instance of the RGE runner
    rge_cache : dict[float, pandas.DataFrame]
        Mapping scale -> pandas.DataFrame for previously computed RGE matrices.
        This dict will be updated in-place when new or extended RGE matrices are
        computed.

    Returns
    -------
    list[pandas.DataFrame]
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


def _resolve_scales(
    rge_dict: Mapping[str, Any], theory_group: "TheoryGroup"
) -> list[float]:
    """
    Resolve observable scales from an rge_dict and a TheoryGroup.

    Parameters
    ----------
    rge_dict : Mapping[str, Any]
        The runcard ``rge:`` block; ``obs_scale`` and ``scale_variation`` are
        the keys read here.
    theory_group : TheoryGroup
        Source of the per-data-point scales used in ``'dynamic'`` mode.

    Returns
    -------
    list[float]
        A single-element list when ``obs_scale`` is a fixed number, or one
        entry per data point when it is ``'dynamic'`` (the default).
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


def build_rge_matrix(
    rge_dict: Mapping[str, Any],
    coeff_list: Sequence[str],
    theory_group: "TheoryGroup",
) -> RGEMatrix:
    """
    Build the RGE matrix for the SMEFT Wilson coefficients.

    Parameters
    ----------
    rge_dict: Mapping[str, Any]
        dictionary with the RGE input parameter options
    coeff_list: Sequence[str]
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
    rge_cache = {}
    rge_runner = RGE.from_rge_dict(rge_dict, coeff_list)

    # load precomputed RGE matrix if it exists. A prefix-relative path is fine:
    # smefitConfig.parse_rge resolves and fetches it early, and RGEMatrix.read_cache
    # does the same for every other caller (external chi2 modules in particular).
    path_to_rge_mat = rge_dict.get("rg_matrix", None)
    if path_to_rge_mat:
        rge_cache = RGEMatrix.read_cache(path_to_rge_mat, rge_runner.settings)

    # compute or fetch the RGE matrix for each scale
    rgemats = resolve_rge_matrices(scales, coeff_list, rge_runner, rge_cache)

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
        settings=rge_runner.settings,
    )
