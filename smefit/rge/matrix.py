"""The on-disk data model: :class:`RGEMatrix`.

Owns both directions of the ``rge_matrix.pkl`` payload layout —
:meth:`RGEMatrix.write` produces it, :func:`read_rge_cache`
consumes it — so the format is described in exactly one module.
"""

import logging
import pathlib
import pickle
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


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
    settings : dict
        The running configuration these matrices were computed with, as
        returned by :attr:`smefit.rge.runner.RGE.settings`; stored in the pickle
        so a later run can check reusability.

    Notes
    -----
    The file :meth:`write` produces is a *scale-keyed cache*: one frame per
    unique scale, with no record of which data point sits at which scale. That
    is what makes it reusable by a later runcard over different data (see
    :func:`smefit.rge.matrix.read_rge_cache` and
    :func:`smefit.rge.loading.resolve_rge_matrices`).
    """

    stacked_mats: jnp.ndarray
    obs_operators: list
    init_operators: list
    scales: list
    settings: dict

    FILENAME = "rge_matrix.pkl"

    def to_dump_dict(self):
        """Build the on-disk payload: ``{'rge_settings': {...}, <scale>: DataFrame}``.

        Duplicate scales collapse to a single entry, so a dynamic-scale fit over
        many data points sharing a scale stores one frame per unique scale.
        """
        to_dump = {"rge_settings": self.settings}
        for scale, matrix in zip(self.scales, self.stacked_mats):
            to_dump[scale] = pd.DataFrame(
                np.asarray(matrix, dtype=float),
                index=self.obs_operators,
                columns=self.init_operators,
            )
        return to_dump

    def write(self, output_path, name=None):
        """Pickle this matrix to ``<output_path>/<name>``, defaulting to :attr:`FILENAME`.

        ``name`` is a complete file name, extension included — the server layer
        discovers these matrices by exact name, so anything other than the
        default is invisible to it.

        The file can be fed back to a later runcard through ``rge.rg_matrix``.
        """
        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / (name or self.FILENAME)
        with open(out_file, "wb") as f:
            pickle.dump(self.to_dump_dict(), f)
        _logger.info("RGE matrix written to %s.", out_file)


def read_rge_cache(path_to_rge_mat, rge_settings):
    """
    Read a precomputed RGE matrix pickle and validate its settings.

    The read counterpart of :meth:`RGEMatrix.to_dump_dict`, and with it the sole
    owner of the payload layout: every key other than ``'rge_settings'`` is a
    scale. Keeping that rule in one place matters because the flat float-keyed
    format is a compatibility contract — :func:`smefit.rge.loading._find_cached_scale`
    does arithmetic on those keys, so a stray non-numeric one would break every
    reader, including older smefit installs reading a matrix shared through the
    server.

    Parameters
    ----------
    path_to_rge_mat : str or pathlib.Path
        Path to the pickle file containing the precomputed RGE matrices.
    rge_settings : dict
        Expected RGE settings to validate against the stored file, as returned
        by :attr:`smefit.rge.runner.RGE.settings`.

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
        payload = pickle.load(f)

    if rge_settings != payload["rge_settings"]:
        raise ValueError("RGE settings do not match RGE matrix precomputed settings.")

    _logger.info("Loaded precomputed RGE matrix from %s.", path_to_rge_mat)
    return {k: v for k, v in payload.items() if k != "rge_settings"}
