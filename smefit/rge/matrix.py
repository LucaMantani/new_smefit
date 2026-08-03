"""The on-disk data model: :class:`RGESettings` and :class:`RGEMatrix`.

Owns the ``rge_matrix.pkl`` payload layout (see :func:`_read_rge_pickle`) —
the only other reader/writer of that layout is :mod:`smefit.rge.loading`.
"""

import logging
import pathlib
import pickle
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .runner import RGE

_logger = logging.getLogger(__name__)


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
    The file :meth:`write` produces is a *scale-keyed cache*: one frame per
    unique scale, with no record of which data point sits at which scale. That
    is what makes it reusable by a later runcard over different data (see
    :func:`smefit.rge.loading.load_precomputed_rge_matrix` and
    :func:`smefit.rge.loading.load_rge_mats_from_scales`), and it is why
    :meth:`from_file` needs the caller to supply the per-data-point ``scales``
    — `FitResult` stores them alongside, in ``fit_results.json``.
    """

    stacked_mats: jnp.ndarray
    obs_operators: list
    init_operators: list
    scales: list
    settings: RGESettings

    FILENAME = "rge_matrix.pkl"

    @classmethod
    def path_in(cls, fit_dir):
        """Path to the ``rge_matrix.pkl`` companion file inside a fit directory."""
        return pathlib.Path(fit_dir) / cls.FILENAME

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

    def write(self, output_path, name=None):
        """Pickle this matrix to ``<output_path>/<name>.pkl`` (default: `FILENAME`).

        The file can be fed back to a later runcard through ``rge.rg_matrix``.
        """
        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{name}.pkl" if name else self.path_in(output_path)
        with open(out_file, "wb") as f:
            pickle.dump(self.to_dump_dict(), f)
        _logger.info("RGE matrix written to %s.", out_file)

    @classmethod
    def from_file(cls, path, scales):
        """Rebuild an RGEMatrix from a pickle written by :meth:`write`, or None.

        Returns None if `scales` is falsy (nothing to rebuild — fits that ran
        without RGE evolution, or results written before their caller started
        recording scales) or if `path` does not exist (the companion file is
        missing, e.g. lost or not fetched), logging a warning in the latter
        case rather than failing the whole load.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to an ``rge_matrix.pkl``.
        scales : list of float
            The per-data-point scales this matrix was built for. The file only
            stores one frame per *unique* scale, so this is what restores the
            row-per-data-point stacking of ``stacked_mats``.

        Raises
        ------
        ValueError
            If the file has no frame for one of the requested scales, i.e. it
            does not belong to this fit.
        """
        if not scales:
            return None
        path = pathlib.Path(path)
        if not path.exists():
            _logger.warning("%s is missing; loading without its RGE matrix.", path)
            return None

        settings_dict, frames = _read_rge_pickle(path)
        if not frames:
            raise ValueError(f"{path} contains no RGE matrices.")

        # Resolve each distinct requested scale once: the stored float keys can
        # differ from the requested ones in the last bits.
        resolved = {}
        for scale in dict.fromkeys(scales):
            key = _find_cached_scale(frames, scale)
            if key is None:
                raise ValueError(
                    f"{path} has no RGE matrix for scale {scale} GeV; it holds "
                    f"{sorted(frames)}. This file belongs to a different fit."
                )
            resolved[scale] = frames[key]

        reference = next(iter(frames.values()))
        return cls(
            stacked_mats=jnp.stack([resolved[scale].values for scale in scales]),
            obs_operators=list(reference.index),
            init_operators=list(reference.columns),
            scales=list(scales),
            settings=RGESettings.from_dict(settings_dict),
        )


def _read_rge_pickle(path_to_rge_mat):
    """Unpickle an RGE matrix file into ``(settings_dict, {scale: DataFrame})``.

    Sole owner of the payload layout: every key other than ``'rge_settings'`` is
    a scale. Keeping that rule in one place matters because the flat float-keyed
    format is a compatibility contract — `_find_cached_scale` does arithmetic on
    those keys, so a stray non-numeric one would break every reader, including
    older smefit installs reading a matrix shared through the server.
    """
    with open(path_to_rge_mat, "rb") as f:
        payload = pickle.load(f)
    settings = payload["rge_settings"]
    frames = {k: v for k, v in payload.items() if k != "rge_settings"}
    return settings, frames


def _find_cached_scale(cache: dict, scale: float, rtol: float = 1e-5) -> float | None:
    """Return the matching key in cache if one exists within relative tolerance, else None."""
    for key in cache:
        if abs(key - scale) <= rtol * abs(key):
            return key
    return None
