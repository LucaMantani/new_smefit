"""The on-disk data model: :class:`RGEMatrix`.

Owns both directions of the ``rge_matrix.pkl`` payload layout —
:meth:`RGEMatrix.write` produces it, :meth:`RGEMatrix.read_cache`
consumes it — so the format is described in exactly one module.
"""

import logging
import numbers
import pathlib
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from smefit.paths import fetch_fit_if_missing, resolve_path

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
    obs_operators : list[str]
        Observable-basis operator names (alphabetically sorted).
    init_operators : list[str]
        Initial Wilson coefficient names (alphabetically sorted).
    scales : list[float]
        One scale per data point for dynamic mode, or a single-element list for
        a fixed observable scale.
    settings : dict[str, Any]
        The running configuration these matrices were computed with, as
        returned by :attr:`smefit.rge.runner.RGE.settings`; stored in the pickle
        so a later run can check reusability.

    Notes
    -----
    The file :meth:`write` produces is a *scale-keyed cache*: one frame per
    unique scale, with no record of which data point sits at which scale. That
    is what makes it reusable by a later runcard over different data (see
    :meth:`read_cache` and :func:`smefit.rge.build.resolve_rge_matrices`).
    """

    stacked_mats: jnp.ndarray
    obs_operators: list[str]
    init_operators: list[str]
    scales: list[float]
    settings: dict[str, Any]

    FILENAME = "rge_matrix.pkl"

    def to_dump_dict(self) -> dict[str | float, Any]:
        """Build the on-disk payload: ``{'rge_settings': {...}, <scale>: DataFrame}``.

        Duplicate scales collapse to a single entry, so a dynamic-scale fit over
        many data points sharing a scale stores one frame per unique scale.

        Returns
        -------
        dict[str | float, Any]
            :attr:`settings` under the ``'rge_settings'`` key, plus one
            ``pandas.DataFrame`` per unique scale keyed by that scale.
        """
        to_dump = {"rge_settings": self.settings}
        for scale, matrix in zip(self.scales, self.stacked_mats):
            to_dump[scale] = pd.DataFrame(
                np.asarray(matrix, dtype=float),
                index=self.obs_operators,
                columns=self.init_operators,
            )
        return to_dump

    def write(self, output_path: str | pathlib.Path, name: str | None = None) -> None:
        """Pickle this matrix to ``<output_path>/<name>``, defaulting to :attr:`FILENAME`.

        The file can be fed back to a later runcard through ``rge.rg_matrix``.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Destination *directory*, created if it does not exist.
        name : str, optional
            A complete file name, extension included — the server layer
            discovers these matrices by exact name, so anything other than the
            default :attr:`FILENAME` is invisible to it.
        """
        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / (name or self.FILENAME)
        with open(out_file, "wb") as f:
            pickle.dump(self.to_dump_dict(), f)
        _logger.info("RGE matrix written to %s.", out_file)

    @staticmethod
    def _validate_payload(payload: Any, path: pathlib.Path) -> None:
        """Check an unpickled object against the layout :meth:`to_dump_dict` writes.

        Anything can be pickled, and `rg_matrix` is a user-supplied path, so a
        wrong file (a fit result, someone else's pickle, a truncated download)
        would otherwise surface as a `KeyError`/`TypeError` deep inside
        :func:`smefit.rge.build.resolve_rge_matrices`, or — for a non-numeric
        key — as a silent cache miss that recomputes everything.

        Parameters
        ----------
        payload : Any
            Whatever ``pickle.load`` returned.
        path : pathlib.Path
            The file it came from, named in every error message.

        Raises
        ------
        ValueError
            If *payload* does not have the documented layout.
        """
        prefix = f"'{path}' is not a valid RGE matrix cache"
        if not isinstance(payload, dict):
            raise ValueError(
                f"{prefix}: expected a dict, got {type(payload).__name__}."
            )
        if "rge_settings" not in payload:
            raise ValueError(f"{prefix}: missing the 'rge_settings' entry.")
        if not isinstance(payload["rge_settings"], dict):
            raise ValueError(
                f"{prefix}: 'rge_settings' must be a dict, got "
                f"{type(payload['rge_settings']).__name__}."
            )

        scales = [k for k in payload if k != "rge_settings"]
        if not scales:
            raise ValueError(f"{prefix}: it holds no matrices, only 'rge_settings'.")
        for scale in scales:
            # _find_cached_scale does arithmetic on these keys
            if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
                raise ValueError(
                    f"{prefix}: every key other than 'rge_settings' must be a "
                    f"scale in GeV, found {scale!r}."
                )
            if not isinstance(payload[scale], pd.DataFrame):
                raise ValueError(
                    f"{prefix}: the entry for scale {scale} must be a DataFrame, "
                    f"got {type(payload[scale]).__name__}."
                )

    @staticmethod
    def read_cache(
        path_to_rge_mat: str | pathlib.Path, rge_settings: Mapping[str, Any]
    ) -> dict[float, pd.DataFrame]:
        """
        Read a precomputed RGE matrix pickle and validate its settings.

        The read counterpart of :meth:`to_dump_dict`, and with it the sole
        owner of the payload layout: every key other than ``'rge_settings'`` is a
        scale. Keeping that rule in one place matters because the flat float-keyed
        format is a compatibility contract — :func:`smefit.rge.build._find_cached_scale`
        does arithmetic on those keys, so a stray non-numeric one would break every
        reader, including older smefit installs reading a matrix shared through the
        server.

        A ``staticmethod`` rather than a constructor: the pickle keys frames by
        *unique* scale, so it cannot rebuild the per-data-point
        :attr:`stacked_mats` and therefore cannot return an ``RGEMatrix``. It
        lives on the class anyway so that both directions of the format sit
        together.

        Parameters
        ----------
        path_to_rge_mat : str or pathlib.Path
            Path to the pickle file containing the precomputed RGE matrices.
            Prefix-relative form (``smefit_results/fits/my_fit/rge_matrix.pkl``)
            is resolved here through :func:`smefit.paths.resolve_path`, and a
            file missing under ``smefit_results/{fits,reports}/<name>/`` is
            downloaded from the server.
        rge_settings : Mapping[str, Any]
            Expected RGE settings to validate against the stored file, as returned
            by :attr:`smefit.rge.runner.RGE.settings`.

        Returns
        -------
        dict[float, pandas.DataFrame]
            Cached RGE matrices keyed by scale in GeV, excluding the
            ``'rge_settings'`` entry.

        Raises
        ------
        FileNotFoundError
            If the (resolved) file does not exist and could not be downloaded.
        ValueError
            If the file is not a readable pickle, does not have the layout
            :meth:`to_dump_dict` writes, or its stored settings do not match
            `rge_settings`.
        """
        path = pathlib.Path(resolve_path(str(path_to_rge_mat)))
        fetch_fit_if_missing(path)
        if not path.exists():
            # Name both forms: an unknown prefix is returned unchanged by
            # resolve_path, so the two differing tells the user which it was.
            raise FileNotFoundError(
                f"RGE matrix '{path_to_rge_mat}' not found (resolved to '{path}')."
            )

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
            raise ValueError(f"'{path}' is not a readable pickle: {e}") from e

        RGEMatrix._validate_payload(payload, path)

        if rge_settings != payload["rge_settings"]:
            raise ValueError(
                "RGE settings do not match RGE matrix precomputed settings."
            )

        _logger.info("Loaded precomputed RGE matrix from %s.", path)
        return {k: v for k, v in payload.items() if k != "rge_settings"}
