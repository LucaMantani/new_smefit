"""The RGE data model: :class:`RGEMatrix` and :class:`RGECache`.

Owns both directions of the ``rge_matrix.pkl`` payload layout —
:meth:`RGEMatrix.write` produces it, :meth:`RGECache.load` consumes it — so
the format is described in exactly one module.

The two classes are the two shapes the same matrices take. An
:class:`RGEMatrix` is what a run computes: one matrix per *data point*, ready to
contract with the theory tables. An :class:`RGECache` is what the file holds:
one matrix per *unique scale*, with no record of which data point sits at which
scale — what makes it reusable by a later run over different data, and why it
cannot be turned back into an :class:`RGEMatrix` on its own. Rebuilding one goes
through :func:`smefit.rge.build_rge_matrix` with the file as its ``rg_matrix``,
which finds every scale already there.
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

from smefit.paths import fetch_fit_if_missing, resolve_fit_dir, resolve_path

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
    The file :meth:`write` produces is a *scale-keyed cache*, read back as an
    :class:`RGECache`: one frame per unique scale, with no record of which data
    point sits at which scale. That is what makes it reusable by a later
    runcard over different data (see :meth:`read_cache` and
    :func:`smefit.rge.build.resolve_rge_matrices`).
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
    def locate(ref: str | pathlib.Path) -> pathlib.Path:
        """Turn an ``rg_matrix`` reference into the path of the pickle it names.

        Three forms are accepted, told apart by their shape::

            rg_matrix: my_fit                                     # a fit name
            rg_matrix: smefit_results/fits/my_fit/CMS_DYMee_13TeV # a directory
            rg_matrix: smefit_results/fits/my_fit/rge_matrix.pkl  # the file itself

        A bare name (no ``/``, no ``.pkl``) is a fit name, looked up like a
        ``fits:`` entry through :func:`smefit.paths.resolve_fit_dir` — in
        ``smefit_results/fits/`` and downloaded from the server if missing. A
        local directory with a bare name must therefore be written ``./name``.
        Any other path is prefix-resolved, and fetched from the server when it
        lies under ``smefit_results/{fits,reports}/<name>/``; a directory then
        stands for the :attr:`FILENAME` inside it.

        Only builds the path: reading it is :meth:`read_cache`'s job.

        Parameters
        ----------
        ref : str or pathlib.Path
            The ``rg_matrix`` value, as written in a runcard.

        Returns
        -------
        pathlib.Path
            The existing pickle file.

        Raises
        ------
        FileNotFoundError
            If the fit, directory or file does not exist and could not be
            downloaded, or a directory holds no :attr:`FILENAME`.
        """
        ref_str = str(ref)
        if ref_str.endswith(".pkl"):
            path = pathlib.Path(resolve_path(ref_str))
            fetch_fit_if_missing(path)
        else:
            if "/" not in ref_str:
                fit_dir = resolve_fit_dir(ref_str)
            else:
                fit_dir = pathlib.Path(resolve_path(ref_str))
                fetch_fit_if_missing(fit_dir)
            path = fit_dir / RGEMatrix.FILENAME

        if not path.exists():
            # Name both forms: an unknown prefix is returned unchanged by
            # resolve_path, so the two differing tells the user which it was.
            raise FileNotFoundError(
                f"RGE matrix '{ref}' not found (resolved to '{path}')."
            )
        return path

    @staticmethod
    def read_cache(
        path_to_rge_mat: str | pathlib.Path, rge_settings: Mapping[str, Any]
    ) -> dict[float, pd.DataFrame]:
        """Load the cached matrices *path_to_rge_mat* names, checked against *rge_settings*.

        What every consumer of a precomputed RGE matrix calls — the dataset
        pipeline through :func:`smefit.rge.build_rge_matrix`, and external chi2
        modules directly. It stays on :class:`RGEMatrix`, where external chi2
        modules call it.

        Parameters
        ----------
        path_to_rge_mat : str or pathlib.Path
            A fit name, a directory or the pickle file itself, loaded through
            :meth:`RGECache.load` — so a prefix-relative path works, and a fit
            missing locally is downloaded from the server.
        rge_settings : Mapping[str, Any]
            The settings the caller runs with, as returned by
            :attr:`smefit.rge.runner.RGE.settings`; the cache must have been
            computed with exactly these.

        Returns
        -------
        dict[float, pandas.DataFrame]
            Cached RGE matrices keyed by scale in GeV.

        Raises
        ------
        FileNotFoundError
            If *path_to_rge_mat* names nothing that exists or could be
            downloaded.
        ValueError
            If the file is not an RGE matrix cache, or was computed with
            settings other than `rge_settings`.
        """
        cache = RGECache.load(path_to_rge_mat)
        if dict(rge_settings) != cache.settings:
            raise ValueError(
                "RGE settings do not match RGE matrix precomputed settings."
            )
        return cache.frames


@dataclass
class RGECache:
    """The contents of an ``rge_matrix.pkl``: matrices keyed by scale.

    The read-side counterpart of :class:`RGEMatrix` — see the module docstring
    for why the two differ. Loaded from a file with :meth:`load`, which is
    what :meth:`RGEMatrix.read_cache` does before checking it against the
    reader's settings, and what a loaded fit holds as
    :attr:`smefit.fit_result.Fit.rge_cache`. The pipeline itself only ever
    takes a reference to the file (``rg_matrix``), never one of these.

    Attributes
    ----------
    settings : dict[str, Any]
        The running configuration the matrices were computed with, as
        :attr:`smefit.rge.runner.RGE.settings` returns it.
    frames : dict[float, pandas.DataFrame]
        One matrix per scale in GeV: observable-basis operators as rows,
        initial coefficients as columns.
    source : pathlib.Path or None
        The file it was loaded from, for messages; None when built in memory.
    """

    settings: dict[str, Any]
    frames: dict[float, pd.DataFrame]
    source: pathlib.Path | None = None

    @classmethod
    def load(cls, ref: str | pathlib.Path) -> "RGECache":
        """Load the cache *ref* names, after checking it is one.

        Parameters
        ----------
        ref : str or pathlib.Path
            A fit name, a directory or the pickle file itself, resolved through
            :meth:`RGEMatrix.locate` — so a prefix-relative path works, and a
            fit missing locally is downloaded from the server.

        Raises
        ------
        FileNotFoundError
            If *ref* names nothing that exists or could be downloaded.
        ValueError
            If the file is not a readable pickle or does not have the layout
            :meth:`RGEMatrix.to_dump_dict` writes.
        """
        path = RGEMatrix.locate(ref)
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
            raise ValueError(f"'{path}' is not a readable pickle: {e}") from e

        cls._validate_payload(payload, path)
        _logger.info("Loaded precomputed RGE matrix from %s.", path)
        return cls(
            settings=payload["rge_settings"],
            frames={k: v for k, v in payload.items() if k != "rge_settings"},
            source=path,
        )

    @staticmethod
    def _validate_payload(payload: Any, path: pathlib.Path) -> None:
        """Check an unpickled object against the layout :meth:`RGEMatrix.to_dump_dict` writes.

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
