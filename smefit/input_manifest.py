"""
smefit.input_manifest.py

A fit's runcard names its input files, ``input/manifest.json`` records their
versions (hashes), so that :meth:`smefit.fit_result.Fit.rebuild_runcard` can
tell when rebuilding the fit would use different ones.
"""

import hashlib
import json
import pathlib
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from smefit.paths import resolve_path

# The smefit checkout, for the commit of the code that ran the fit.
_SMEFIT_SOURCE = pathlib.Path(__file__).resolve().parents[1]


def _sha256(path: pathlib.Path) -> Optional[str]:
    """The sha256 of a file, or None when it does not exist."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(directory: pathlib.Path) -> Optional[str]:
    """The commit checked out where *directory* lives.

    None outside a git checkout, or when git is not installed: a commit is
    only a lead to what changed, never needed to check a rebuild.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _known_and_different(was: Optional[str], now: Optional[str]) -> bool:
    """Whether two commits differ; an unknown one (no git) differs from none."""
    return was is not None and now is not None and was != now


@dataclass(frozen=True)
class InputManifest:
    """The input files a fit was computed from, as they were when it ran.

    Attributes
    ----------
    files : dict of str to str or None
        sha256 of every input file, keyed by a machine-independent name:
        ``commondata/<dataset>.yaml``, ``theory/<dataset>.json`` and
        ``external_chi2/<ClassName>``; None for a file that does not exist.
        A theory file is hashed whole, so a change to any of its orders
        counts. Of an external likelihood only the module file named by its
        ``path`` is hashed: whatever it loads by itself is opaque to smefit.
    database : dict of str to str or None
        The git commit of the ``commondata`` and ``theory`` directories, to
        trace a fit back to a database version.
    smefit_commit : str or None
        The git commit of the smefit checkout that ran the fit.
    """

    files: Dict[str, Optional[str]] = field(default_factory=dict)
    database: Dict[str, Optional[str]] = field(default_factory=dict)
    smefit_commit: Optional[str] = None

    # Where in a fit directory the manifest lives, next to input/runcard.yaml.
    RELATIVE_PATH = pathlib.PurePosixPath("input", "manifest.json")

    @classmethod
    def from_runcard(cls, runcard: Mapping) -> "InputManifest":
        """The manifest of the input files *runcard* names, on this machine.

        Paths are resolved as the runcard parsers resolve them, through
        :func:`smefit.paths.resolve_path`.
        """
        datasets = runcard.get("datasets") or []
        files, database = {}, {}
        for key, role, ext in (
            ("data_path", "commondata", "yaml"),
            ("theory_path", "theory", "json"),
        ):
            if not datasets or runcard.get(key) is None:
                continue
            directory = pathlib.Path(resolve_path(str(runcard[key])))
            for ds in datasets:
                name = f"{ds['name']}.{ext}"
                files[f"{role}/{name}"] = _sha256(directory / name)
            database[role] = _git_commit(directory)
        for class_name, entry in (runcard.get("external_chi2") or {}).items():
            module = pathlib.Path(resolve_path(str(entry["path"])))
            files[f"external_chi2/{class_name}"] = _sha256(module)
        return cls(files, database, _git_commit(_SMEFIT_SOURCE))

    def differences(self, current: "InputManifest") -> Tuple[List[str], List[str]]:
        """How *current* differs from this manifest: ``(changed, notes)``.

        ``changed`` lists the input files whose content differs: a rebuild
        from them is not the fit's likelihood. ``notes`` lists what is worth a
        warning only: input files that are missing (a rebuild needing one
        fails loading it anyway), database commits that moved alongside
        changed files, and a different smefit commit.
        """
        changed, notes = [], []
        for key in sorted(self.files.keys() | current.files.keys()):
            was, now = self.files.get(key), current.files.get(key)
            if was == now:
                continue
            if now is None:
                notes.append(f"{key}: missing")
            else:
                changed.append(f"{key}: changed")

        # With the files unchanged, the rest of the database is irrelevant.
        if changed:
            for role in sorted(self.database.keys() | current.database.keys()):
                was, now = self.database.get(role), current.database.get(role)
                if _known_and_different(was, now):
                    notes.append(f"{role} database commit: {was} -> {now}")

        if _known_and_different(self.smefit_commit, current.smefit_commit):
            notes.append(
                f"smefit commit: {self.smefit_commit} -> {current.smefit_commit}"
            )
        return changed, notes

    def write(self, fit_dir) -> None:
        """Write the manifest to ``input/manifest.json`` in *fit_dir*."""
        path = pathlib.Path(fit_dir) / self.RELATIVE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, fit_dir) -> Optional["InputManifest"]:
        """The manifest of *fit_dir*; None for a fit written before manifests."""
        path = pathlib.Path(fit_dir) / cls.RELATIVE_PATH
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text()))
