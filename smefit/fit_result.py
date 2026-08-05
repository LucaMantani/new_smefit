"""
smefit.fit_result.py

Fit containers shared across fitting routines.

Two classes live here:

- :class:`FitResult`, the bare numerical outcome of a fit — free parameters,
  best-fit point, likelihood, samples — and its ``fit_results.json`` schema.
  This is what the fitting routines return and write.
- :class:`Fit`, a fit that already exists on disk: the :class:`FitResult` it
  produced, held as an attribute, plus the metadata describing how it was run.
  It is built only by loading, never by a fitting routine, and it is what
  downstream consumers (plotting, reports) work with.

New attributes describing a fit belong on :class:`Fit`, which is meant to grow;
``FitResult`` stays the minimal numerical record and owns the
``fit_results.json`` schema on its own.
"""

import json
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import jax.numpy as jnp
import yaml
from rich import box
from rich.console import Console
from rich.table import Table

from smefit.priors import _build_dist
from smefit.whitening import WhitenTransform

log = logging.getLogger(__name__)


# Fit actions are named "run_<fit type>_fit", or "run_individual_<fit
# type>_fits" when the fit is run one coefficient at a time.
_FIT_ACTION_RE = re.compile(r"^run_(individual_)?(?P<fit_type>.+?)_fits?$")


def _parse_fit_action(action: str):
    """``(fit type, ran individually)`` for a fit action, ``None`` for anything else.

    The action a runcard ran is what says which sampler produced a fit and how
    it was driven, and its name spells both out. It is looked up in
    :mod:`smefit.fit_actions` too, so that an action which merely reads like a
    fit is not taken for one.
    """
    from smefit import fit_actions  # imported here: fit_actions imports this module

    match = _FIT_ACTION_RE.match(action)
    if match is None or not callable(getattr(fit_actions, action, None)):
        return None
    return match["fit_type"], action.startswith("run_individual_")


def _fit_runcard(path: pathlib.Path) -> Optional[Dict]:
    """The runcard a fit was run with, or None when it cannot be found.

    A fit directory keeps a copy in ``input/runcard.yaml``. The per-coefficient
    subdirectories of an individual fit do not, so for those the runcard of the
    parent fit — the one that ran them — is used instead.
    """
    candidates = [path / "input" / "runcard.yaml"]
    if path.parent.name == "individual_fits":
        candidates.append(path.parent.parent / "input" / "runcard.yaml")

    for runcard in candidates:
        if runcard.exists():
            with runcard.open() as f:
                return yaml.safe_load(f) or {}
    return None


def _runcard_actions(config: Mapping) -> List[str]:
    """Action names listed under ``actions_``, without their arguments."""
    actions = config.get("actions_") or []
    names = []
    for action in actions:
        # nested-namespace entries are mappings, not actions themselves
        if not isinstance(action, str):
            continue
        names.append(action.split("(", 1)[0].strip())
    return names


def _as_joint_payload(d: Mapping) -> Dict:
    """A payload normalised to the single-fit schema :class:`FitResult` reads.

    The summary of individual fits records one chi2 and one evidence per
    coefficient rather than a single joint value; there is no joint likelihood
    to report, so it becomes NaN and the evidence is dropped.
    """
    d = dict(d)
    max_loglikelihood = d.get("max_loglikelihood")
    d["max_loglikelihood"] = (
        float("nan") if max_loglikelihood is None else float(max_loglikelihood)
    )
    if isinstance(d.get("logz"), dict):
        d["logz"] = None
    return d


def _metadata_from_runcard(path: pathlib.Path):
    """How a fit was configured, read from the runcard it was run with.

    The runcard is the authoritative record of a fit's configuration, so it is
    the only source of this metadata — ``fit_results.json`` holds the numbers a
    fit produced, not the settings it was given.

    Returns ``(use_quad, fit_type, ran_individually)``.
    """
    config = _fit_runcard(path)
    if config is None:
        log.warning(
            "No input/runcard.yaml found for '%s': assuming a linear, joint "
            "fit of unknown type. Consumers that depend on it (e.g. the "
            "contour style of plots) may need it set explicitly.",
            path,
        )
        return False, None, False

    for action in _runcard_actions(config):
        parsed = _parse_fit_action(action)
        if parsed is not None:
            fit_type, ran_individually = parsed
            break
    else:
        log.warning(
            "The runcard of '%s' runs no known fit action: its fit type is "
            "left unset.",
            path,
        )
        fit_type, ran_individually = None, False

    return bool(config.get("use_quad", False)), fit_type, ran_individually


def _format_prior(spec: Optional[Mapping]) -> str:
    if spec is None:
        return "-"
    if spec.get("dist") == "exact_posterior":
        return f"ExactPosterior"
    return str(_build_dist(spec))


@dataclass
class FitResult:
    """Container for the numerical result of a fit.

    Attributes
    ----------
    free_parameters : list[str]
        Names of free (fitted) parameters, in CoefficientGroup order.
    best_fit_point : dict[str, float]
        Best-fit value for every coefficient (free and derived).
    max_loglikelihood : float
        Maximum log-likelihood value: ``-chi2_best / 2``.
    num_data : int
        Total number of data points used in the fit.
    logz : float or None
        Log evidence (only filled by nested-sampling routines).
    samples : dict[str, jnp.ndarray] or None
        Posterior samples for every coefficient,
        shape ``(n_samples,)`` per entry.
    whitening_transformation : WhitenTransform or None
        Affine unwhitening transform (matrix, shift). Saved when whitening is
        active.
    """

    free_parameters: List[str]
    best_fit_point: Dict[str, float]
    max_loglikelihood: float
    num_data: int
    logz: Optional[float] = None
    samples: Optional[Dict[str, jnp.ndarray]] = None
    prior_specs: Optional[Dict[str, Mapping]] = None
    whitening_transformation: Optional[WhitenTransform] = None
    whitening_active: bool = False

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def n_free(self) -> int:
        return len(self.free_parameters)

    @property
    def ndof(self) -> int:
        return self.num_data - self.n_free

    @property
    def chi2_val(self) -> float:
        return float(-2.0 * self.max_loglikelihood)

    @property
    def chi2_ndof(self) -> float:
        return self.chi2_val / self.ndof if self.ndof > 0 else float("nan")

    @property
    def std(self) -> Dict[str, float]:
        """Standard deviation of posterior samples per coefficient."""
        if self.samples is None:
            return {}
        return {name: float(jnp.std(vals)) for name, vals in self.samples.items()}

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        return float(self.n_free * jnp.log(self.num_data) - 2 * self.max_loglikelihood)

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return float(2 * self.n_free - 2 * self.max_loglikelihood)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a coloured summary table using ``rich``."""
        console = Console()
        unc = self.std

        # --- header panel ---
        console.rule("[bold cyan]Fit Result[/bold cyan]")
        console.print(f"  [bold]n_data[/bold]   = {self.num_data}")
        console.print(f"  [bold]n_free[/bold]   = {self.n_free}")
        console.print(f"  [bold]ndof[/bold]     = {self.ndof}")
        console.print(f"  [bold]chi2[/bold]     = [yellow]{self.chi2_val:.4f}[/yellow]")
        console.print(
            f"  [bold]chi2/dof[/bold] = [{'green' if self.chi2_ndof < 2 else 'red'}]"
            f"{self.chi2_ndof:.4f}[/]"
        )
        if self.logz is not None:
            console.print(f"  [bold]log Z[/bold]    = [cyan]{self.logz:.4f}[/cyan]")

        # --- coefficient table ---
        table = Table(
            box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta"
        )
        table.add_column("Coefficient", style="cyan", no_wrap=True)
        table.add_column("Best fit", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Type", justify="center", style="dim")
        show_prior = bool(self.prior_specs)
        if show_prior:
            prior_col = "Prior (whitened)" if self.whitening_active else "Prior"
            table.add_column(prior_col, justify="left", style="dim")

        for name, val in self.best_fit_point.items():
            u = unc.get(name, float("nan"))
            kind = "free" if name in self.free_parameters else "derived"
            row = [name, f"{val:.6f}", f"{u:.6f}", kind]
            if show_prior:
                row.append(_format_prior(self.prior_specs.get(name)))
            table.add_row(*row)

        console.print(table)
        console.rule(style="dim")

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, output_path) -> None:
        """Serialise this result to JSON and write it to *output_path*."""
        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        unc = self.std
        payload = {
            "free_parameters": self.free_parameters,
            "num_data": self.num_data,
            "n_free": self.n_free,
            "ndof": self.ndof,
            "max_loglikelihood": self.max_loglikelihood,
            "chi2": self.chi2_val,
            "chi2_ndof": self.chi2_ndof,
            "logz": self.logz,
            "best_fit_point": self.best_fit_point,
            "std": unc,
            "bic": self.bic,
            "aic": self.aic,
            "samples": (
                {name: vals.tolist() for name, vals in self.samples.items()}
                if self.samples is not None
                else None
            ),
            "prior_specs": self.prior_specs,
            "whitening_transformation": (
                self.whitening_transformation.to_dict()
                if self.whitening_transformation is not None
                else None
            ),
            "whitening_active": self.whitening_active,
        }

        out_file = output_path / "fit_results.json"
        with out_file.open("w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path) -> "FitResult":
        """Load a FitResult from a directory containing fit_results.json."""
        p = pathlib.Path(path) / "fit_results.json"
        with p.open() as f:
            d = json.load(f)
        return cls.from_payload(d)

    @classmethod
    def from_payload(cls, d: Mapping) -> "FitResult":
        """Rebuild a FitResult from a decoded ``fit_results.json`` payload.

        Split out of :meth:`from_json` so that a reader which has already
        inspected the payload — :meth:`Fit.from_json` — can hand it straight
        over instead of decoding the file twice.
        """
        free_parameters = d["free_parameters"]
        samples = (
            {name: jnp.array(vals) for name, vals in d["samples"].items()}
            if d.get("samples")
            else None
        )
        whitening_transformation = (
            WhitenTransform.from_dict(d["whitening_transformation"])
            if d.get("whitening_transformation")
            else None
        )
        return cls(
            free_parameters=free_parameters,
            best_fit_point=d["best_fit_point"],
            max_loglikelihood=d["max_loglikelihood"],
            num_data=d["num_data"],
            logz=d.get("logz"),
            samples=samples,
            prior_specs=d.get("prior_specs"),
            whitening_transformation=whitening_transformation,
            whitening_active=d.get("whitening_active", False),
        )


@dataclass
class Fit:
    """A fit that exists on disk: what it produced, and how it was run.

    A ``Fit`` is built only by loading — :meth:`from_json` — never by a fitting
    routine, which returns the bare :class:`FitResult` it computed and knows
    nothing of the directory it will be written to. This is the class to extend
    when a fit needs to carry something new about itself; ``FitResult`` stays
    the numerical record.

    None of this metadata is serialised: a fit's numbers live in
    ``fit_results.json``, while how it was configured is read back from the
    runcard it was run with (``input/runcard.yaml`` in the fit directory), the
    authoritative record of that.

    Attributes
    ----------
    fit_results : FitResult
        The numbers the fit produced — best-fit point, likelihood, samples.
    fit_name : str
        Identity of the fit: the name of the directory it was loaded from,
        which is what users refer to it by.
    label : str or None
        Legend label, from the ``label`` key of the fit's ``fits`` entry. It is
        used verbatim, so it may be raw LaTeX; when unset, consumers fall back
        to ``fit_name``. How the fit is presented in a given plot, not a
        property of the fit itself, so it comes from the runcard that loads the
        fit rather than from the fit directory.
    fit_type : str or None
        Which routine produced the fit — ``analytic``, ``hessian``,
        ``ultranest`` or ``blackjax`` — from the fit action the runcard ran.
        None when the runcard cannot be found.
    use_quad : bool
        Whether the fit included quadratic EFT corrections, from ``use_quad``
        in the runcard. Downstream consumers need it after the run: it is what
        says whether the posterior can be expected to be Gaussian.
    individual_fit : bool
        True for the summary of one-at-a-time individual fits, i.e. when the
        runcard ran a ``run_individual_*_fits`` action — the only thing that
        says so. Every coefficient was then fitted with the others held at
        their baseline, so the samples are independent 1D posteriors rather
        than a joint one — see :class:`FitResultGroup`.
    """

    fit_results: FitResult
    fit_name: str
    label: Optional[str] = None
    fit_type: Optional[str] = None
    use_quad: bool = False
    individual_fit: bool = False

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path, label: Optional[str] = None) -> "Fit":
        """Load a Fit from a directory containing ``fit_results.json``.

        Both payloads written by this module are accepted: the standard one
        from :meth:`FitResult.write`, and the summary of individual fits from
        :meth:`FitResultGroup.write_summary`. The latter records one chi2 and
        one evidence per coefficient rather than joint ones, so its
        ``fit_results`` carries a NaN likelihood and no evidence.

        ``label`` is how the caller chooses to present the fit; the directory
        knows nothing about it, so it is the one piece of metadata passed in
        rather than read back.
        """
        path = pathlib.Path(path)
        with (path / "fit_results.json").open() as f:
            d = json.load(f)

        use_quad, fit_type, ran_individually = _metadata_from_runcard(path)

        return cls(
            fit_results=FitResult.from_payload(_as_joint_payload(d)),
            # The directory name is the identity users refer to in runcards.
            fit_name=path.name,
            label=label,
            fit_type=fit_type,
            use_quad=use_quad,
            # A per-coefficient subdirectory of an individual run holds one
            # coefficient's own fit, not the merged 1D posteriors — only the
            # summary directory is the individual fit.
            individual_fit=ran_individually and path.parent.name != "individual_fits",
        )


class FitResultGroup:
    """A collection of FitResult objects from individual parameter fits."""

    def __init__(self, results: List[FitResult]):
        self.results = results

    def print_summary(self) -> None:
        """Print a combined summary table with one row per fit."""
        console = Console()
        console.rule("[bold cyan]Individual Parameter Fits[/bold cyan]")

        table = Table(
            box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta"
        )
        show_prior = any(r.prior_specs for r in self.results)
        table.add_column("Coefficient", style="cyan", no_wrap=True)
        table.add_column("Best fit", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("chi2", justify="right")
        table.add_column("chi2/dof", justify="right")
        if show_prior:
            table.add_column("Prior", justify="left", style="dim")

        for result in self.results:
            name = result.free_parameters[0]
            val = result.best_fit_point.get(name, float("nan"))
            std = result.std.get(name, float("nan"))
            row = [
                name,
                f"{val:.6f}",
                f"{std:.6f}",
                f"{result.chi2_val:.4f}",
                f"{result.chi2_ndof:.4f}",
            ]
            if show_prior:
                spec = result.prior_specs.get(name) if result.prior_specs else None
                row.append(_format_prior(spec))
            table.add_row(*row)

        console.print(table)
        console.rule(style="dim")

    def write_results(self, output_path) -> None:
        """Write each FitResult to its own subdirectory and a combined summary."""
        base = pathlib.Path(output_path) / "individual_fits"
        for result in self.results:
            name = result.free_parameters[0]
            result.write(base / name)
        self.write_summary(output_path)

    def write_summary(self, output_path) -> None:
        """Write a combined fit_results.json aggregating all individual fits."""
        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        free_parameters = [r.free_parameters[0] for r in self.results]
        num_data = self.results[0].num_data if self.results else 0

        merged_best_fit: Dict[str, float] = {}
        merged_std: Dict[str, float] = {}
        merged_samples: Dict[str, list] = {}
        chi2_per_coeff: Dict[str, float] = {}
        chi2_ndof_per_coeff: Dict[str, float] = {}
        logz_per_coeff: Dict[str, Optional[float]] = {}
        prior_specs: Dict = {}

        for result in self.results:
            name = result.free_parameters[0]
            merged_best_fit[name] = result.best_fit_point[name]
            merged_std[name] = result.std.get(name, float("nan"))
            chi2_per_coeff[name] = result.chi2_val
            chi2_ndof_per_coeff[name] = result.chi2_ndof
            logz_per_coeff[name] = result.logz
            if result.samples is not None and name in result.samples:
                vals = result.samples[name]
                merged_samples[name] = (
                    vals.tolist() if hasattr(vals, "tolist") else list(vals)
                )
            if result.prior_specs:
                prior_specs.update(result.prior_specs)

        payload = {
            "free_parameters": free_parameters,
            "num_data": num_data,
            "n_free": len(free_parameters),
            "best_fit_point": merged_best_fit,
            "std": merged_std,
            "chi2": chi2_per_coeff,
            "chi2_ndof": chi2_ndof_per_coeff,
            "logz": logz_per_coeff,
            "samples": merged_samples if merged_samples else None,
            "prior_specs": prior_specs if prior_specs else None,
            "whitening_active": (
                self.results[0].whitening_active if self.results else False
            ),
        }

        out_file = output_path / "fit_results.json"
        with out_file.open("w") as f:
            json.dump(payload, f, indent=2)
