"""
smefit.fit_result.py

Fit containers shared across fitting routines.

Three classes live here:

- :class:`FitResult`, the bare numerical outcome of a fit — free parameters,
  best-fit point, likelihood, samples — and its ``fit_results.json`` schema.
  This is what the fitting routines return and write.
- :class:`FitResultGroup`, the outcome of fitting each coefficient on its own:
  one :class:`FitResult` per coefficient, each a genuine single-parameter fit
  with its own likelihood and evidence. It writes those results to their
  subdirectories and aggregates them into the summary ``fit_results.json``.
- :class:`Fit`, a fit that already exists on disk: the result it produced —
  a :class:`FitResult`, or a :class:`FitResultGroup` when it was run one
  coefficient at a time — held as an attribute, plus the metadata describing
  how it was run. It is built only by loading, never by a fitting routine, and
  it is what downstream consumers (plotting, reports) work with.

Anything new describing how a fit was run belongs on :class:`Fit`, which is
meant to grow, and as a property derived from the runcard it holds rather than
as a field of its own — see its docstring. The result classes stay the minimal
numerical record and own the ``fit_results.json`` schemas on their own.
"""

import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Mapping, Optional, Union

import jax.numpy as jnp
import pandas as pd
import yaml
from rich import box
from rich.console import Console
from rich.table import Table

from smefit.priors import _build_dist
from smefit.whitening import WhitenTransform

log = logging.getLogger(__name__)


def _load_json(path) -> Dict:
    """Decode a JSON file, naming it when it will not parse.

    A decoder locates a fault *within* a file but never names the file itself,
    and a fit is read from two of them — so "line 1 column 2" on its own says
    nothing about which one to go and look at. Every reader here goes through
    this, so that reading a fit fails the same way wherever it is read from.

    A missing file raises :class:`FileNotFoundError` untouched: that is not a
    malformed fit, it is an absent one.
    """
    path = pathlib.Path(path)
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"'{path}' is not valid JSON: {e}") from e


def _load_yaml(path) -> Dict:
    """Decode a YAML file, naming it when it will not parse.

    The counterpart of :func:`_load_json`, and it fails the same way. An empty
    file reads as an empty mapping: a runcard that sets nothing is still a
    runcard, and every :class:`Fit` property falls back to its default.
    """
    path = pathlib.Path(path)
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"'{path}' is not valid YAML: {e}") from e


# Fit actions are named "run_<fit type>_fit", or "run_individual_<fit
# type>_fits" when the fit is run one coefficient at a time.
_FIT_ACTION_RE = re.compile(
    r"^run_(?P<individual>individual_)?(?P<fit_type>.+?)_fits?$"
)


def _parse_fit_action(action: str) -> Optional[Dict]:
    """What a fit action's name says about the fit, ``None`` for anything else.

    Fit actions are named to a convention that spells out both which sampler
    they run and whether they drive it one coefficient at a time, so the name
    of the action a runcard ran is what says how a fit was produced::

        run_analytic_fit              ->  {"fit_type": "analytic", "is_individual_fit": False}
        run_individual_blackjax_fits  ->  {"fit_type": "blackjax", "is_individual_fit": True}
        report                        ->  None

    The keys name the :class:`Fit` properties they answer, so that what is
    being read stays legible at every hop between here and them, with ``is_``
    marking the one that is a flag.

    A name that matches the convention is looked up in
    :mod:`smefit.fit_actions` as well, so that an action which merely reads
    like a fit is not taken for one.
    """
    from smefit import fit_actions  # imported here: fit_actions imports this module

    match = _FIT_ACTION_RE.match(action)
    if match is None or not callable(getattr(fit_actions, action, None)):
        return None
    return {
        "fit_type": match["fit_type"],
        "is_individual_fit": match["individual"] is not None,
    }


def _runcard_actions(config: Mapping) -> List[str]:
    """The names of the actions a runcard's ``actions_`` list runs.

    Entries come in three shapes, and this reduces them to bare names::

        actions_:
          - run_analytic_fit      # a plain action    ->  "run_analytic_fit"
          - report(main=True)     # with arguments    ->  "report"
          - scan:                 # a namespace, i.e. a mapping of actions
              - plot_chi2         #                   ->  skipped

    Arguments are dropped: they configure an action, they do not change which
    one ran. A namespace entry is a mapping rather than an action itself, so it
    is stepped over and not descended into — an action nested under one is not
    reported here.
    """
    actions = config.get("actions_") or []
    names = []
    for action in actions:
        # nested-namespace entries are mappings, not actions themselves
        if not isinstance(action, str):
            continue
        names.append(action.split("(", 1)[0].strip())
    return names


def _runcard_fit_action(config: Mapping) -> Optional[Dict]:
    """What the fit action a runcard ran says about the fit.

    The first fit action of the list is the one that produced the fit: a
    runcard runs a single fit, whatever else it lists alongside it. Everything
    known about how a fit was produced is read from that one name, through
    :func:`_parse_fit_action`.

    None when the runcard runs no fit action at all — a report-only runcard,
    say. Saying so is left to the caller, so that it is said once, when the fit
    is loaded, rather than at every lookup: see :meth:`Fit.from_folder`.
    """
    for action in _runcard_actions(config):
        parsed = _parse_fit_action(action)
        if parsed is not None:
            return parsed
    return None


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
    def correlations(self) -> pd.DataFrame:
        """Pearson correlations of the posterior samples, as a square frame.

        Free parameters only, in :attr:`free_parameters` order: a derived
        coefficient is a function of them, so its correlations carry nothing the
        free ones do not already.

        A parameter whose samples never moved has no correlation with anything,
        and gets a row and a column of ``nan`` rather than an error — a fit that
        left one parameter unexplored is still worth looking at.
        """
        if self.samples is None:
            raise ValueError(
                "This fit stored no posterior samples, so it has no "
                "correlations to report."
            )
        posterior = jnp.stack([self.samples[name] for name in self.free_parameters])
        return pd.DataFrame(
            # A single free parameter correlates only with itself, and corrcoef
            # returns that as a scalar rather than as the 1x1 matrix it is.
            jnp.atleast_2d(jnp.corrcoef(posterior)),
            index=self.free_parameters,
            columns=self.free_parameters,
        )

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
        return cls.from_payload(_load_json(pathlib.Path(path) / "fit_results.json"))

    @classmethod
    def from_payload(cls, d: Mapping) -> "FitResult":
        """Rebuild a FitResult from a decoded ``fit_results.json`` payload.

        Split out of :meth:`from_json` so that a reader which has already
        inspected the payload — :meth:`Fit.from_folder` — can hand it straight
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


class FitResultGroup:
    """A collection of FitResult objects from individual parameter fits."""

    def __init__(self, results: List[FitResult]):
        self.results = results

    # ------------------------------------------------------------------
    # The FitResult fields that still mean something one at a time
    # ------------------------------------------------------------------
    #
    # A group is not a FitResult and deliberately does not pretend to be one:
    # it has no joint likelihood, no single best-fit point, no correlations.
    # These two fields are the ones a coefficient answers on its own, and they
    # are exposed under their FitResult names so that a consumer looking at one
    # coefficient at a time — the 1D bounds routines — reads both kinds of fit
    # the same way. Anything reading two coefficients *together* (a contour, a
    # correlation) must not: individual posteriors were sampled independently,
    # so pairing them up would draw a correlation that was never fitted.

    @property
    def free_parameters(self) -> List[str]:
        """The coefficients fitted, in the order they were fitted.

        One per individual fit, each free in its own.
        """
        return [result.free_parameters[0] for result in self.results]

    @property
    def samples(self) -> Optional[Dict[str, jnp.ndarray]]:
        """Posterior samples per coefficient, from its own individual fit.

        ``None`` when no individual fit kept any, as for :class:`FitResult`.
        """
        samples = {}
        for result in self.results:
            name = result.free_parameters[0]
            if result.samples is not None and name in result.samples:
                samples[name] = result.samples[name]
        return samples or None

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

    @classmethod
    def from_payload(cls, d: Mapping) -> "FitResultGroup":
        """Rebuild the group a summary payload was aggregated from.

        The counterpart of :meth:`write_summary`, and the reader of the schema
        that method owns — so the two stay side by side, as
        :meth:`FitResult.write` and :meth:`FitResult.from_payload` do.

        :meth:`write_summary` records per coefficient everything a
        :class:`FitResult` needs, so each coefficient's slice of the summary is
        itself a single-fit payload and is read back through
        :meth:`FitResult.from_payload` — the one place that knows that schema.
        """
        chi2 = d.get("chi2") or {}
        logz = d.get("logz") or {}
        samples = d.get("samples") or {}
        prior_specs = d.get("prior_specs") or {}

        results = []
        for name in d["free_parameters"]:
            spec = prior_specs.get(name)
            results.append(
                FitResult.from_payload(
                    {
                        "free_parameters": [name],
                        "best_fit_point": {name: d["best_fit_point"][name]},
                        # the likelihood of this coefficient's own fit
                        "max_loglikelihood": -0.5 * float(chi2[name]),
                        "num_data": d["num_data"],
                        "logz": logz.get(name),
                        "samples": {name: samples[name]} if name in samples else None,
                        "prior_specs": {name: spec} if spec is not None else None,
                        "whitening_active": d.get("whitening_active", False),
                    }
                )
            )
        return cls(results)


@dataclass
class Fit:
    """A fit that exists on disk: what it produced, and how it was run.

    A ``Fit`` is built only by loading — :meth:`from_folder` — never by a fitting
    routine, which returns the bare :class:`FitResult` it computed and knows
    nothing of the directory it will be written to. This is the class to extend
    when a fit needs to carry something new about itself; ``FitResult`` stays
    the numerical record.

    How the fit was configured is held as the whole runcard it was run with
    (:attr:`fit_runcard`), never copied out key by key into fields of its own.
    Everything derived from it — :attr:`use_quad`, :attr:`fit_type`, … — is a
    property computed on demand, so a consumer that needs something new about a
    fit gets a new property here and no signature anywhere changes. Reach for
    :meth:`setting` for a runcard key that has no property yet.

    None of this is serialised: a fit's numbers live in ``fit_results.json``,
    while how it was configured is read back from ``input/runcard.yaml`` in the
    fit directory, the authoritative record of that.

    Attributes
    ----------
    fit_results : FitResult or FitResultGroup
        The numbers the fit produced — best-fit point, likelihood, samples. A
        :class:`FitResultGroup` when the fit was run one coefficient at a time
        (:attr:`individual_fit`): there is no joint likelihood to hold then,
        only the single-parameter result of every coefficient.
    fit_name : str
        Identity of the fit: the name of the directory it was loaded from,
        which is what users refer to it by.
    label : str or None
        Legend label, from the ``label`` key of the fit's ``fits`` entry. It is
        used verbatim, so it may be raw LaTeX; when unset, consumers fall back
        to ``fit_name``. How the fit is presented in a given plot, not a
        property of the fit itself, so it comes from the runcard that loads the
        fit rather than from the fit directory.
    fit_runcard : dict
        The runcard the fit was run with, read from ``input/runcard.yaml``.
        :meth:`from_folder` requires it: a fit whose configuration is unknown
        cannot be loaded. It defaults to empty only for a ``Fit`` built by
        hand, in which case every property below falls back to its default.
    """

    fit_results: Union[FitResult, FitResultGroup]
    fit_name: str
    label: Optional[str] = None
    fit_runcard: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        """The fit's name — not cosmetic, it ends up in file names.

        reportengine names the output of a provider run once per fit after the
        namespace element it was run for, and it does so by calling ``str`` on
        it (``spec_to_nice_name``, ``reportengine/formattingtools.py``). Without
        this, the dataclass repr is used, is over the length reportengine
        tolerates, and every per-fit figure and table falls back to being named
        ``fits0``, ``fits1``, … instead of after the fit it belongs to.
        """
        return self.fit_name

    @property
    def plot_label(self) -> str:
        """How to name this fit inside a plot.

        The runcard's :attr:`label` where it gave one, and the fit's name
        otherwise, so a plot always says which fit it is drawn from. Distinct
        from :meth:`__str__`, which names *files* and so has to stay the plain
        fit name: a label is presentation and may be raw LaTeX.
        """
        return self.label or self.fit_name

    # ------------------------------------------------------------------
    # How the fit was configured — derived from the runcard
    # ------------------------------------------------------------------

    def setting(self, key: str, default: Any = None) -> Any:
        """The value of a runcard key, or *default* when it is not set.

        The escape hatch for a setting that has no property of its own (yet):
        it keeps consumers from reaching into :attr:`fit_runcard` directly, so
        that giving the key a property later changes nothing for them.
        """
        return self.fit_runcard.get(key, default)

    @property
    def use_quad(self) -> bool:
        """Whether the fit included quadratic EFT corrections.

        Downstream consumers need it after the run: it is what says whether the
        posterior can be expected to be Gaussian.
        """
        return bool(self.setting("use_quad", False))

    @property
    def fit_type(self) -> Optional[str]:
        """Which routine produced the fit.

        ``analytic``, ``hessian``, ``ultranest`` or ``blackjax``, from the fit
        action the runcard ran. None when it runs no fit action at all.
        """
        return self._fit_action["fit_type"]

    @property
    def individual_fit(self) -> bool:
        """True for the summary of one-at-a-time individual fits.

        That is, when the runcard ran a ``run_individual_*_fits`` action — the
        only thing that says so. Every coefficient was then fitted with the
        others held at their baseline, so the samples are independent 1D
        posteriors rather than a joint one — see :class:`FitResultGroup`.
        """
        return self._fit_action["is_individual_fit"]

    @cached_property
    def _fit_action(self) -> Dict:
        """What the fit action the runcard ran says about this fit.

        Keyed by the property that answers it, so that a property is one
        lookup. Cached: :attr:`fit_type` and :attr:`individual_fit` are two
        halves of the same answer. A runcard that ran no fit action leaves both
        at their default; :meth:`from_folder` is what reports that.
        """
        return _runcard_fit_action(self.fit_runcard) or {
            "fit_type": None,
            "is_individual_fit": False,
        }

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_folder(cls, path, label: Optional[str] = None) -> "Fit":
        """Load a Fit from a fit directory.

        The whole directory is read, not just one file of it: the numbers come
        from ``fit_results.json`` and how the fit was run from
        ``input/runcard.yaml``.

        Both payloads written by this module are accepted, and the runcard
        action says which one to expect: a ``run_individual_*_fits`` action
        wrote the summary of :meth:`FitResultGroup.write_summary`, which is
        read back as the :class:`FitResultGroup` it was aggregated from so that
        every coefficient keeps its own chi2 and evidence; any other fit action
        wrote the standard payload of :meth:`FitResult.write`.

        ``label`` is how the caller chooses to present the fit; the directory
        knows nothing about it, so it is the one piece of metadata passed in
        rather than read back.

        Raises
        ------
        FileNotFoundError
            If either file is missing. Both are written by every smefit run, so
            a directory without them is not a fit: either the run never
            finished, or this is not a fit directory at all. Loading it half
            way — numbers without the runcard that says how they were produced
            — would only push the failure to whichever consumer needs the
            metadata.
        ValueError
            If either file is present but cannot be read as a fit: unparsable
            JSON or YAML, or a runcard and a payload that disagree about
            whether the fit was run one coefficient at a time. Reading a fit
            fails the same way whichever of its files is at fault; naming that
            file is left to :func:`_load_json` and :func:`_load_yaml`.
        """
        path = pathlib.Path(path)

        # The numbers the fit produced, and — from the runcard, the
        # authoritative record of it and the only source — how it was
        # configured. Each is decoded once and dispatched on below.
        fit_results_payload = _load_json(path / "fit_results.json")
        fit_runcard = _load_yaml(path / "input" / "runcard.yaml")

        # Whether a fit was run one coefficient at a time is something about
        # how it was run, so the action it ran is what says so — never the
        # shape of the payload, which is only a consequence of it.
        fit_action = _runcard_fit_action(fit_runcard)
        if fit_action is None:
            log.warning(
                "The runcard of '%s' runs no known fit action: its fit type is "
                "left unset, and it is read as a joint fit.",
                path.name,
            )
        ran_individually = fit_action is not None and fit_action["is_individual_fit"]

        # The action decides how the payload is read; this only turns a
        # directory whose two files disagree into a clear error rather than a
        # confusing one from the reader that is picked.
        payload_is_summary = isinstance(fit_results_payload.get("chi2"), dict)
        if payload_is_summary != ran_individually:
            wrote, holds = (
                ("an individual fit", "a single joint chi2")
                if ran_individually
                else ("a joint fit", "a chi2 per coefficient")
            )
            raise ValueError(
                f"'{path}' is inconsistent: its runcard ran {wrote}, but its "
                f"fit_results.json holds {holds}."
            )

        # Each class reads the schema it writes; the action picks which.
        reader = FitResultGroup if ran_individually else FitResult

        return cls(
            fit_results=reader.from_payload(fit_results_payload),
            # The directory name is the identity users refer to in runcards.
            fit_name=path.name,
            label=label,
            fit_runcard=fit_runcard,
        )
