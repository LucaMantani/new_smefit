"""
smefit.fit_result.py

FitResult dataclass shared across fitting routines.
"""

import json
import logging
import pathlib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional

import jax.numpy as jnp
from rich import box
from rich.console import Console
from rich.table import Table

from smefit.priors import _build_dist
from smefit.whitening import WhitenTransform

if TYPE_CHECKING:
    # Import-time only: smefit.rge monkey-patches wilson and ckmutil when it is
    # imported, and fit_result is loaded by every fit action.
    from smefit.rge import RGEMatrix

log = logging.getLogger(__name__)


def _format_prior(spec: Optional[Mapping]) -> str:
    if spec is None:
        return "-"
    if spec.get("dist") == "exact_posterior":
        return f"ExactPosterior"
    return str(_build_dist(spec))


@dataclass
class FitResult:
    """Container for the result of a fit.

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
    rge_matrix : RGEMatrix or None
        The RGE matrix the fit ran with, when the runcard had an `rge:` block.
        This is a *companion artefact*, not part of `fit_results.json`: `write`
        pickles it separately to `rge_matrix.pkl` so a later runcard can reuse
        it via `rge.rg_matrix`. It is deliberately not restored by `from_json` —
        the pickle keys frames by unique scale, so the per-data-point stacking
        in `stacked_mats` cannot be reconstructed from it.
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
    rge_matrix: Optional["RGEMatrix"] = field(default=None, repr=False, compare=False)

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
        """Serialise this result to JSON and write it to *output_path*.

        Also writes `rge_matrix.pkl` alongside it when the fit used RGE running.
        """
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

        if self.rge_matrix is not None:
            self.rge_matrix.write(output_path)

    @classmethod
    def from_json(cls, path) -> "FitResult":
        """Load a FitResult from a directory containing fit_results.json."""
        p = pathlib.Path(path) / "fit_results.json"
        with p.open() as f:
            d = json.load(f)
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
    """A collection of FitResult objects from individual parameter fits.

    All the results share one `rge_matrix` object, so the group takes it over
    from them and writes a single copy at the top of the output directory rather
    than an identical one per coefficient. They share it because
    `produce_rge_matrix` depends on the *global* `coefficients` node, not on
    `individual_coefficients` — every individual `EFTModel` is handed the same
    full matrix and slices out the columns it needs in `EFTModel._apply_rge`.
    That also means the matrix written here spans every fitted coefficient, not
    just one, which is what makes it reusable via `rge.rg_matrix`.

    If that ever stops holding, the extra matrices would be silently dropped —
    they all target the same filename — so it is checked rather than assumed.
    """

    def __init__(
        self, results: List[FitResult], rge_matrix: Optional["RGEMatrix"] = None
    ):
        self.results = results
        if rge_matrix is None:
            matrices = [r.rge_matrix for r in results if r.rge_matrix is not None]
            if len({id(m) for m in matrices}) > 1:
                log.warning(
                    "Individual fits carry %d different RGE matrices; only the "
                    "first is written to rge_matrix.pkl.",
                    len({id(m) for m in matrices}),
                )
            rge_matrix = matrices[0] if matrices else None
        self.rge_matrix = rge_matrix

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
            # Strip the shared matrix from the per-coefficient writes; the group
            # emits a single copy at the top level below.
            replace(result, rge_matrix=None).write(base / name)
        self.write_summary(output_path)
        if self.rge_matrix is not None:
            self.rge_matrix.write(output_path)

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
            "individual_fit": True,
        }

        out_file = output_path / "fit_results.json"
        with out_file.open("w") as f:
            json.dump(payload, f, indent=2)
