"""
smefit.fit_result.py

FitResult dataclass shared across fitting routines.
"""

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import jax.numpy as jnp
from rich import box
from rich.console import Console
from rich.table import Table


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
    """

    free_parameters: List[str]
    best_fit_point: Dict[str, float]
    max_loglikelihood: float
    num_data: int
    logz: Optional[float] = None
    samples: Optional[Dict[str, jnp.ndarray]] = None

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

        for name, val in self.best_fit_point.items():
            u = unc.get(name, float("nan"))
            kind = "free" if name in self.free_parameters else "derived"
            table.add_row(name, f"{val:.6f}", f"{u:.6f}", kind)

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
        }

        out_file = output_path / "fit_results.json"
        with out_file.open("w") as f:
            json.dump(payload, f, indent=2)


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
        table.add_column("Coefficient", style="cyan", no_wrap=True)
        table.add_column("Best fit", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("chi2", justify="right")
        table.add_column("chi2/dof", justify="right")

        for result in self.results:
            name = result.free_parameters[0]
            val = result.best_fit_point.get(name, float("nan"))
            std = result.std.get(name, float("nan"))
            table.add_row(
                name,
                f"{val:.6f}",
                f"{std:.6f}",
                f"{result.chi2_val:.4f}",
                f"{result.chi2_ndof:.4f}",
            )

        console.print(table)
        console.rule(style="dim")

    def write_results(self, output_path) -> None:
        """Write each FitResult to its own subdirectory."""
        base = pathlib.Path(output_path) / "individual_fits"
        for result in self.results:
            name = result.free_parameters[0]
            result.write(base / name)
