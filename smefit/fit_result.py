"""
smefit.fit_result.py

FitResult dataclass shared across fitting routines.
"""

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
