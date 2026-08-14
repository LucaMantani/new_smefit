"""
smefit.pca.py

Principal component analysis of the Fisher information matrix.

The Fisher matrix ``F = 0.5 * d2chi2/dc2`` says how sharply the likelihood
responds to each direction in coefficient space. Its eigenvectors are the
principal directions — the linear combinations of Wilson coefficients the data
actually speak about — and the eigenvalues say how loudly: a large ``lambda_i``
is a well-constrained direction, ``lambda_i ~ 0`` a flat one, along which the
posterior is set by the prior rather than by the data. The width the data allow
along a direction is ``sigma_i = 1 / sqrt(lambda_i)``.
"""

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

log = logging.getLogger(__name__)


def _fix_signs(vectors):
    """Make the largest-magnitude entry of each column positive.

    An eigenvector is defined only up to an overall sign, and LAPACK's choice is
    not stable under small changes of the input. Without a convention, rerunning
    the same runcard can flip a column and with it every number in the table and
    every colour in the heatmap.
    """
    n_cols = vectors.shape[1]
    pivot = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivot, np.arange(n_cols)])
    signs[signs == 0.0] = 1.0
    return vectors * signs


def _warn_on_degeneracy(eigenvalues, flat_mask):
    """Warn when two constrained directions share an eigenvalue.

    Eigenvectors of a degenerate eigenvalue are not unique — only the subspace
    they span is — so the individual directions reported for such a pair are an
    arbitrary basis of it. Flat directions are excluded: they are all ~0 and so
    all mutually degenerate by construction, which is the thing the flat flag
    already says.
    """
    keep = ~flat_mask
    if keep.sum() < 2:
        return
    constrained = eigenvalues[keep]
    gaps = np.abs(np.diff(constrained))
    close = np.flatnonzero(gaps < 1e-6 * constrained[0])
    if close.size:
        pairs = ", ".join(f"PC{i + 1}/PC{i + 2}" for i in close)
        log.warning(
            "PCA: near-degenerate eigenvalues (%s). Only the subspace each pair "
            "spans is determined; the individual directions reported for them "
            "are an arbitrary basis of it.",
            pairs,
        )


@dataclass
class PCA:
    """Eigendecomposition of a Fisher information matrix.

    Attributes
    ----------
    eigenvalues : np.ndarray, shape (n_free,)
        Eigenvalues in descending order.
    eigenvectors : np.ndarray, shape (n_free, n_free)
        Eigenvectors as **columns**: ``eigenvectors[:, i]`` goes with
        ``eigenvalues[i]``, and its entries are the weights of each coefficient
        in that principal direction.
    coeff_names : list of str
        The free coefficients, in the order of the eigenvector rows.
    threshold : float
        Directions with ``lambda_i / lambda_max`` below this are called flat.
    min_weight : float
        Components smaller than this are left out of the human-readable
        description of a direction.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    coeff_names: List[str]
    threshold: float
    min_weight: float

    @property
    def n_components(self) -> int:
        return len(self.eigenvalues)

    @property
    def component_names(self) -> List[str]:
        return [f"PC{i + 1}" for i in range(self.n_components)]

    @property
    def constraints(self) -> np.ndarray:
        """``sigma_i = 1 / sqrt(lambda_i)``, the width the data allow along PC i.

        Infinite where an eigenvalue is not positive: an exactly flat direction,
        or — if the centre is not a minimum — a negatively curved one.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = 1.0 / np.sqrt(self.eigenvalues)
        return np.where(self.eigenvalues > 0, sigma, np.inf)

    @property
    def eigenvalue_ratios(self) -> np.ndarray:
        """``lambda_i / lambda_max``, the scale-free measure of constraint."""
        return self.eigenvalues / self.eigenvalues[0]

    @property
    def flat_mask(self) -> np.ndarray:
        """Boolean mask of the directions the data do not constrain."""
        return self.eigenvalue_ratios < self.threshold

    @property
    def n_flat(self) -> int:
        return int(self.flat_mask.sum())

    def as_frame(self) -> pd.DataFrame:
        """Eigenvectors as a DataFrame: rows = coefficients, columns = PCs."""
        return pd.DataFrame(
            self.eigenvectors, index=self.coeff_names, columns=self.component_names
        )

    def describe(self, i: int) -> str:
        """Render principal direction *i* as a linear combination.

        Terms are ordered by decreasing weight and those below ``min_weight``
        are dropped, so the string names what the direction is *about* rather
        than reproducing the column. The leading term is always kept, however
        small, so a direction is never described as nothing.
        """
        v = self.eigenvectors[:, i]
        order = np.argsort(np.abs(v))[::-1]
        kept = [j for j in order if abs(v[j]) >= self.min_weight] or [order[0]]

        terms = []
        for pos, j in enumerate(kept):
            sign = "-" if v[j] < 0 else ("+" if pos else "")
            terms.append(f"{sign} {abs(v[j]):.2f} {self.coeff_names[j]}".strip())
        return " ".join(terms)

    def to_dict(self) -> dict:
        """JSON-serialisable representation, used by :func:`run_pca`.

        Infinite widths are written as ``null``: an unconstrained direction has
        no width, and ``Infinity`` is not valid JSON for readers outside Python.
        """
        sigma = self.constraints
        return {
            "coeff_names": list(self.coeff_names),
            "component_names": self.component_names,
            "eigenvalues": self.eigenvalues.tolist(),
            "eigenvectors": self.eigenvectors.tolist(),
            "constraints": [None if not np.isfinite(s) else float(s) for s in sigma],
            "eigenvalue_ratios": self.eigenvalue_ratios.tolist(),
            "flat": self.flat_mask.tolist(),
            "combinations": [self.describe(i) for i in range(self.n_components)],
            "threshold": self.threshold,
            "min_weight": self.min_weight,
        }


def pca(total_fisher_information_matrix, pca_settings) -> PCA:
    """Diagonalise the total Fisher matrix.

    Parameters
    ----------
    total_fisher_information_matrix : pd.DataFrame
        Square Fisher matrix over the free coefficients, from the node of the
        same name in ``smefit.fisher``.
    pca_settings : dict
        Settings dict from ``parse_pca_settings``. Deliberately required, with
        no default of its own: the parser is the single place the defaults are
        written down, so a runcard running a PCA action has to carry the
        ``pca_settings:`` block, empty if it wants nothing but the defaults.

    Returns
    -------
    PCA
    """
    names = list(total_fisher_information_matrix.index)
    fisher = np.asarray(total_fisher_information_matrix.values, dtype=float)
    # The AD Hessian is symmetric up to rounding; eigh reads one triangle only,
    # so symmetrise rather than let the choice of triangle decide the answer.
    fisher = 0.5 * (fisher + fisher.T)

    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = _fix_signs(eigenvectors[:, order])

    if eigenvalues[0] <= 0:
        # Every ratio is measured against the largest eigenvalue, and there is
        # nothing to measure against if even that one has no curvature.
        raise ValueError(
            f"PCA: the Fisher matrix has no positive eigenvalue (largest is "
            f"{eigenvalues[0]:.3e}), so the likelihood has no constrained "
            f"direction at all. Check the datasets and coefficients entering "
            f"the fit."
        )

    result = PCA(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        coeff_names=names,
        threshold=pca_settings["threshold"],
        min_weight=pca_settings["min_weight"],
    )

    if eigenvalues[-1] < 0:
        log.warning(
            "PCA: smallest eigenvalue is %.3e < 0, so the point the Fisher "
            "matrix was evaluated at is a saddle rather than a minimum. The "
            "directions are still meaningful; the widths along the negative "
            "ones are not.",
            eigenvalues[-1],
        )
    _warn_on_degeneracy(eigenvalues, result.flat_mask)
    log.log(
        logging.WARNING if result.n_flat else logging.INFO,
        "PCA: eigenvalues in [%.3e, %.3e], %d/%d directions below "
        "threshold=%.1e (unconstrained by the data).",
        eigenvalues[-1],
        eigenvalues[0],
        result.n_flat,
        result.n_components,
        result.threshold,
    )
    return result


def run_pca(pca, output_path):
    """Print the principal-component spectrum and write ``pca.json``.

    Parameters
    ----------
    pca : PCA
        The decomposition, from the node of the same name.
    output_path : pathlib.Path
        Directory the run writes to.
    """
    console = Console()
    console.rule("[bold cyan]Principal Component Analysis[/bold cyan]")
    console.print(f"  [bold]n_free[/bold]    = {pca.n_components}")
    console.print(f"  [bold]threshold[/bold] = {pca.threshold:.1e}")
    console.print(
        f"  [bold]n_flat[/bold]    = "
        f"[{'red' if pca.n_flat else 'green'}]{pca.n_flat}[/]"
    )

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("PC", style="cyan", no_wrap=True)
    table.add_column("Eigenvalue", justify="right")
    table.add_column("Sigma", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Direction", justify="left")

    sigma = pca.constraints
    for i, name in enumerate(pca.component_names):
        flat = bool(pca.flat_mask[i])
        style = "dim red" if flat else None
        table.add_row(
            name,
            f"{pca.eigenvalues[i]:.4e}",
            "inf" if not np.isfinite(sigma[i]) else f"{sigma[i]:.4e}",
            f"{pca.eigenvalue_ratios[i]:.2e}",
            pca.describe(i),
            style=style,
        )

    console.print(table)
    if pca.n_flat:
        console.print(
            f"  [red]{pca.n_flat} direction(s) shown dimmed are flat[/red]: the "
            "data do not constrain them, so the prior sets their width."
        )
    console.rule(style="dim")

    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "pca.json").open("w") as f:
        json.dump(pca.to_dict(), f, indent=2)
