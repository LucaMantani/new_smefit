"""
smefit.utils_actions.py

Reportengine utility actions for smefit.
"""

import json
import pathlib
import shutil

import numpy as np
import yaml
from rich import box
from rich.console import Console
from rich.table import Table


def write_pseudodata(pseudodata, theory_path, output_path):
    """Write pseudodata DataGroup to YAML files under output_path/pseudodata/.

    For datasets with a '_proj' suffix, the corresponding theory JSON is also
    copied into output_path/pseudodata_theory/ so the projected datasets can be
    used directly in a subsequent fit runcard.
    """
    out_dir = pathlib.Path(output_path) / "pseudodata"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in pseudodata.datasets:
        syst = np.asarray(ds.syst_err)  # (n_sys, n_data)
        lumi = np.asarray(ds.luminosity)

        data_dict = {
            "dataset_name": ds.name,
            "num_data": ds.num_data,
            "data_central": np.asarray(ds.central_values).tolist(),
            "statistical_error": np.asarray(ds.stat_err).tolist(),
            "systematics": (
                syst.tolist() if ds.num_data > 1 else syst.flatten().tolist()
            ),
            "sys_names": list(ds.sys_names),
            "sys_type": list(ds.sys_types),
        }

        if not np.isnan(lumi).all():
            data_dict["luminosity"] = lumi.tolist() if len(lumi) > 1 else float(lumi[0])

        with open(out_dir / f"{ds.name}.yaml", "w") as f:
            yaml.dump(data_dict, f, sort_keys=False)

        if ds.name.endswith("_proj"):
            original_name = ds.name.removesuffix("_proj")
            src = pathlib.Path(theory_path) / f"{original_name}.json"
            if src.exists():
                th_dir = pathlib.Path(output_path) / "pseudodata_theory"
                th_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, th_dir / f"{ds.name}.json")


def run_pca(pca, output_path):
    """Print the principal-component spectrum and write ``pca.json``.

    Parameters
    ----------
    pca : smefit.pca.PCA
        The decomposition, from the ``pca`` node.
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
