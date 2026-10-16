"""
smefit.utils_actions.py

Reportengine utility actions for smefit.
"""

import pathlib
import shutil

import numpy as np
import yaml


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
    """Print the principal-component analysis and save it."""
    pca.print_summary()
    pca.write(output_path)


def write_rge_matrix(rge_matrix, output_path):
    """Compute the RGE matrix and save it, without running a fit.

    The write itself happens in ``produce_rge_matrix``, which saves the matrix
    under ``output_path`` whenever it builds it, so a fit writes it too.

    The resulting ``<output_path>/rge_matrix.pkl`` can be passed to a later
    runcard through ``rge.rg_matrix``.
    """
    if rge_matrix is None:
        raise ValueError(
            "write_rge_matrix needs an 'rge:' block in the runcard; without one "
            "there is no RGE matrix to compute."
        )
    out_file = pathlib.Path(output_path) / rge_matrix.FILENAME
    if not out_file.exists():
        rge_matrix.write(output_path)
