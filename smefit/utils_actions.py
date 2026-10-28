"""
smefit.utils_actions.py

Reportengine utility actions for smefit.
"""

import pathlib
import shutil

import numpy as np
import yaml

from smefit.dataset_info import _compile_preview, _preview_tex


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


def write_dataset_table(
    dataset_table_tex, dataset_bibliography, output_path, compile_pdf=False
):
    """Write the LaTeX table of the runcard's datasets, for a paper's appendix.

    ``dataset_table.tex`` holds the table. With ``fetch_bibtex: true`` the
    table cites its references with INSPIRE keys, and ``dataset_table.bib``
    holds their BibTeX. Otherwise — the default, or INSPIRE unreachable — no
    ``.bib`` is written, and the table links to its references and compiles on
    its own.

    With ``compile_pdf: true``, ``dataset_table_preview.tex`` typesets the table
    and its bibliography as a document of their own, compiled into
    ``dataset_table_preview.pdf`` when pdflatex is installed. No auxiliary file
    is left behind, and a failed compilation only warns.
    """
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "dataset_table.tex").write_text(dataset_table_tex, encoding="utf-8")
    with_bibliography = bool(dataset_bibliography.bibtex)
    if with_bibliography:
        (output_path / "dataset_table.bib").write_text(
            dataset_bibliography.bibtex, encoding="utf-8"
        )
    if compile_pdf:
        (output_path / "dataset_table_preview.tex").write_text(
            _preview_tex(with_bibliography), encoding="utf-8"
        )
        _compile_preview(output_path, with_bibliography)
