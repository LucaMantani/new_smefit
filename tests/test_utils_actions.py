"""Unit tests for smefit.utils_actions — the side-effecting report actions."""

import json

import numpy as np
import pandas as pd

from smefit.pca import pca
from smefit.utils_actions import run_pca

SETTINGS = {"threshold": 1.0e-3, "min_weight": 0.01}


def _pca_of(diagonal, names):
    fisher = pd.DataFrame(np.diag(diagonal), index=names, columns=names)
    return pca(fisher, SETTINGS)


def test_run_pca_writes_pca_json(tmp_path):
    """The action creates its output directory and serialises the decomposition."""
    result = _pca_of([4.0, 1.0], ["OpA", "OpB"])

    run_pca(result, tmp_path / "out")

    payload = json.loads((tmp_path / "out" / "pca.json").read_text())
    assert payload["component_names"] == ["PC1", "PC2"]
    assert payload["eigenvalues"] == [4.0, 1.0]
    assert payload["flat"] == [False, False]


def test_run_pca_writes_null_for_an_unconstrained_direction(tmp_path):
    """Infinite widths are not valid JSON, so they go out as null."""
    result = _pca_of([4.0, 0.0], ["OpA", "OpB"])

    run_pca(result, tmp_path / "out")

    payload = json.loads((tmp_path / "out" / "pca.json").read_text())
    assert payload["constraints"][1] is None
    assert payload["flat"] == [False, True]
