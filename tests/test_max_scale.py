"""The energy-scale cut: the global ``max_scale`` key and its per-dataset form."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from reportengine.configparser import ConfigError

from smefit.api import smefitAPI
from smefit.config import smefitConfig
from smefit.loader import load_dataset, load_theory

# TESTDATA's three points, given distinct scales so that a cut can split them.
_SCALES = [100.0, 200.0, 300.0]


@pytest.fixture
def cfg(tmp_path):
    c = smefitConfig.__new__(smefitConfig)
    c.output_path = tmp_path
    return c


@pytest.fixture
def database(tmp_path, fixtures_data_path, fixtures_theory_path):
    """A copy of the fixtures with TESTDATA at scales ``_SCALES``, and a second
    dataset OTHER identical to it, for per-dataset cuts."""
    data_path = tmp_path / "commondata"
    theory_path = tmp_path / "theory"
    data_path.mkdir()
    theory_path.mkdir()
    theory = json.loads((fixtures_theory_path / "TESTDATA.json").read_text())
    theory["scales"] = _SCALES
    commondata = yaml.safe_load((fixtures_data_path / "TESTDATA.yaml").read_text())
    for name in ("TESTDATA", "OTHER"):
        (theory_path / f"{name}.json").write_text(json.dumps(theory))
        (data_path / f"{name}.yaml").write_text(
            yaml.dump({**commondata, "dataset_name": name})
        )
    return data_path, theory_path


def _runcard(data_path: Path, theory_path: Path, **extra: Any) -> dict[str, Any]:
    datasets = extra.pop("datasets", [{"name": "TESTDATA", "order": "NLO"}])
    return {
        "data_path": str(data_path),
        "theory_path": str(theory_path),
        "use_quad": True,
        "datasets": datasets,
        "coefficients": {
            name: {"free": True, "prior": {"dist": "uniform", "low": -5, "high": 5}}
            for name in ("OpA", "OpB")
        },
        **extra,
    }


# ---------------------------------------------------------------------------
# Dataset.select / Theory.select
# ---------------------------------------------------------------------------


def test_dataset_select_cuts_every_per_point_array(fixtures_data_path):
    ds = load_dataset(fixtures_data_path, {"name": "TESTDATA"})
    cut = ds.select([True, False, True])
    assert cut.num_data == 2
    assert jnp.allclose(cut.central_values, jnp.array([1.0, 3.0]))
    assert jnp.allclose(cut.stat_err, jnp.array([0.1, 0.3]))
    assert cut.syst_err.shape == (2, 2)
    assert jnp.allclose(cut.syst_err[0], jnp.array([0.05, 0.15]))
    assert cut.luminosity.shape == (2,)
    assert cut.sys_names == ds.sys_names
    assert ds.num_data == 3  # the original is untouched


def test_theory_select_cuts_predictions_and_covmat(fixtures_theory_path):
    th = load_theory(fixtures_theory_path, {"name": "TESTDATA", "order": "NLO"})
    cut = th.select([False, True, True])
    assert cut.n_data == 2
    assert jnp.allclose(cut.sm_pred, jnp.array([20.0, 30.0]))
    assert jnp.allclose(cut.sm_covmat, jnp.diag(jnp.array([0.04, 0.09])))
    assert jnp.allclose(cut.eft_lin_pred, th.eft_lin_pred[1:])
    assert jnp.allclose(cut.eft_quad_pred, th.eft_quad_pred[1:])
    assert cut.scales.shape == (2,)
    assert cut.operators == th.operators


# ---------------------------------------------------------------------------
# parse_max_scale / produce_scale_masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1.0])
def test_parse_max_scale_rejects_non_positive_numbers(cfg, value):
    with pytest.raises(ConfigError, match="must be positive"):
        cfg.parse_max_scale(value)


def test_parse_max_scale_rejects_strings(cfg):
    with pytest.raises(ConfigError):
        cfg.parse_max_scale("3TeV")


def test_parse_max_scale_accepts_null(cfg):
    assert cfg.parse_max_scale(None) is None


def test_no_cut_reads_no_theory_file(cfg):
    """With no max_scale anywhere the mapping is empty, and needs no
    theory_path: a runcard reading only the commondata still resolves."""
    datasets = [{"name": "NOT_ON_DISK", "order": "NLO"}]
    assert cfg.produce_scale_masks(datasets) == {}


def test_global_cut(cfg, database):
    _, theory_path = database
    datasets = [{"name": "TESTDATA"}, {"name": "OTHER"}]
    masks = cfg.produce_scale_masks(datasets, theory_path, max_scale=250.0)
    assert set(masks) == {"TESTDATA", "OTHER"}
    np.testing.assert_array_equal(masks["TESTDATA"], [True, True, False])


def test_cut_keeps_a_point_at_the_scale(cfg, database):
    _, theory_path = database
    masks = cfg.produce_scale_masks([{"name": "TESTDATA"}], theory_path, 200.0)
    np.testing.assert_array_equal(masks["TESTDATA"], [True, True, False])


def test_dataset_cut_overrides_global(cfg, database):
    _, theory_path = database
    datasets = [{"name": "TESTDATA", "max_scale": 150}, {"name": "OTHER"}]
    masks = cfg.produce_scale_masks(datasets, theory_path, max_scale=250.0)
    np.testing.assert_array_equal(masks["TESTDATA"], [True, False, False])
    np.testing.assert_array_equal(masks["OTHER"], [True, True, False])


def test_dataset_cut_without_global(cfg, database):
    _, theory_path = database
    datasets = [{"name": "TESTDATA", "max_scale": 150}, {"name": "OTHER"}]
    masks = cfg.produce_scale_masks(datasets, theory_path)
    assert set(masks) == {"TESTDATA"}


def test_null_exempts_a_dataset_from_the_global_cut(cfg, database):
    _, theory_path = database
    datasets = [{"name": "TESTDATA", "max_scale": None}, {"name": "OTHER"}]
    masks = cfg.produce_scale_masks(datasets, theory_path, max_scale=150.0)
    assert set(masks) == {"OTHER"}


def test_invalid_dataset_cut_raises(database):
    """A dataset entry's max_scale goes through the same parser as the global."""
    data_path, theory_path = database
    datasets = [{"name": "TESTDATA", "order": "NLO", "max_scale": -3}]
    runcard = _runcard(data_path, theory_path, datasets=datasets)
    with pytest.raises(ConfigError, match="must be positive"):
        smefitAPI.data(**runcard)


def test_cut_emptying_a_dataset_raises(cfg, database):
    _, theory_path = database
    datasets = [{"name": "TESTDATA", "max_scale": 50.0}, {"name": "OTHER"}]
    with pytest.raises(ConfigError, match=r"every data point of: TESTDATA"):
        cfg.produce_scale_masks(datasets, theory_path)


def test_cut_requires_theory_path(cfg):
    with pytest.raises(ConfigError, match="requires theory_path"):
        cfg.produce_scale_masks([{"name": "TESTDATA"}], None, max_scale=100.0)


def test_data_and_theory_cached_per_cut(cfg, database):
    """A different cut is a different DataGroup/TheoryGroup, not the cached one."""
    data_path, theory_path = database
    datasets = [{"name": "TESTDATA"}]
    loose = cfg.produce_scale_masks(datasets, theory_path, max_scale=250.0)
    tight = cfg.produce_scale_masks(datasets, theory_path, max_scale=150.0)
    assert cfg.produce_data(datasets, data_path, loose).num_data == 2
    assert cfg.produce_data(datasets, data_path, tight).num_data == 1
    datasets = [{"name": "TESTDATA", "order": "NLO"}]
    assert cfg.produce_theory(datasets, theory_path, loose).n_data == 2
    assert cfg.produce_theory(datasets, theory_path, tight).n_data == 1


# ---------------------------------------------------------------------------
# Through reportengine
# ---------------------------------------------------------------------------


def test_cut_reaches_the_chi2(database):
    """The cut data and theory agree, and the chi2 counts only the kept points."""
    data_path, theory_path = database
    datasets = [
        {"name": "TESTDATA", "order": "NLO"},
        {"name": "OTHER", "order": "NLO", "max_scale": None},
    ]
    runcard = _runcard(data_path, theory_path, datasets=datasets, max_scale=150.0)

    data = smefitAPI.data(**runcard)
    theory = smefitAPI.theory(**runcard)
    assert data.ndata_list == [3, 1]  # OTHER, TESTDATA: sorted by name
    assert theory.n_data == data.num_data == 4
    assert smefitAPI.fit_covmat(**runcard).shape == (4, 4)
    assert smefitAPI.chi2(**runcard).num_data == 4


def test_no_cut_matches_the_uncut_fixtures(database):
    data_path, theory_path = database
    runcard = _runcard(data_path, theory_path)
    assert smefitAPI.chi2(**runcard).num_data == 3


def test_max_scale_is_a_fit_setup_key():
    """Under fitcontext the fit's cut, not the report runcard's, must apply."""
    from smefit.config import _FIT_SETUP_KEYS

    assert "max_scale" in _FIT_SETUP_KEYS
