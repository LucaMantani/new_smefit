"""Unit tests for smefit.loader — load_dataset and load_theory."""

import json

import jax.numpy as jnp
import pytest
import yaml

from smefit.loader import load_dataset, load_theory

# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_basic(fixtures_data_path):
    ds = load_dataset(fixtures_data_path, {"name": "TESTDATA"})
    assert ds.name == "TESTDATA"
    assert ds.num_data == 3
    assert jnp.allclose(ds.central_values, jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(ds.stat_err, jnp.array([0.1, 0.2, 0.3]))


def test_load_dataset_scalar_luminosity(fixtures_data_path):
    ds = load_dataset(fixtures_data_path, {"name": "TESTDATA"})
    assert ds.luminosity.shape == (3,)
    assert jnp.allclose(ds.luminosity, jnp.full(3, 1.5))


def test_load_dataset_list_luminosity(tmp_path):
    data = {
        "dataset_name": "TMP",
        "num_data": 3,
        "data_central": [1.0, 2.0, 3.0],
        "statistical_error": [0.1, 0.2, 0.3],
        "systematics": [[0.05, 0.10, 0.15]],
        "sys_names": ["UNCORR"],
        "sys_type": ["ADD"],
        "luminosity": [10.0, 20.0, 30.0],
    }
    (tmp_path / "TMP.yaml").write_text(yaml.dump(data))
    ds = load_dataset(tmp_path, {"name": "TMP"})
    assert jnp.allclose(ds.luminosity, jnp.array([10.0, 20.0, 30.0]))


def test_load_dataset_no_luminosity(tmp_path):
    data = {
        "dataset_name": "TMP",
        "num_data": 2,
        "data_central": [1.0, 2.0],
        "statistical_error": [0.1, 0.2],
        "systematics": [[0.05, 0.10]],
        "sys_names": ["UNCORR"],
        "sys_type": ["ADD"],
    }
    (tmp_path / "TMP.yaml").write_text(yaml.dump(data))
    ds = load_dataset(tmp_path, {"name": "TMP"})
    assert jnp.all(jnp.isnan(ds.luminosity))


def test_load_dataset_length_mismatch(tmp_path):
    data = {
        "dataset_name": "TMP",
        "num_data": 3,
        "data_central": [1.0, 2.0],  # only 2 values, but num_data=3
        "statistical_error": [0.1, 0.2, 0.3],
        "systematics": [[0.05, 0.10, 0.15]],
        "sys_names": ["UNCORR"],
        "sys_type": ["ADD"],
    }
    (tmp_path / "TMP.yaml").write_text(yaml.dump(data))
    with pytest.raises(ValueError):
        load_dataset(tmp_path, {"name": "TMP"})


def test_load_dataset_not_found(fixtures_data_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(fixtures_data_path, {"name": "NONEXISTENT"})


def test_load_dataset_zero_stat_unc(fixtures_data_path):
    ds = load_dataset(fixtures_data_path, {"name": "TESTDATA", "stat_unc": "zero"})
    assert jnp.allclose(ds.stat_err, jnp.zeros(3))


def test_load_dataset_zero_syst_unc(fixtures_data_path):
    ds = load_dataset(fixtures_data_path, {"name": "TESTDATA", "syst_unc": "zero"})
    assert jnp.allclose(ds.syst_err, jnp.zeros_like(ds.syst_err))


def test_load_dataset_invalid_unc_type(fixtures_data_path):
    with pytest.raises(ValueError):
        load_dataset(
            fixtures_data_path, {"name": "TESTDATA", "stat_unc": "nonexistent"}
        )
    with pytest.raises(ValueError):
        load_dataset(
            fixtures_data_path, {"name": "TESTDATA", "syst_unc": "nonexistent"}
        )


# ---------------------------------------------------------------------------
# load_theory
# ---------------------------------------------------------------------------


def test_load_theory_basic(fixtures_theory_path):
    ds = {"name": "TESTDATA", "order": "NLO"}
    th = load_theory(fixtures_theory_path, ds)
    assert th.name == "TESTDATA"
    assert jnp.allclose(th.sm_pred, jnp.array([10.0, 20.0, 30.0]))
    # operators must be sorted
    assert th.operators == sorted(th.operators)


def test_load_theory_operators_from_quadratic(fixtures_theory_path):
    # TESTDATA.json has OpA*OpB in NLO; both must appear in operators
    ds = {"name": "TESTDATA", "order": "NLO"}
    th = load_theory(fixtures_theory_path, ds)
    assert "OpA" in th.operators
    assert "OpB" in th.operators


def test_load_theory_missing_cov_type(fixtures_theory_path):
    with pytest.raises(ValueError):
        load_theory(
            fixtures_theory_path,
            {"name": "TESTDATA", "order": "NLO", "theory_cov": "nonexistent"},
        )


def test_load_theory_zero_cov_type(fixtures_theory_path):
    # "zero" opts out of a theory covmat, even though "theory_cov_zero" is not in the JSON
    th = load_theory(
        fixtures_theory_path, {"name": "TESTDATA", "order": "NLO", "theory_cov": "zero"}
    )
    assert th.sm_covmat.shape == (3, 3)
    assert jnp.allclose(th.sm_covmat, jnp.zeros((3, 3)))


def test_load_theory_not_found(fixtures_theory_path):
    with pytest.raises(FileNotFoundError):
        load_theory(fixtures_theory_path, {"name": "NONEXISTENT", "order": "NLO"})
