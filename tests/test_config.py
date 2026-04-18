"""Unit tests for smefit/config.py."""

import os
import pathlib
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from reportengine.configparser import ConfigError

from smefit.chi2 import Chi2
from smefit.config import smefitConfig
from smefit.core import CoefficientGroup, DataGroup, TheoryGroup
from smefit.priors import Prior

# ---------------------------------------------------------------------------
# Fixture: bare smefitConfig instance (no reportengine DAG)
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    """smefitConfig instance created via __new__, bypassing reportengine's __init__."""
    c = smefitConfig.__new__(smefitConfig)
    c.output_path = tmp_path
    return c


# ---------------------------------------------------------------------------
# parse_data_path / parse_theory_path
# ---------------------------------------------------------------------------


def test_parse_data_path_valid(cfg, tmp_path):
    result = cfg.parse_data_path(str(tmp_path))
    assert result == tmp_path


def test_parse_data_path_missing(cfg, tmp_path):
    missing = str(tmp_path / "nonexistent")
    with pytest.raises(ValueError, match="does not exist"):
        cfg.parse_data_path(missing)


def test_parse_theory_path_valid(cfg, tmp_path):
    result = cfg.parse_theory_path(str(tmp_path))
    assert result == tmp_path


def test_parse_theory_path_missing(cfg, tmp_path):
    missing = str(tmp_path / "nonexistent")
    with pytest.raises(ValueError, match="does not exist"):
        cfg.parse_theory_path(missing)


# ---------------------------------------------------------------------------
# parse_coefficients
# ---------------------------------------------------------------------------


def test_parse_coefficients_builds_group(cfg):
    raw = {
        "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1.0, "high": 1.0}},
        "OpB": {"free": False, "value": 2.0},
    }
    group = cfg.parse_coefficients(raw)
    assert isinstance(group, CoefficientGroup)
    assert "OpA" in group.free_names
    assert "OpB" not in group.free_names


def test_parse_coefficients_with_valid_expr(cfg):
    raw = {
        "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1.0, "high": 1.0}},
        "OpC": {"free": False, "vars": ["OpA"], "expr": "OpA**2"},
    }
    group = cfg.parse_coefficients(raw)
    assert "OpC" in group.names
    assert "OpC" not in group.free_names


def test_parse_coefficients_invalid_vars_raises(cfg):
    """vars must reference a free coefficient; referencing a fixed one should raise."""
    raw = {
        "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1.0, "high": 1.0}},
        "OpB": {"free": False, "value": 2.0},
        "OpC": {"free": False, "vars": ["OpB"], "expr": "OpB**2"},
    }
    with pytest.raises(ValueError, match="must be a free coefficient"):
        cfg.parse_coefficients(raw)


# ---------------------------------------------------------------------------
# parse_rge
# ---------------------------------------------------------------------------


def test_parse_rge_valid(cfg):
    rge = {"init_scale": 1000.0, "obs_scale": "dynamic"}
    result = cfg.parse_rge(rge)
    assert result == rge


def test_parse_rge_missing_init_scale_raises(cfg):
    with pytest.raises(ConfigError):
        cfg.parse_rge({"obs_scale": "dynamic"})


def test_parse_rge_invalid_obs_scale_raises(cfg):
    with pytest.raises(ConfigError):
        cfg.parse_rge({"init_scale": 1000.0, "obs_scale": "bad_value"})


def test_parse_rge_fixed_obs_scale_ok(cfg):
    rge = {"init_scale": 1000.0, "obs_scale": 91.2}
    result = cfg.parse_rge(rge)
    assert result["obs_scale"] == 91.2


# ---------------------------------------------------------------------------
# parse_ultranest_settings
# ---------------------------------------------------------------------------


def test_parse_ultranest_defaults(cfg, tmp_path):
    result = cfg.parse_ultranest_settings({}, tmp_path)
    assert result["ultranest_seed"] == 123456
    assert result["sampler_plot"] is False
    assert result["ReactiveNS_settings"]["resume"] == "overwrite"
    assert result["ReactiveNS_settings"]["vectorized"] is False
    assert "ultranest_logs" in result["ReactiveNS_settings"]["log_dir"]


def test_parse_ultranest_resume_false_becomes_overwrite(cfg, tmp_path):
    settings = {"ReactiveNS_settings": {"resume": False}}
    result = cfg.parse_ultranest_settings(settings, tmp_path)
    assert result["ReactiveNS_settings"]["resume"] == "overwrite"


def test_parse_ultranest_resume_true_existing_dir(cfg, tmp_path):
    log_dir = str(tmp_path / "ultranest_logs")
    os.makedirs(log_dir)
    settings = {"ReactiveNS_settings": {"log_dir": log_dir, "resume": True}}
    result = cfg.parse_ultranest_settings(settings, tmp_path)
    assert result["ReactiveNS_settings"]["resume"] is True


def test_parse_ultranest_resume_missing_dir_raises(cfg, tmp_path):
    missing_dir = str(tmp_path / "no_such_dir")
    settings = {"ReactiveNS_settings": {"log_dir": missing_dir, "resume": True}}
    with pytest.raises(FileNotFoundError):
        cfg.parse_ultranest_settings(settings, tmp_path)


# ---------------------------------------------------------------------------
# parse_blackjax_settings
# ---------------------------------------------------------------------------


def test_parse_blackjax_defaults(cfg, tmp_path):
    result = cfg.parse_blackjax_settings({}, tmp_path)
    assert result["n_live"] == 500
    assert result["n_posterior_samples"] == 1000
    assert result["log_precision"] == -2
    assert result["seed"] == 0
    assert "blackjax_logs" in result["log_dir"]


def test_parse_blackjax_custom_values(cfg, tmp_path):
    settings = {"n_live": 200, "seed": 42}
    result = cfg.parse_blackjax_settings(settings, tmp_path)
    assert result["n_live"] == 200
    assert result["seed"] == 42
    assert result["n_posterior_samples"] == 1000  # still default


# ---------------------------------------------------------------------------
# produce_fit_covmat
# ---------------------------------------------------------------------------


def test_produce_fit_covmat_exp(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    covmat = cfg.produce_fit_covmat(data, theory)
    assert covmat.shape == (3, 3)


def test_produce_fit_covmat_with_theory_covmat(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    covmat_no_th = cfg.produce_fit_covmat(data, theory, use_theory_covmat=False)
    # use a fresh cfg to bypass the cache
    cfg2 = smefitConfig.__new__(smefitConfig)
    covmat_with_th = cfg2.produce_fit_covmat(data, theory, use_theory_covmat=True)
    # adding sm_covmat makes entries larger
    assert jnp.all(covmat_with_th >= covmat_no_th)


def test_produce_fit_covmat_caching(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    result1 = cfg.produce_fit_covmat(data, theory)
    result2 = cfg.produce_fit_covmat(data, theory)
    assert result1 is result2


def test_produce_fit_covmat_name_mismatch_raises(cfg, dataset_a, theory_b):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_b])
    with pytest.raises(ValueError, match="different datasets"):
        cfg.produce_fit_covmat(data, theory)


# ---------------------------------------------------------------------------
# _build_chi2_impl
# ---------------------------------------------------------------------------


def test_build_chi2_no_data_no_ext_raises(cfg):
    with pytest.raises(ConfigError):
        cfg._build_chi2_impl()


def test_build_chi2_base_only(cfg):
    mock_eft = MagicMock()
    mock_eft.coefficients.free_names = ["OpA"]
    mock_data = MagicMock()
    mock_data.num_data = 5

    with patch("smefit.config.build_chi2", return_value=lambda c: jnp.sum(c**2)):
        result = cfg._build_chi2_impl(
            eft_model=mock_eft, data=mock_data, fit_covmat=jnp.eye(3)
        )
    assert isinstance(result, Chi2)
    assert result.num_data == 5
    assert not result.has_external


def test_build_chi2_ext_only(cfg):
    ext = Chi2(lambda c: jnp.sum(c**2), param_names=["OpA"], num_data=7)
    result = cfg._build_chi2_impl(ext_chi2_func=[ext])
    assert isinstance(result, Chi2)
    assert result.num_data == 7
    assert result.has_external


def test_build_chi2_combined(cfg):
    mock_eft = MagicMock()
    mock_eft.coefficients.free_names = ["OpA"]
    mock_data = MagicMock()
    mock_data.num_data = 5
    ext = Chi2(lambda c: jnp.sum(c**2), param_names=["OpA"], num_data=7)

    with patch("smefit.config.build_chi2", return_value=lambda c: jnp.sum(c**2)):
        result = cfg._build_chi2_impl(
            eft_model=mock_eft,
            data=mock_data,
            fit_covmat=jnp.eye(3),
            ext_chi2_func=[ext],
        )
    assert result.num_data == 12  # 5 + 7
    assert result.has_external


# ---------------------------------------------------------------------------
# produce_prior
# ---------------------------------------------------------------------------


def test_produce_prior_no_whitening(cfg, coeff_group):
    result = cfg.produce_prior(coeff_group)
    assert isinstance(result, Prior)
    assert result.param_names == ["OpA"]


def test_produce_prior_with_whitening(cfg, coeff_group):
    whitening = {"sigma_prior": 3.0, "eps": 1e-8}
    result = cfg.produce_prior(coeff_group, whitening=whitening)
    assert isinstance(result, Prior)
    # Each distribution should be uniform on [-3, 3]
    from smefit.priors import _UniformDist

    assert isinstance(result.dists[0], _UniformDist)
    assert result.dists[0].low == pytest.approx(-3.0)
    assert result.dists[0].high == pytest.approx(3.0)
