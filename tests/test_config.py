"""Unit tests for smefit/config.py."""

import logging
import os
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from reportengine.configparser import ConfigError
from reportengine.namespaces import NSList

from smefit.chi2 import Chi2
from smefit.config import smefitConfig
from smefit.core import Coefficient, CoefficientGroup, DataGroup, TheoryGroup
from smefit.model import EFTModel
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


def test_parse_data_path_prefix_resolved(cfg, tmp_path):
    (tmp_path / "smefit_database" / "commondata").mkdir(parents=True)
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_database": str(tmp_path / "smefit_database")},
    ):
        result = cfg.parse_data_path("smefit_database/commondata")
    assert result == tmp_path / "smefit_database" / "commondata"


def test_parse_data_path_prefix_missing_key(cfg):
    with patch("smefit.paths.load_user_paths", return_value={}):
        with pytest.raises(ValueError, match="smefit_setup_local"):
            cfg.parse_data_path("smefit_database/commondata")


def test_parse_theory_path_valid(cfg, tmp_path):
    result = cfg.parse_theory_path(str(tmp_path))
    assert result == tmp_path


def test_parse_theory_path_missing(cfg, tmp_path):
    missing = str(tmp_path / "nonexistent")
    with pytest.raises(ValueError, match="does not exist"):
        cfg.parse_theory_path(missing)


def test_parse_theory_path_prefix_resolved(cfg, tmp_path):
    (tmp_path / "smefit_database" / "theory").mkdir(parents=True)
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_database": str(tmp_path / "smefit_database")},
    ):
        result = cfg.parse_theory_path("smefit_database/theory")
    assert result == tmp_path / "smefit_database" / "theory"


def test_parse_theory_path_prefix_missing_key(cfg):
    with patch("smefit.paths.load_user_paths", return_value={}):
        with pytest.raises(ValueError, match="smefit_setup_local"):
            cfg.parse_theory_path("smefit_database/theory")


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
    """vars may not reference another expression coefficient."""
    raw = {
        "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1.0, "high": 1.0}},
        "OpB": {"free": False, "vars": ["OpA"], "expr": "OpA**2"},
        "OpC": {"free": False, "vars": ["OpB"], "expr": "OpB**2"},
    }
    with pytest.raises(
        ValueError, match="must be a free coefficient or a fixed coefficient"
    ):
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


def test_build_chi2_baseline_from_coefficients(cfg):
    """baseline flows from the coefficients node into the Chi2 (base-only path)."""
    prior = {"dist": "uniform", "low": -1.0, "high": 1.0}
    coefficients = CoefficientGroup(
        [
            Coefficient(name="OpA", free=True, prior=prior, baseline_value=0.7),
            Coefficient(name="OpB", free=True, prior=prior, baseline_value=-0.3),
        ]
    )
    mock_eft = MagicMock()
    mock_eft.coefficients.free_names = ["OpA", "OpB"]
    mock_data = MagicMock()
    mock_data.num_data = 5

    with patch("smefit.config.build_chi2", return_value=lambda c: jnp.sum(c**2)):
        result = cfg._build_chi2_impl(
            eft_model=mock_eft,
            data=mock_data,
            fit_covmat=jnp.eye(3),
            coefficients=coefficients,
        )
    assert jnp.allclose(result.baseline, jnp.array([0.7, -0.3]))


def test_build_chi2_ext_only_baseline_from_coefficients(cfg):
    """External-chi2-only fits still honor baseline_value (no eft_model)."""
    prior = {"dist": "uniform", "low": -1.0, "high": 1.0}
    coefficients = CoefficientGroup(
        [Coefficient(name="OpA", free=True, prior=prior, baseline_value=1.5)]
    )
    ext = Chi2(lambda c: jnp.sum(c**2), param_names=["OpA"], num_data=7)
    result = cfg._build_chi2_impl(ext_chi2_func=[ext], coefficients=coefficients)
    assert result.has_external
    assert jnp.allclose(result.baseline, jnp.array([1.5]))


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
# produce_datasets_chi2
# ---------------------------------------------------------------------------

_PRIOR = {"dist": "uniform", "low": -1.0, "high": 1.0}


def test_produce_datasets_chi2_returns_list(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    model = EFTModel(
        theory,
        CoefficientGroup([Coefficient("OpA", free=True, prior=_PRIOR)]),
        use_quad=False,
    )

    result = cfg.produce_datasets_chi2(
        eft_model=model, data=data, fit_covmat=jnp.eye(3)
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Chi2)
    assert result[0].name == "DS_A"
    assert result[0].num_data == 3


def test_produce_datasets_chi2_appends_external(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    model = EFTModel(
        theory,
        CoefficientGroup([Coefficient("OpA", free=True, prior=_PRIOR)]),
        use_quad=False,
    )
    ext = Chi2(lambda c: jnp.sum(c**2), ["OpA"], num_data=5, name="EXT")

    result = cfg.produce_datasets_chi2(
        eft_model=model, data=data, fit_covmat=jnp.eye(3), ext_chi2_func=[ext]
    )

    assert len(result) == 2
    assert result[0].name == "DS_A"
    assert result[1] is ext


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


# ---------------------------------------------------------------------------
# parse_chi2_scan_settings
# ---------------------------------------------------------------------------


def test_parse_chi2_scan_settings_defaults(cfg):
    result = cfg.parse_chi2_scan_settings({})
    assert result["n_points"] == 50


def test_parse_chi2_scan_settings_custom(cfg):
    result = cfg.parse_chi2_scan_settings({"n_points": 20})
    assert result["n_points"] == 20


def test_parse_chi2_scan_settings_unknown_key_warns(cfg, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        cfg.parse_chi2_scan_settings({"n_points": 10, "bad_key": "oops"})
    assert any("bad_key" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# parse_gradient_descent_settings
# ---------------------------------------------------------------------------


def test_parse_gradient_descent_settings_defaults(cfg):
    result = cfg.parse_gradient_descent_settings({})
    assert result["sm_solution"] is False
    assert result["n_steps"] == 2000
    assert result["tol"] == pytest.approx(1e-8)


def test_parse_gradient_descent_settings_custom(cfg):
    result = cfg.parse_gradient_descent_settings(
        {"sm_solution": True, "n_steps": 500, "tol": 1e-6}
    )
    assert result["sm_solution"] is True
    assert result["n_steps"] == 500
    assert result["tol"] == pytest.approx(1e-6)


def test_parse_gradient_descent_settings_unknown_key_warns(cfg, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        cfg.parse_gradient_descent_settings({"unknown_key": 99})
    assert any("unknown_key" in r.message for r in caplog.records)


# parse_hessian_settings
# ---------------------------------------------------------------------------


def test_parse_hessian_settings_defaults(cfg):
    result = cfg.parse_hessian_settings({})
    assert result["n_samples"] == 10000
    assert result["seed"] == 42


def test_parse_hessian_settings_custom(cfg):
    result = cfg.parse_hessian_settings({"n_samples": 500, "seed": 7})
    assert result["n_samples"] == 500
    assert result["seed"] == 7


def test_parse_hessian_settings_unknown_key_warns(cfg, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        cfg.parse_hessian_settings({"unknown_key": 99})
    assert any("unknown_key" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# parse_optimizer_settings / produce_optimizer
# ---------------------------------------------------------------------------


def test_parse_optimizer_settings_passthrough(cfg):
    settings = {"optimizer": "sgd", "optimizer_hyperparams": {"learning_rate": 0.1}}
    result = cfg.parse_optimizer_settings(settings)
    assert result["optimizer"] == "sgd"
    assert result["optimizer_hyperparams"]["learning_rate"] == pytest.approx(0.1)


def test_parse_optimizer_settings_unknown_key_warns(cfg, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        cfg.parse_optimizer_settings({"bad_key": 1})
    assert any("bad_key" in r.message for r in caplog.records)


def test_produce_optimizer_default(cfg):
    """No optimizer_settings → default Adam is returned."""
    opt = cfg.produce_optimizer()
    assert hasattr(opt, "init") and hasattr(opt, "update")


def test_produce_optimizer_with_settings(cfg):
    """Custom hyperparams are accepted without error."""
    settings = {"optimizer": "adam", "optimizer_hyperparams": {"learning_rate": 5e-3}}
    opt = cfg.produce_optimizer(settings)
    assert hasattr(opt, "init") and hasattr(opt, "update")


def test_produce_optimizer_with_clipnorm(cfg):
    """Clipnorm wraps the optimizer in a chain."""
    import optax

    settings = {
        "optimizer": "adam",
        "optimizer_hyperparams": {"learning_rate": 1e-2},
        "clipnorm": 1.0,
    }
    opt = cfg.produce_optimizer(settings)
    # optax.chain returns a GradientTransformationExtraArgs / named tuple;
    # the key check is that init/update exist and the object is not plain adam
    assert hasattr(opt, "init") and hasattr(opt, "update")
    # A chain wraps multiple transforms; verify by initialising on a dummy param
    import jax.numpy as jnp

    state = opt.init(jnp.zeros(2))
    assert state is not None


def test_produce_optimizer_with_scheduler(cfg):
    """A scheduler is injected as learning_rate without error."""
    settings = {
        "optimizer": "adam",
        "optimizer_hyperparams": {},
        "scheduler": {
            "name": "linear_schedule",
            "params": {
                "init_value": 1e-2,
                "end_value": 1e-4,
                "transition_steps": 1000,
            },
        },
    }
    opt = cfg.produce_optimizer(settings)
    assert hasattr(opt, "init") and hasattr(opt, "update")


# ---------------------------------------------------------------------------
# produce_data_groups
# ---------------------------------------------------------------------------


def test_produce_data_groups_no_group_keys_returns_none(cfg):
    datasets = [{"name": "DS_A", "order": "LO"}, {"name": "DS_B", "order": "NLO_QCD"}]
    assert cfg.produce_data_groups(datasets) is None


def test_produce_data_groups_all_grouped(cfg):
    datasets = [
        {"name": "DS_A", "order": "LO", "group": "G1"},
        {"name": "DS_B", "order": "LO", "group": "G2"},
        {"name": "DS_C", "order": "LO", "group": "G1"},
    ]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A", "DS_C"], "G2": ["DS_B"]}


def test_produce_data_groups_partial_grouping(cfg):
    """Datasets without a group key are simply omitted from the result."""
    datasets = [
        {"name": "DS_A", "order": "LO", "group": "G1"},
        {"name": "DS_B", "order": "LO"},
    ]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A"]}


def test_produce_data_groups_preserves_insertion_order(cfg):
    datasets = [
        {"name": "DS_C", "order": "LO", "group": "G3"},
        {"name": "DS_A", "order": "LO", "group": "G1"},
        {"name": "DS_B", "order": "LO", "group": "G2"},
    ]
    result = cfg.produce_data_groups(datasets)
    assert list(result.keys()) == ["G3", "G1", "G2"]


def test_produce_data_groups_merges_ext_chi2_groups(cfg):
    cfg._ext_chi2_groups = {"EXT_DS": "G2"}
    datasets = [{"name": "DS_A", "order": "LO", "group": "G1"}]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A"], "G2": ["EXT_DS"]}


def test_produce_data_groups_no_ext_chi2_attr(cfg):
    """Works correctly when _ext_chi2_groups was never set (no external_chi2 in runcard)."""
    datasets = [{"name": "DS_A", "order": "LO", "group": "G1"}]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A"]}


def test_produce_data_groups_ext_chi2_appended_to_existing_group(cfg):
    cfg._ext_chi2_groups = {"EXT_DS": "G1"}
    datasets = [{"name": "DS_A", "order": "LO", "group": "G1"}]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A", "EXT_DS"]}


# ---------------------------------------------------------------------------
# parse_external_chi2
# ---------------------------------------------------------------------------


def test_parse_external_chi2_strips_group(cfg):
    raw = {
        "MyExt": {"path": "/some/path.py", "group": "G1", "use_quad": True},
    }
    result = cfg.parse_external_chi2(raw)
    assert "group" not in result["MyExt"]
    assert result["MyExt"]["path"] == "/some/path.py"
    assert result["MyExt"]["use_quad"] is True


def test_parse_external_chi2_caches_groups(cfg):
    raw = {
        "ExtA": {"path": "/a.py", "group": "G1"},
        "ExtB": {"path": "/b.py", "group": "G2"},
    }
    cfg.parse_external_chi2(raw)
    assert cfg._ext_chi2_groups == {"ExtA": "G1", "ExtB": "G2"}


def test_parse_external_chi2_no_group_key(cfg):
    raw = {"MyExt": {"path": "/some/path.py", "use_quad": False}}
    result = cfg.parse_external_chi2(raw)
    assert result == {"MyExt": {"path": "/some/path.py", "use_quad": False}}
    assert cfg._ext_chi2_groups == {}


def test_parse_external_chi2_mixed_group_and_no_group(cfg):
    raw = {
        "ExtA": {"path": "/a.py", "group": "G1"},
        "ExtB": {"path": "/b.py"},
    }
    result = cfg.parse_external_chi2(raw)
    assert "group" not in result["ExtA"]
    assert result["ExtB"] == {"path": "/b.py"}
    assert cfg._ext_chi2_groups == {"ExtA": "G1"}


def test_parse_external_chi2_resolves_prefix_path(cfg):
    raw = {"MyExt": {"path": "new_smefit/external_chi2/foo.py"}}
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"new_smefit": "/home/user/smefit/new_smefit"},
    ):
        result = cfg.parse_external_chi2(raw)
    assert (
        result["MyExt"]["path"] == "/home/user/smefit/new_smefit/external_chi2/foo.py"
    )


# ---------------------------------------------------------------------------
# Mass-scan producers
# ---------------------------------------------------------------------------


def test_produce_individual_mass_scales_requires_single_free_coefficient(cfg):
    cg = CoefficientGroup(
        [
            Coefficient("OpA", free=True, prior=_PRIOR),
            Coefficient("OpB", free=True, prior=_PRIOR),
        ]
    )
    with pytest.raises(ConfigError) as exc_info:
        cfg.produce_individual_mass_scales(cg, {"n_points": 5})
    assert exc_info.value.bad_item == ["OpA", "OpB"]


def test_produce_individual_mass_scales_uses_uniform_prior(cfg):
    cg = CoefficientGroup(
        [
            Coefficient(
                "OpM", free=True, prior={"dist": "uniform", "low": -2.0, "high": 2.0}
            )
        ]
    )
    result = cfg.produce_individual_mass_scales(cg, {"n_points": 5})
    assert isinstance(result, NSList)
    assert len(result) == 5
    assert result[0] == pytest.approx(-2.0)
    assert result[-1] == pytest.approx(2.0)


def test_produce_individual_mass_scales_fallback_range_and_warns(cfg, caplog):
    cg = CoefficientGroup(
        [
            Coefficient(
                "OpM", free=True, prior={"dist": "normal", "loc": 0.0, "scale": 1.0}
            )
        ]
    )
    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        result = cfg.produce_individual_mass_scales(cg, {"n_points": 3})
    assert result[0] == pytest.approx(-1.0)
    assert result[-1] == pytest.approx(1.0)
    assert any("lacks a uniform prior" in r.message for r in caplog.records)


def test_produce_individual_mass_scales_default_n_points(cfg):
    cg = CoefficientGroup(
        [
            Coefficient(
                "OpM", free=True, prior={"dist": "uniform", "low": 0.0, "high": 1.0}
            )
        ]
    )
    result = cfg.produce_individual_mass_scales(cg, {})
    assert len(result) == 50


def test_produce_individual_mass_rge_matrix_overrides_init_scale(cfg, theory_a):
    cg = CoefficientGroup([Coefficient("OpA", free=True, prior=_PRIOR)])
    theory = TheoryGroup([theory_a])
    rge = {"init_scale": 10.0, "obs_scale": 1000.0}

    with patch("smefit.config.load_rge_matrix", return_value="RGE_MATRIX") as mock_load:
        result = cfg.produce_individual_mass_rge_matrix(rge, cg, theory, 42.0)

    assert result == "RGE_MATRIX"
    _, kwargs = mock_load.call_args
    assert kwargs["rge_dict"]["init_scale"] == 42.0
    assert kwargs["coeff_list"] == ["OpA"]
    assert kwargs["theory_group"] is theory
    assert kwargs["save_path"] is None
    # the caller's rge dict must not be mutated in place
    assert rge["init_scale"] == 10.0


def test_produce_individual_mass_eft_model_builds_model(cfg, theory_a):
    cg = CoefficientGroup([Coefficient("OpA", free=True, prior=_PRIOR)])
    theory = TheoryGroup([theory_a])

    result = cfg.produce_individual_mass_eft_model(
        theory, cg, individual_mass_rge_matrix=None, use_quad=True
    )

    assert isinstance(result, EFTModel)
    assert result.use_quad is True
    assert result.coefficients is cg


def test_produce_individual_mass_ext_chi2_func_overrides_init_scale_when_rge_set(cfg):
    cg = CoefficientGroup([Coefficient("OpA", free=True, prior=_PRIOR)])
    rge = {"init_scale": 10.0}

    with patch("smefit.config.load_external_chi2", return_value="EXT") as mock_load:
        result = cfg.produce_individual_mass_ext_chi2_func(
            cg, ["ext_mod"], 42.0, rge=rge
        )

    assert result == "EXT"
    _, kwargs = mock_load.call_args
    assert kwargs["rge_dict"]["init_scale"] == 42.0
    assert rge["init_scale"] == 10.0  # caller's dict must not be mutated


def test_produce_individual_mass_ext_chi2_func_no_rge(cfg):
    cg = CoefficientGroup([Coefficient("OpA", free=True, prior=_PRIOR)])

    with patch("smefit.config.load_external_chi2", return_value="EXT") as mock_load:
        result = cfg.produce_individual_mass_ext_chi2_func(
            cg, ["ext_mod"], 42.0, rge=None
        )

    assert result == "EXT"
    _, kwargs = mock_load.call_args
    assert kwargs["rge_dict"] is None


def test_produce_individual_mass_chi2_delegates_to_build_chi2_impl(cfg):
    mock_eft = MagicMock()
    mock_eft.coefficients.free_names = ["OpA"]
    mock_data = MagicMock()
    mock_data.num_data = 5

    with patch("smefit.config.build_chi2", return_value=lambda c: jnp.sum(c**2)):
        result = cfg.produce_individual_mass_chi2(
            individual_mass_eft_model=mock_eft, data=mock_data, fit_covmat=jnp.eye(3)
        )

    assert isinstance(result, Chi2)
    assert result.num_data == 5
    assert not result.has_external
