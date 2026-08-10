"""Unit tests for smefit.external_chi2."""

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from smefit.chi2 import Chi2
from smefit.external_chi2 import load_external_chi2
from smefit.rge import RGEMatrix

FAKE_CHI2_CLASS = """\
class FakeChi2:
    num_data = 5

    def __init__(self, coefficients, rge_dict, **kwargs):
        self.coefficients = coefficients
        self.rge_dict = rge_dict

    def compute_chi2(self, c):
        return 0.0
"""


# ---------------------------------------------------------------------------
# Successful import
# ---------------------------------------------------------------------------


def test_load_single_external_chi2(tmp_path, coeff_group):
    fake_module = tmp_path / "fake_chi2.py"
    fake_module.write_text(FAKE_CHI2_CLASS)

    external_chi2 = {"FakeChi2": {"path": str(fake_module)}}
    result = load_external_chi2(external_chi2, coeff_group, {})

    assert len(result) == 1
    assert isinstance(result[0], Chi2)
    assert result[0].num_data == 5


def test_load_external_chi2_param_names(tmp_path, coeff_group):
    fake_module = tmp_path / "fake_chi2b.py"
    fake_module.write_text(FAKE_CHI2_CLASS.replace("FakeChi2", "FakeChi2b"))

    external_chi2 = {"FakeChi2b": {"path": str(fake_module)}}
    result = load_external_chi2(external_chi2, coeff_group, {})

    assert result[0].param_names == coeff_group.free_names


# ---------------------------------------------------------------------------
# Multiple entries
# ---------------------------------------------------------------------------


FAKE_CHI2_CLASS_B = """\
class SecondChi2:
    num_data = 10

    def __init__(self, coefficients, rge_dict, **kwargs):
        pass

    def compute_chi2(self, c):
        return 1.0
"""


def test_load_multiple_external_chi2(tmp_path, coeff_group):
    module_a = tmp_path / "chi2_a.py"
    module_a.write_text(FAKE_CHI2_CLASS)
    module_b = tmp_path / "chi2_b.py"
    module_b.write_text(FAKE_CHI2_CLASS_B)

    external_chi2 = {
        "FakeChi2": {"path": str(module_a)},
        "SecondChi2": {"path": str(module_b)},
    }
    result = load_external_chi2(external_chi2, coeff_group, {})

    assert len(result) == 2
    num_data_values = {r.num_data for r in result}
    assert num_data_values == {5, 10}


# ---------------------------------------------------------------------------
# Extra kwargs are forwarded
# ---------------------------------------------------------------------------


FAKE_CHI2_WITH_KWARG = """\
class KwargChi2:
    num_data = 3

    def __init__(self, coefficients, rge_dict, scale=1.0):
        self.scale = scale

    def compute_chi2(self, c):
        return self.scale
"""


def test_extra_kwargs_forwarded(tmp_path, coeff_group):
    module = tmp_path / "kwarg_chi2.py"
    module.write_text(FAKE_CHI2_WITH_KWARG)

    external_chi2 = {"KwargChi2": {"path": str(module), "scale": 42.0}}
    result = load_external_chi2(external_chi2, coeff_group, {})

    assert len(result) == 1
    assert result[0].num_data == 3


# ---------------------------------------------------------------------------
# A forwarded kwarg holding a prefix-relative path
# ---------------------------------------------------------------------------


FAKE_CHI2_WITH_RGE_MATRIX = """\
from smefit.rge import RGEMatrix


class RGEChi2:
    num_data = 2

    def __init__(self, coefficients, rge_dict, rg_matrix=None):
        self.cache = RGEMatrix.read_cache(rg_matrix, rge_dict)

    def compute_chi2(self, c):
        return 0.0
"""


def test_prefix_relative_rg_matrix_kwarg_loads(tmp_path, coeff_group):
    """`rg_matrix` is forwarded verbatim, so read_cache is what resolves it.

    load_external_chi2 has no way to know which of the extra kwargs are paths;
    prefix support for this one lives in RGEMatrix.read_cache.
    """
    settings = {"init_scale": 1000.0, "yukawa": "top"}
    RGEMatrix(
        stacked_mats=jnp.array([[[1.0]]]),
        obs_operators=["Op1"],
        init_operators=["OpA"],
        scales=[100.0],
        settings=settings,
    ).write(tmp_path / "fits" / "my_fit")

    module = tmp_path / "rge_chi2.py"
    module.write_text(FAKE_CHI2_WITH_RGE_MATRIX)

    external_chi2 = {
        "RGEChi2": {
            "path": str(module),
            "rg_matrix": f"smefit_results/fits/my_fit/{RGEMatrix.FILENAME}",
        }
    }
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_results": str(tmp_path)},
    ):
        result = load_external_chi2(external_chi2, coeff_group, settings)

    assert result[0].num_data == 2


# ---------------------------------------------------------------------------
# ModuleNotFoundError for bad path
# ---------------------------------------------------------------------------


def test_module_not_found(tmp_path, coeff_group):
    external_chi2 = {"FakeChi2": {"path": str(tmp_path / "nonexistent.py")}}
    with pytest.raises(ModuleNotFoundError):
        load_external_chi2(external_chi2, coeff_group, {})
