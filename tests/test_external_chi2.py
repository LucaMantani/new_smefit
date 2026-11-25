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


def test_group_is_not_forwarded(tmp_path, coeff_group):
    """`group` is for report aggregation: a class not taking it still loads."""
    module = tmp_path / "kwarg_chi2_grouped.py"
    module.write_text(FAKE_CHI2_WITH_KWARG)

    external_chi2 = {"KwargChi2": {"path": str(module), "scale": 2.0, "group": "G"}}
    result = load_external_chi2(external_chi2, coeff_group, {})

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


@pytest.mark.parametrize(
    "rg_matrix", ["my_fit", f"smefit_results/fits/my_fit/{RGEMatrix.FILENAME}"]
)
def test_prefix_relative_rg_matrix_kwarg_loads(tmp_path, coeff_group, rg_matrix):
    """`rg_matrix` is forwarded verbatim, so read_cache is what resolves it —
    a fit name included.

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
            "rg_matrix": rg_matrix,
        }
    }
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_results": str(tmp_path)},
    ):
        result = load_external_chi2(external_chi2, coeff_group, settings)

    assert result[0].num_data == 2


# ---------------------------------------------------------------------------
# Load failures: always a ValueError naming the plugin to avoid silent drops
#
# ---------------------------------------------------------------------------


def test_module_not_found(tmp_path, coeff_group):
    external_chi2 = {"FakeChi2": {"path": str(tmp_path / "nonexistent.py")}}
    with pytest.raises(ValueError, match="FakeChi2") as info:
        load_external_chi2(external_chi2, coeff_group, {})
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_module_raising_on_import(tmp_path, coeff_group):
    module = tmp_path / "raises_on_import.py"
    module.write_text("CONFIG = {}\nSCALE = CONFIG['scale']\n")
    with pytest.raises(ValueError, match="FakeChi2") as info:
        load_external_chi2({"FakeChi2": {"path": str(module)}}, coeff_group, {})
    assert isinstance(info.value.__cause__, KeyError)


def test_entry_without_path(coeff_group):
    with pytest.raises(ValueError, match="FakeChi2.*'path'"):
        load_external_chi2({"FakeChi2": {}}, coeff_group, {})


def test_module_without_the_class(tmp_path, coeff_group):
    module = tmp_path / "no_such_class.py"
    module.write_text(FAKE_CHI2_CLASS)
    with pytest.raises(ValueError, match="does not define .* Missing") as info:
        load_external_chi2({"Missing": {"path": str(module)}}, coeff_group, {})
    assert isinstance(info.value.__cause__, AttributeError)


@pytest.mark.parametrize(
    "stem, body",
    [
        ("missing_option", "self.scale = settings['scale']"),
        ("missing_rge", "self.scale = rge_dict['init_scale']"),  # no rge: block
    ],
)
def test_constructor_key_error(tmp_path, coeff_group, stem, body):
    """The two cases of issue #174: neither may leave the loader a KeyError."""
    module = tmp_path / f"ctor_key_error_{stem}.py"
    module.write_text(
        "class BrokenChi2:\n"
        "    num_data = 2\n"
        "    def __init__(self, coefficients, rge_dict=None, settings=None):\n"
        "        settings = settings or {}\n"
        "        rge_dict = rge_dict or {}\n"
        f"        {body}\n"
        "    def compute_chi2(self, c):\n"
        "        return 0.0\n"
    )
    with pytest.raises(ValueError, match="BrokenChi2") as info:
        load_external_chi2({"BrokenChi2": {"path": str(module)}}, coeff_group, None)
    assert not isinstance(info.value, KeyError)
    assert isinstance(info.value.__cause__, KeyError)


_INTERFACE_TEMPLATE = """\
import numpy as np

class {name}:
    num_data = {num_data}

    def __init__(self, coefficients, rge_dict, **kwargs):
        pass

{compute}
"""
_COMPUTE = "    def compute_chi2(self, c):\n        return 0.0\n"


@pytest.mark.parametrize(
    "name, num_data, compute, match",
    [
        ("NoCompute", "2", "", "callable compute_chi2"),
        ("AttrCompute", "2", "    compute_chi2 = 1.0\n", "callable compute_chi2"),
        ("NoNumData", "None", _COMPUTE, "num_data to an integer"),
        ("FloatNumData", "2.0", _COMPUTE, "num_data to an integer"),
        ("BoolNumData", "True", _COMPUTE, "num_data to an integer"),
    ],
)
def test_interface_is_checked(tmp_path, coeff_group, name, num_data, compute, match):
    module = tmp_path / f"{name.lower()}.py"
    module.write_text(
        _INTERFACE_TEMPLATE.format(name=name, num_data=num_data, compute=compute)
    )
    with pytest.raises(ValueError, match=f"{name}.*{match}"):
        load_external_chi2({name: {"path": str(module)}}, coeff_group, {})


def test_numpy_integer_num_data_is_accepted(tmp_path, coeff_group):
    """A module counting its points with an array's size returns a numpy int."""
    module = tmp_path / "numpy_num_data.py"
    module.write_text(
        _INTERFACE_TEMPLATE.format(
            name="NumpyNumData", num_data="np.int64(4)", compute=_COMPUTE
        )
    )
    (ext,) = load_external_chi2(
        {"NumpyNumData": {"path": str(module)}}, coeff_group, {}
    )
    assert ext.num_data == 4


# ---------------------------------------------------------------------------
# max_scale: resolved per entry against the global cut, forwarded when set
# ---------------------------------------------------------------------------


# num_data reports the cut the constructor received, 0 for none
FAKE_CHI2_WITH_CUT = """\
class {name}:
    def __init__(self, coefficients, rge_dict, max_scale=None):
        self.num_data = 0 if max_scale is None else int(max_scale)

    def compute_chi2(self, c):
        return 0.0
"""


def _cut_module(tmp_path, name):
    module = tmp_path / f"{name.lower()}.py"
    module.write_text(FAKE_CHI2_WITH_CUT.format(name=name))
    return str(module)


def test_global_max_scale_is_forwarded(tmp_path, coeff_group):
    external_chi2 = {"CutGlobal": {"path": _cut_module(tmp_path, "CutGlobal")}}
    result = load_external_chi2(external_chi2, coeff_group, None, max_scale=700.0)
    assert result[0].num_data == 700


def test_entry_max_scale_overrides_global(tmp_path, coeff_group):
    path = _cut_module(tmp_path, "CutOwn")
    external_chi2 = {"CutOwn": {"path": path, "max_scale": 300.0}}
    result = load_external_chi2(external_chi2, coeff_group, None, max_scale=700.0)
    assert result[0].num_data == 300


def test_entry_max_scale_without_global(tmp_path, coeff_group):
    path = _cut_module(tmp_path, "CutOnlyOwn")
    external_chi2 = {"CutOnlyOwn": {"path": path, "max_scale": 300.0}}
    result = load_external_chi2(external_chi2, coeff_group, None)
    assert result[0].num_data == 300


def test_null_max_scale_exempts_an_entry(tmp_path, coeff_group):
    """Exempt, the module gets no max_scale at all: one taking none loads."""
    module = tmp_path / "kwarg_chi2_exempt.py"
    module.write_text(FAKE_CHI2_WITH_KWARG.replace("KwargChi2", "KwargChi2Exempt"))
    external_chi2 = {"KwargChi2Exempt": {"path": str(module), "max_scale": None}}
    result = load_external_chi2(external_chi2, coeff_group, None, max_scale=700.0)
    assert result[0].num_data == 3


def test_cut_on_a_module_taking_no_max_scale_raises(tmp_path, coeff_group):
    from reportengine.configparser import ConfigError

    module = tmp_path / "kwarg_chi2_uncuttable.py"
    module.write_text(FAKE_CHI2_WITH_KWARG.replace("KwargChi2", "KwargChi2Uncut"))
    external_chi2 = {"KwargChi2Uncut": {"path": str(module)}}
    with pytest.raises(ConfigError, match="takes no max_scale argument"):
        load_external_chi2(external_chi2, coeff_group, None, max_scale=700.0)


def test_parse_external_chi2_rejects_a_non_positive_max_scale(tmp_path):
    from reportengine.configparser import ConfigError

    from smefit.config import smefitConfig

    cfg = smefitConfig.__new__(smefitConfig)
    with pytest.raises(ConfigError, match="must be positive"):
        cfg.parse_external_chi2({"X": {"path": str(tmp_path), "max_scale": -1}})


def test_drell_yan_module_applies_the_cut(coeff_group):
    """A shipped module cuts its own bins: 17 of CMS_DYMee_13TeV's 79 are at
    or below 1000 GeV."""
    from pathlib import Path

    path = (
        Path(__file__).parents[1] / "external_chi2" / "drell_yan" / "CMS_DYMee_13TeV.py"
    )
    entry = {"path": str(path), "use_quad": False, "order": "LO"}
    uncut = load_external_chi2({"CMS_DYMee_13TeV": entry}, coeff_group, None)
    cut = load_external_chi2(
        {"CMS_DYMee_13TeV": entry}, coeff_group, None, max_scale=1000.0
    )
    assert uncut[0].num_data == 79
    assert cut[0].num_data == 17
