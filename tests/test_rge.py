"""Tests for smefit/rge/.

Fast unit tests run without real Wilson evolution.
Slow tests (marked @pytest.mark.slow) call the real wilson package.
"""

import pytest
import wilson

from smefit.rge import RGE, RGEMatrix
from smefit.rge.loading import _resolve_scales
from smefit.rge.runner import _wilson_params, evolve_gs

# ---------------------------------------------------------------------------
# Unit tests — no I/O, no real wilson evolution
# ---------------------------------------------------------------------------


def test_rge_init_invalid_yukawa():
    with pytest.raises(ValueError, match="Yukawa parameter not supported"):
        RGE(["OpBox"], init_scale=1000, yukawa="invalid")


def test_rge_init_valid_yukawas():
    for yukawa in ("top", "none", "full"):
        rge = RGE(["OpBox"], init_scale=1000, yukawa=yukawa)
        assert rge.yukawa == yukawa


def test_evolve_gs_at_mz_returns_gs():
    """evolve_gs(mz) should equal gs exactly (log(mz/mz) = 0)."""
    from smefit.constants import gs, mz

    result = evolve_gs(mz)
    assert abs(result - gs) < 1e-10


def test_evolve_gs_decreases_above_mz():
    """The strong coupling decreases at higher scales."""
    from smefit.constants import mz

    gs_mz = evolve_gs(mz)
    gs_high = evolve_gs(10 * mz)
    assert gs_high < gs_mz


def test_wilson_params_top_zeroes_vus():
    """Inside _wilson_params('top'), Vus should be 0; restored afterwards."""
    original = wilson.run.smeft.smpar.p.get("Vus")
    with _wilson_params("top", adm_QCD=False):
        inside = wilson.run.smeft.smpar.p.get("Vus")
    restored = wilson.run.smeft.smpar.p.get("Vus")

    assert inside == pytest.approx(0.0)
    assert restored == pytest.approx(original)


def test_wilson_params_restored_after_exception():
    """Parameters are restored even when an exception is raised inside the context."""
    original = wilson.run.smeft.smpar.p.get("Vus")
    with pytest.raises(RuntimeError):
        with _wilson_params("top", adm_QCD=False):
            raise RuntimeError("boom")
    assert wilson.run.smeft.smpar.p.get("Vus") == pytest.approx(original)


def test_resolve_scales_fixed(theory_a):
    """A fixed obs_scale returns a single-element list regardless of n_data."""
    from smefit.core import TheoryGroup

    tg = TheoryGroup([theory_a])
    scales = _resolve_scales({"init_scale": 1000, "obs_scale": 91.2}, tg)
    assert scales == [91.2]


def test_resolve_scales_dynamic(theory_a):
    """Dynamic obs_scale returns one scale per data point."""
    from smefit.core import TheoryGroup

    tg = TheoryGroup([theory_a])
    scales = _resolve_scales({"init_scale": 1000, "obs_scale": "dynamic"}, tg)
    assert len(scales) == 3  # theory_a has 3 data points
    assert all(s == pytest.approx(100.0) for s in scales)


def test_resolve_scales_dynamic_default(theory_a):
    """obs_scale defaults to 'dynamic' when absent."""
    from smefit.core import TheoryGroup

    tg = TheoryGroup([theory_a])
    scales = _resolve_scales({"init_scale": 1000}, tg)
    assert len(scales) == 3


# ---------------------------------------------------------------------------
# Slow tests — call real wilson RGE evolution
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rge_matrix_returns_dataframe():
    """RGEmatrix should return a pandas DataFrame with the requested operator as column."""
    import pandas as pd

    rge = RGE(["OpBox"], init_scale=1000, yukawa="top")
    result = rge.RGEmatrix(200)
    assert isinstance(result, pd.DataFrame)
    assert "OpBox" in result.columns


@pytest.mark.slow
def test_rge_matrix_finite_values():
    """All non-zero entries in RGEmatrix should be finite."""
    import jax.numpy as jnp

    rge = RGE(["OpBox"], init_scale=1000, yukawa="top")
    result = rge.RGEmatrix(200)
    assert all(jnp.isfinite(v) for v in result.values.flatten())


@pytest.mark.slow
def test_load_rge_matrix_shape(theory_a):
    """load_rge_matrix should return an RGEMatrix with a 3-D stacked array."""
    from smefit.core import TheoryGroup
    from smefit.rge import load_rge_matrix

    tg = TheoryGroup([theory_a])
    rge_dict = {"init_scale": 1000, "obs_scale": "dynamic"}
    result = load_rge_matrix(rge_dict, ["OpBox"], tg)

    assert isinstance(result, RGEMatrix)
    assert result.stacked_mats.ndim == 3
    assert result.init_operators == ["OpBox"]
    # dynamic: 3 data points (all scale=100), one unique scale → shape (1, n_obs_ops, 1)
    # The actual number of unique scales may collapse to 1 since all are 100.0
    n_unique_scales = result.stacked_mats.shape[0]
    assert n_unique_scales >= 1
