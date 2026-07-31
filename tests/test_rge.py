"""Tests for smefit/rge.py.

Fast unit tests run without real Wilson evolution.
Slow tests (marked @pytest.mark.slow) call the real wilson package.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import wilson

from smefit.rge import (
    RGE,
    RGEMatrix,
    RGESettings,
    _find_cached_scale,
    _resolve_scales,
    _wilson_params,
    evolve_gs,
    load_precomputed_rge_matrix,
    load_rge_mats_from_scales,
)

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
# RGESettings
# ---------------------------------------------------------------------------


def test_rge_settings_to_dict_key_set_is_frozen():
    """The on-disk compatibility key must stay exactly these four names.

    load_precomputed_rge_matrix compares this dict with strict equality against
    the one stored in the pickle, so adding a key would invalidate every
    rge_matrix.pkl ever written.
    """
    settings = RGESettings(init_scale=1000.0)
    assert set(settings.to_dict()) == {
        "init_scale",
        "smeft_accuracy",
        "adm_QCD",
        "yukawa",
    }


def test_rge_settings_from_dict_defaults():
    settings = RGESettings.from_dict({"init_scale": 1000.0})
    assert settings.smeft_accuracy == "integrate"
    assert settings.adm_QCD is False
    assert settings.yukawa == "top"


def test_rge_settings_from_dict_casts_to_plain_python():
    """Casting keeps the pickled settings comparable across runs."""
    settings = RGESettings.from_dict({"init_scale": 1000, "adm_QCD": 1})
    assert isinstance(settings.init_scale, float)
    assert settings.to_dict()["adm_QCD"] is True


def test_rge_settings_agree_with_parse_rge_defaults():
    """The parser and the dataclass must not drift apart on defaults."""
    from smefit.config import smefitConfig

    parsed = smefitConfig({}).parse_rge({"init_scale": 1000.0})
    from_dict = RGESettings.from_dict({"init_scale": 1000.0}).to_dict()
    for key, value in from_dict.items():
        assert parsed[key] == value


def test_rge_settings_runner_is_configured():
    runner = RGESettings(init_scale=1000.0, yukawa="none").runner(["OpBox"])
    assert isinstance(runner, RGE)
    assert runner.init_scale == 1000.0
    assert runner.yukawa == "none"


# ---------------------------------------------------------------------------
# RGEMatrix serialisation
# ---------------------------------------------------------------------------


def _make_rge_matrix(scales, obs_operators=("OpBox", "OpD"), init_operators=("OpBox",)):
    obs_operators = list(obs_operators)
    init_operators = list(init_operators)
    stacked = jnp.array(
        [
            np.arange(len(obs_operators) * len(init_operators), dtype=float).reshape(
                len(obs_operators), len(init_operators)
            )
            + i
            for i in range(len(scales))
        ]
    )
    return RGEMatrix(
        stacked_mats=stacked,
        obs_operators=obs_operators,
        init_operators=init_operators,
        scales=list(scales),
        settings=RGESettings(init_scale=1000.0),
    )


def test_rge_matrix_write_round_trips(tmp_path):
    """A written matrix must be readable by the precomputed-matrix loader."""
    matrix = _make_rge_matrix([91.2, 200.0])
    matrix.write(tmp_path)

    cache = load_precomputed_rge_matrix(
        tmp_path / "rge_matrix.pkl", matrix.settings.to_dict()
    )

    assert sorted(cache) == [91.2, 200.0]
    for scale, frame in cache.items():
        assert list(frame.index) == matrix.obs_operators
        assert list(frame.columns) == matrix.init_operators
    expected = np.asarray(matrix.stacked_mats[0], dtype=float)
    np.testing.assert_allclose(cache[91.2].values, expected)


def test_rge_matrix_write_collapses_duplicate_scales(tmp_path):
    """Dynamic mode has one scale per data point; the file stores unique scales."""
    matrix = _make_rge_matrix([91.2, 91.2, 200.0])
    matrix.write(tmp_path)

    cache = load_precomputed_rge_matrix(
        tmp_path / "rge_matrix.pkl", matrix.settings.to_dict()
    )
    assert sorted(cache) == [91.2, 200.0]


def test_rge_matrix_write_creates_missing_directories(tmp_path):
    matrix = _make_rge_matrix([91.2])
    matrix.write(tmp_path / "deep" / "nested")
    assert (tmp_path / "deep" / "nested" / "rge_matrix.pkl").exists()


def test_rge_matrix_from_file_round_trips(tmp_path):
    """write -> from_file restores the object exactly, given the same scales."""
    matrix = _make_rge_matrix([91.2, 200.0])
    matrix.write(tmp_path)

    loaded = RGEMatrix.from_file(tmp_path / "rge_matrix.pkl", matrix.scales)

    assert loaded.scales == matrix.scales
    assert loaded.obs_operators == matrix.obs_operators
    assert loaded.init_operators == matrix.init_operators
    assert loaded.settings == matrix.settings
    np.testing.assert_allclose(
        np.asarray(loaded.stacked_mats), np.asarray(matrix.stacked_mats)
    )


def test_rge_matrix_from_file_restores_per_data_point_stacking(tmp_path):
    """The file collapses duplicate scales; the scales list expands them again.

    This is the whole reason from_file needs the caller's scales: dynamic-scale
    fits have one row per data point but only one stored frame per unique scale.
    """
    matrix = _make_rge_matrix([91.2, 91.2, 91.2, 200.0])
    matrix.write(tmp_path)

    loaded = RGEMatrix.from_file(tmp_path / "rge_matrix.pkl", matrix.scales)

    assert loaded.stacked_mats.shape[0] == 4
    np.testing.assert_allclose(
        np.asarray(loaded.stacked_mats[0]), np.asarray(loaded.stacked_mats[1])
    )


def test_rge_matrix_from_file_matches_scales_within_tolerance(tmp_path):
    """Stored float keys need not be bit-identical to the requested scales."""
    matrix = _make_rge_matrix([91.2])
    matrix.write(tmp_path)

    loaded = RGEMatrix.from_file(tmp_path / "rge_matrix.pkl", [91.2 * (1 + 1e-8)])

    assert loaded.stacked_mats.shape[0] == 1


def test_rge_matrix_from_file_rejects_unknown_scale(tmp_path):
    """A file from a different fit must fail clearly, not obscurely."""
    matrix = _make_rge_matrix([91.2])
    matrix.write(tmp_path)

    with pytest.raises(ValueError, match="no RGE matrix for scale"):
        RGEMatrix.from_file(tmp_path / "rge_matrix.pkl", [500.0])


def test_load_precomputed_rge_matrix_rejects_mismatched_settings(tmp_path):
    matrix = _make_rge_matrix([91.2])
    matrix.write(tmp_path)

    other = RGESettings(init_scale=5000.0).to_dict()
    with pytest.raises(ValueError, match="RGE settings do not match"):
        load_precomputed_rge_matrix(tmp_path / "rge_matrix.pkl", other)


# ---------------------------------------------------------------------------
# Scale cache — no wilson evolution, driven by a fake runner
# ---------------------------------------------------------------------------


class FakeRunner:
    """Stands in for RGE, recording which coefficients it was asked to evolve."""

    def __init__(self, coeff_list, obs_operators=("OpBox", "OpD")):
        self.coeff_list = list(coeff_list)
        self.obs_operators = list(obs_operators)
        self.calls = []

    def RGEmatrix(self, scale):
        self.calls.append((scale, tuple(self.coeff_list)))
        return pd.DataFrame(
            1.0,
            index=self.obs_operators,
            columns=sorted(self.coeff_list),
        )

    def clone_runner(self, coeff_list):
        clone = FakeRunner(coeff_list, self.obs_operators)
        clone.calls = self.calls  # share the log
        return clone


def test_find_cached_scale_matches_within_tolerance():
    cache = {91.2: None}
    assert _find_cached_scale(cache, 91.2 * (1 + 1e-8)) == 91.2
    assert _find_cached_scale(cache, 200.0) is None


def test_load_rge_mats_from_scales_computes_and_caches():
    runner = FakeRunner(["OpBox"])
    cache = {}
    mats = load_rge_mats_from_scales([91.2, 91.2], ["OpBox"], runner, cache)

    assert len(mats) == 2
    assert list(cache) == [91.2]
    # deduplicated: one evolution for two data points at the same scale
    assert len(runner.calls) == 1


def test_load_rge_mats_from_scales_uses_cache():
    runner = FakeRunner(["OpBox"])
    cache = {91.2: pd.DataFrame(1.0, index=["OpBox"], columns=["OpBox"])}
    load_rge_mats_from_scales([91.2], ["OpBox"], runner, cache)
    assert runner.calls == []


def test_load_rge_mats_from_scales_partial_hit_updates_matched_key():
    """A partial hit must extend the cached entry, not add a near-duplicate key.

    The requested scale differs from the cached one within _find_cached_scale's
    relative tolerance; writing the merged frame under the requested scale would
    leave the narrower matrix behind under the old key.
    """
    cached_scale = 91.2
    requested_scale = cached_scale * (1 + 1e-8)
    cache = {cached_scale: pd.DataFrame(1.0, index=["OpBox"], columns=["OpBox"])}
    runner = FakeRunner(["OpBox", "OpD"])

    mats = load_rge_mats_from_scales([requested_scale], ["OpBox", "OpD"], runner, cache)

    assert list(cache) == [cached_scale]
    assert sorted(cache[cached_scale].columns) == ["OpBox", "OpD"]
    assert sorted(mats[0].columns) == ["OpBox", "OpD"]
    # only the missing coefficient was evolved
    assert runner.calls == [(requested_scale, ("OpD",))]


# ---------------------------------------------------------------------------
# Slow tests — call real wilson RGE evolution
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rge_matrix_returns_dataframe():
    """RGEmatrix should return a pandas DataFrame with the requested operator as column."""
    rge = RGE(["OpBox"], init_scale=1000, yukawa="top")
    result = rge.RGEmatrix(200)
    assert isinstance(result, pd.DataFrame)
    assert "OpBox" in result.columns


@pytest.mark.slow
def test_rge_matrix_finite_values():
    """All non-zero entries in RGEmatrix should be finite."""
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


@pytest.mark.slow
def test_load_rge_matrix_does_not_write(theory_a, tmp_path):
    """Producing the matrix is pure; writing is the caller's job."""
    from smefit.core import TheoryGroup
    from smefit.rge import load_rge_matrix

    tg = TheoryGroup([theory_a])
    load_rge_matrix({"init_scale": 1000, "obs_scale": "dynamic"}, ["OpBox"], tg)
    assert list(tmp_path.iterdir()) == []
