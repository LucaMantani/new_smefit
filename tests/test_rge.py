"""Tests for smefit/rge/.

Fast unit tests run without real Wilson evolution.
Slow tests (marked @pytest.mark.slow) call the real wilson package.
"""

import pickle
from unittest.mock import patch

import jax.numpy as jnp
import pandas as pd
import pytest
import wilson

from smefit.core import TheoryGroup
from smefit.rge import RGE, RGEMatrix, build_rge_matrix, resolve_rge_matrices
from smefit.rge.build import _find_cached_scale, _resolve_scales
from smefit.rge.runner import _wilson_params, evolve_gs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SETTINGS = {
    "init_scale": 1000.0,
    "smeft_accuracy": "integrate",
    "adm_QCD": False,
    "yukawa": "top",
}


def _matrix(stacked, obs_operators, init_operators, scales, settings=None):
    """Build an RGEMatrix from plain nested lists."""
    return RGEMatrix(
        stacked_mats=jnp.array(stacked),
        obs_operators=obs_operators,
        init_operators=init_operators,
        scales=scales,
        settings=settings if settings is not None else dict(SETTINGS),
    )


class _StubRunner:
    """Duck-typed stand-in for RGE, so cache logic is testable without wilson.

    ``resolve_rge_matrices`` only ever calls ``clone_runner`` and ``RGEmatrix``.
    """

    def __init__(self, frames=None):
        self.frames = frames or {}
        self.clone_calls = []
        self.matrix_calls = []

    def clone_runner(self, coeff_list):
        self.clone_calls.append(sorted(coeff_list))
        return self

    def RGEmatrix(self, scale):
        self.matrix_calls.append(scale)
        return self.frames[scale].copy()


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


def test_wilson_params_none_also_zeroes_top_mass():
    """'none' switches off every Yukawa, the top included — unlike 'top'."""
    with _wilson_params("top", adm_QCD=False):
        m_t_top = wilson.run.smeft.smpar.p["m_t"]
    with _wilson_params("none", adm_QCD=False):
        m_t_none = wilson.run.smeft.smpar.p["m_t"]

    assert m_t_top != pytest.approx(0.0)
    assert m_t_none == pytest.approx(0.0)


def test_wilson_params_full_restores_default_yukawas():
    """'full' puts back the package defaults, so Vus is non-zero again."""
    from smefit.rge.runner import default_params

    with _wilson_params("full", adm_QCD=False):
        assert wilson.run.smeft.smpar.p["Vus"] == pytest.approx(default_params["Vus"])
        assert wilson.run.smeft.smpar.p["m_t"] == pytest.approx(default_params["m_t"])


def test_wilson_params_adm_qcd_switches_off_electroweak():
    """adm_QCD=True zeroes alpha_e and shrinks the EW masses to nothing."""
    with _wilson_params("top", adm_QCD=True):
        assert wilson.run.smeft.smpar.p["alpha_e"] == pytest.approx(0.0)
        assert wilson.run.smeft.smpar.p["m_W"] == pytest.approx(1e-20)
        assert wilson.run.smeft.smpar.p["m_h"] == pytest.approx(1e-20)


# ---------------------------------------------------------------------------
# RGEbasis — smefit → Warsaw translation (no evolution, so fast)
# ---------------------------------------------------------------------------


def test_rge_basis_plain_operator():
    """An entry without a 'value' gets unit coefficients, in GeV^-2."""
    rge = RGE(["OpBox"], init_scale=1000.0)
    assert rge.RGEbasis["OpBox"] == {"phiBox": pytest.approx(1e-6)}


def test_rge_basis_explicit_value():
    """An entry with a plain numeric 'value' uses it verbatim."""
    rge = RGE(["OWWW"], init_scale=1000.0)
    assert rge.RGEbasis["OWWW"] == {"W": pytest.approx(-1.0 * 1e-6)}


def test_rge_basis_otg_uses_gs_evolved_to_init_scale():
    """OtG is the '-gs' special case: gs is evaluated at the *initial* scale."""
    rge = RGE(["OtG"], init_scale=1000.0)
    assert rge.RGEbasis["OtG"] == {"uG_33": pytest.approx(-evolve_gs(1000.0) * 1e-6)}


def test_rge_basis_unknown_coefficient_is_null_vector(caplog):
    """A coefficient absent from the WCxf table is treated as an external coupling."""
    import logging

    rge = RGE(["UVcoupling"], init_scale=1000.0)
    with caplog.at_level(logging.WARNING, logger="smefit.rge.runner"):
        basis = rge.RGEbasis

    assert basis == {"UVcoupling": {}}
    assert "not present in the WCxf translation dictionary" in caplog.text


# ---------------------------------------------------------------------------
# map_to_smefit — Warsaw → smefit translation (no evolution, so fast)
# ---------------------------------------------------------------------------


def test_map_to_smefit_otg_uses_gs_evolved_to_target_scale():
    """OtG's inverse is the '-1/gs' special case, evaluated at the *target* scale."""
    rge = RGE(["OtG"], init_scale=1000.0)
    # 1e-6 GeV^-2 in, times 1e6 to get TeV^-2, times the -1/gs coefficient
    result = rge.map_to_smefit({"uG_33": complex(1e-6, 0.0)}, 500.0)
    assert result["OtG"] == pytest.approx(-1.0 / evolve_gs(500.0))


def test_map_to_smefit_skips_operators_with_no_overlap():
    """Only smefit operators touching an evolved Warsaw key appear in the output."""
    rge = RGE(["OpBox"], init_scale=1000.0)
    result = rge.map_to_smefit({"phiBox": complex(2e-6, 0.0)}, 500.0)

    assert result["OpBox"] == pytest.approx(2.0)
    assert "OtG" not in result


# ---------------------------------------------------------------------------
# RGEmatrix_dict — external couplings and the imaginary-value guard
# ---------------------------------------------------------------------------


def test_rge_matrix_dict_short_circuits_external_couplings():
    """A null-vector coefficient returns no entries without ever calling wilson.

    ``_StubRunner`` is not involved here: the point is that the real runner
    performs no evolution at all for such a coefficient, which is why this test
    is not marked slow.
    """
    rge = RGE(["UVcoupling"], init_scale=1000.0)
    assert rge.RGEmatrix_dict(500.0) == {"UVcoupling": {}}


def test_rge_matrix_external_coupling_gives_an_all_zero_column():
    """The matrix keeps the column but every observable row drops out."""
    rge = RGE(["UVcoupling"], init_scale=1000.0)
    result = rge.RGEmatrix(500.0)

    assert list(result.columns) == ["UVcoupling"]
    assert result.empty  # all-zero rows are stripped


def test_rge_matrix_dict_rejects_imaginary_results(monkeypatch):
    """A complex evolved coefficient is a bug in the setup, not a valid result."""
    from smefit.rge import runner as runner_mod

    class _FakeRun:
        dict = {"phiBox": complex(1e-3, 1e-3)}

    class _FakeWilson:
        def __init__(self, *args, **kwargs):
            pass

        def set_option(self, *args, **kwargs):
            pass

        def match_run(self, **kwargs):
            return _FakeRun()

    monkeypatch.setattr(runner_mod.wilson, "Wilson", _FakeWilson)

    rge = RGE(["OpBox"], init_scale=1000.0)
    with pytest.raises(ValueError, match="Imaginary values"):
        rge.RGEmatrix_dict(500.0)


def test_clone_runner_keeps_settings_and_sorts_names():
    """A clone differs only in its coefficient list."""
    rge = RGE(
        ["OpBox"],
        init_scale=1234.0,
        accuracy="leadinglog",
        adm_QCD=True,
        yukawa="none",
    )
    clone = rge.clone_runner(["OtG", "OpD"])

    assert clone.wc_names == ["OpD", "OtG"]
    assert clone.settings == rge.settings


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


def test_resolve_scales_applies_scale_variation(theory_a):
    """scale_variation rescales every dynamic scale."""
    tg = TheoryGroup([theory_a])
    scales = _resolve_scales(
        {"init_scale": 1000, "obs_scale": "dynamic", "scale_variation": 2.0}, tg
    )
    assert scales == [pytest.approx(200.0)] * 3


def test_resolve_scales_ignores_variation_for_fixed_scale(theory_a):
    """A fixed obs_scale wins outright — scale_variation is never applied to it."""
    tg = TheoryGroup([theory_a])
    scales = _resolve_scales(
        {"init_scale": 1000, "obs_scale": 91.2, "scale_variation": 2.0}, tg
    )
    assert scales == [91.2]


# ---------------------------------------------------------------------------
# Scale cache lookup
# ---------------------------------------------------------------------------


def test_find_cached_scale_exact_hit():
    assert _find_cached_scale({100.0: "a", 250.0: "b"}, 100.0) == 100.0


def test_find_cached_scale_within_relative_tolerance():
    """Scales computed by different routes differ in the last bits; rtol absorbs that."""
    assert _find_cached_scale({100.0: "a"}, 100.0 * (1 + 1e-6)) == 100.0


def test_find_cached_scale_miss_returns_none():
    assert _find_cached_scale({100.0: "a"}, 150.0) is None
    assert _find_cached_scale({}, 100.0) is None


# ---------------------------------------------------------------------------
# resolve_rge_matrices — cache reuse, extension and in-place update
# ---------------------------------------------------------------------------


def test_resolve_rge_matrices_full_cache_hit_never_computes():
    """Every requested coefficient is cached, so no evolution is triggered."""
    cache = {100.0: pd.DataFrame([[1.0, 2.0]], index=["Op1"], columns=["OpA", "OpB"])}
    runner = _StubRunner()

    out = resolve_rge_matrices([100.0], ["OpA"], runner, cache)

    assert len(out) == 1
    assert list(out[0].columns) == ["OpA"]
    assert out[0].loc["Op1", "OpA"] == pytest.approx(1.0)
    assert runner.matrix_calls == []


def test_resolve_rge_matrices_cache_miss_computes_and_caches():
    """An unseen scale is computed once and stored under that scale."""
    frame = pd.DataFrame([[3.0]], index=["Op1"], columns=["OpA"])
    runner = _StubRunner({100.0: frame})
    cache = {}

    out = resolve_rge_matrices([100.0], ["OpA"], runner, cache)

    assert runner.matrix_calls == [100.0]
    assert list(cache) == [100.0]
    assert out[0].loc["Op1", "OpA"] == pytest.approx(3.0)


def test_resolve_rge_matrices_deduplicates_repeated_scales():
    """Many data points at one scale cost a single evolution."""
    runner = _StubRunner({100.0: pd.DataFrame([[3.0]], index=["Op1"], columns=["OpA"])})

    out = resolve_rge_matrices([100.0, 100.0, 100.0], ["OpA"], runner, {})

    assert runner.matrix_calls == [100.0]
    assert len(out) == 3
    # copies, not shared references — callers pad these frames in place
    assert out[0] is not out[1]


def test_resolve_rge_matrices_extends_a_partial_cache():
    """Only the missing coefficients are computed; the rest come from the cache."""
    cache = {100.0: pd.DataFrame([[1.0]], index=["Op1"], columns=["OpA"])}
    runner = _StubRunner({100.0: pd.DataFrame([[5.0]], index=["Op2"], columns=["OpB"])})

    out = resolve_rge_matrices([100.0], ["OpA", "OpB"], runner, cache)

    assert runner.clone_calls == [["OpB"]]
    assert list(out[0].columns) == ["OpA", "OpB"]
    # the union of both row sets, with the gaps filled by zeros
    assert out[0].loc["Op1", "OpA"] == pytest.approx(1.0)
    assert out[0].loc["Op1", "OpB"] == pytest.approx(0.0)
    assert out[0].loc["Op2", "OpA"] == pytest.approx(0.0)
    assert out[0].loc["Op2", "OpB"] == pytest.approx(5.0)
    # the widened matrix replaces the narrow one in the cache
    assert list(cache[100.0].columns) == ["OpA", "OpB"]


def test_resolve_rge_matrices_extends_cache_under_the_matched_key():
    """A tolerance-matched hit must not leave a stale narrower duplicate behind."""
    cache = {100.0: pd.DataFrame([[1.0]], index=["Op1"], columns=["OpA"])}
    requested = 100.0 * (1 + 1e-6)
    runner = _StubRunner(
        {requested: pd.DataFrame([[5.0]], index=["Op1"], columns=["OpB"])}
    )

    resolve_rge_matrices([requested], ["OpA", "OpB"], runner, cache)

    assert list(cache) == [100.0]
    assert list(cache[100.0].columns) == ["OpA", "OpB"]


# ---------------------------------------------------------------------------
# RGEMatrix — the rge_matrix.pkl payload, both directions
# ---------------------------------------------------------------------------


def test_to_dump_dict_layout():
    """The payload is 'rge_settings' plus one float-keyed frame per scale."""
    mat = _matrix([[[1.0, 2.0], [3.0, 4.0]]], ["Op1", "Op2"], ["OpA", "OpB"], [100.0])
    payload = mat.to_dump_dict()

    assert payload["rge_settings"] == SETTINGS
    assert [k for k in payload if k != "rge_settings"] == [100.0]
    frame = payload[100.0]
    assert list(frame.index) == ["Op1", "Op2"]
    assert list(frame.columns) == ["OpA", "OpB"]
    assert frame.loc["Op2", "OpB"] == pytest.approx(4.0)


def test_to_dump_dict_collapses_duplicate_scales():
    """Data points sharing a scale store one frame, not one per point."""
    mat = _matrix([[[1.0]], [[2.0]]], ["Op1"], ["OpA"], [100.0, 100.0])
    payload = mat.to_dump_dict()

    assert [k for k in payload if k != "rge_settings"] == [100.0]
    # last frame written wins
    assert payload[100.0].loc["Op1", "OpA"] == pytest.approx(2.0)


def test_write_read_round_trip(tmp_path):
    """write() then read_cache() returns the scale-keyed cache, settings stripped."""
    mat = _matrix([[[1.0, 2.0], [3.0, 4.0]]], ["Op1", "Op2"], ["OpA", "OpB"], [100.0])
    mat.write(tmp_path)

    assert (tmp_path / RGEMatrix.FILENAME).exists()

    cache = RGEMatrix.read_cache(tmp_path / RGEMatrix.FILENAME, SETTINGS)

    assert "rge_settings" not in cache
    assert list(cache) == [100.0]
    assert cache[100.0].loc["Op2", "OpA"] == pytest.approx(3.0)


def test_write_creates_missing_directories_and_honours_name(tmp_path):
    mat = _matrix([[[1.0]]], ["Op1"], ["OpA"], [100.0])
    target = tmp_path / "nested" / "deeper"
    mat.write(target, name="custom_name.pkl")

    assert (target / "custom_name.pkl").exists()
    assert not (target / RGEMatrix.FILENAME).exists()


def test_read_cache_rejects_mismatched_settings(tmp_path):
    """Settings are compared with strict equality — a differing yukawa is fatal."""
    mat = _matrix([[[1.0]]], ["Op1"], ["OpA"], [100.0])
    mat.write(tmp_path)

    with pytest.raises(ValueError, match="do not match"):
        RGEMatrix.read_cache(
            tmp_path / RGEMatrix.FILENAME, {**SETTINGS, "yukawa": "full"}
        )


@pytest.mark.parametrize(
    "payload, expected",
    [
        (["not", "a", "dict"], "expected a dict"),
        ({100.0: pd.DataFrame()}, "missing the 'rge_settings' entry"),
        ({"rge_settings": "top"}, "'rge_settings' must be a dict"),
        ({"rge_settings": SETTINGS}, "holds no matrices"),
        (
            {"rge_settings": SETTINGS, "obs_operators": ["Op1"]},
            "must be a scale in GeV",
        ),
        (
            {"rge_settings": SETTINGS, 100.0: [[1.0]]},
            "must be a DataFrame",
        ),
    ],
)
def test_read_cache_rejects_a_pickle_that_is_not_an_rge_matrix(
    tmp_path, payload, expected
):
    """`rg_matrix` is a user-supplied path: a wrong file must say so here, not
    fail obscurely inside resolve_rge_matrices."""
    target = tmp_path / RGEMatrix.FILENAME
    with open(target, "wb") as f:
        pickle.dump(payload, f)

    with pytest.raises(ValueError, match=expected):
        RGEMatrix.read_cache(target, SETTINGS)


def test_read_cache_rejects_a_file_that_is_not_a_pickle(tmp_path):
    target = tmp_path / RGEMatrix.FILENAME
    target.write_text("name: not a pickle at all\n")

    with pytest.raises(ValueError, match="not a readable pickle"):
        RGEMatrix.read_cache(target, SETTINGS)


def test_read_cache_resolves_prefix_path(tmp_path):
    """A prefix-relative path is resolved here, not only at config-parse time.

    This is what lets an external chi2 module pass a runcard-style
    `smefit_results/...` path straight through to read_cache.
    """
    fit_dir = tmp_path / "fits" / "my_fit"
    mat = _matrix([[[1.0]]], ["Op1"], ["OpA"], [100.0])
    mat.write(fit_dir)

    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_results": str(tmp_path)},
    ):
        cache = RGEMatrix.read_cache(
            f"smefit_results/fits/my_fit/{RGEMatrix.FILENAME}", SETTINGS
        )

    assert cache[100.0].loc["Op1", "OpA"] == pytest.approx(1.0)


def test_read_cache_downloads_a_missing_fit(tmp_path):
    """A matrix missing under smefit_results/fits goes through the server fetch."""
    fit_dir = tmp_path / "fits" / "my_fit"
    mat = _matrix([[[1.0]]], ["Op1"], ["OpA"], [100.0])

    def fake_fetch(path):
        mat.write(path.parent)

    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_results": str(tmp_path)},
    ), patch("smefit.rge.matrix.fetch_fit_if_missing", side_effect=fake_fetch) as fetch:
        cache = RGEMatrix.read_cache(
            f"smefit_results/fits/my_fit/{RGEMatrix.FILENAME}", SETTINGS
        )

    fetch.assert_called_once_with(fit_dir / RGEMatrix.FILENAME)
    assert cache[100.0].loc["Op1", "OpA"] == pytest.approx(1.0)


def test_read_cache_missing_file_names_both_forms(tmp_path):
    """An unresolved prefix is diagnosable: the error shows raw and resolved."""
    # deliberately not under smefit_results/fits, so no download is attempted
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_results": str(tmp_path)},
    ):
        with pytest.raises(FileNotFoundError) as exc:
            RGEMatrix.read_cache("smefit_results/nowhere/rge_matrix.pkl", SETTINGS)

    assert "smefit_results/nowhere/rge_matrix.pkl" in str(exc.value)
    assert str(tmp_path / "nowhere" / "rge_matrix.pkl") in str(exc.value)


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
def test_build_rge_matrix_shape(theory_a):
    """build_rge_matrix should return an RGEMatrix with a 3-D stacked array."""
    from smefit.core import TheoryGroup
    from smefit.rge import build_rge_matrix

    tg = TheoryGroup([theory_a])
    rge_dict = {"init_scale": 1000, "obs_scale": "dynamic"}
    result = build_rge_matrix(rge_dict, ["OpBox"], tg)

    assert isinstance(result, RGEMatrix)
    assert result.stacked_mats.ndim == 3
    assert result.init_operators == ["OpBox"]
    # dynamic: 3 data points (all scale=100), one unique scale → shape (1, n_obs_ops, 1)
    # The actual number of unique scales may collapse to 1 since all are 100.0
    n_unique_scales = result.stacked_mats.shape[0]
    assert n_unique_scales >= 1


@pytest.mark.slow
def test_rge_evolve_is_linear_and_sums_shared_warsaw_keys():
    """RGEevolve superposes coefficients before evolving.

    OtW and OtZ both feed the Warsaw key uB_33, so this exercises the
    accumulate branch as well as the linearity the whole module assumes.
    """
    rge = RGE(["OtW", "OtZ"], init_scale=1000.0)

    single = rge.RGEevolve({"OtW": 1.0, "OtZ": 1.0}, 200.0)
    double = rge.RGEevolve({"OtW": 2.0, "OtZ": 2.0}, 200.0)

    assert single
    for op, value in single.items():
        assert double[op] == pytest.approx(2.0 * value, rel=1e-8, abs=1e-12)


# ---------------------------------------------------------------------------
# build_rge_matrix — reading a precomputed pickle needs no evolution, so this
# stays fast even though it goes through the full top-level entry point.
# ---------------------------------------------------------------------------


def test_build_rge_matrix_reads_precomputed_cache_and_pads_rows(
    tmp_path, theory_a, theory_b
):
    """A fully-populated pickle short-circuits wilson, and rows are padded to the union.

    theory_a sits at scale 100 and theory_b at 200, with a different observable
    operator generated at each, so the two frames must be padded to a common
    row set before they can be stacked.
    """
    tg = TheoryGroup([theory_a, theory_b])
    payload = {
        "rge_settings": dict(SETTINGS),
        100.0: pd.DataFrame([[1.0]], index=["Op1"], columns=["OpBox"]),
        200.0: pd.DataFrame([[2.0]], index=["Op2"], columns=["OpBox"]),
    }
    path = tmp_path / RGEMatrix.FILENAME
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)

    rge_dict = {"init_scale": 1000.0, "obs_scale": "dynamic", "rg_matrix": str(path)}
    result = build_rge_matrix(rge_dict, ["OpBox"], tg)

    assert result.obs_operators == ["Op1", "Op2"]
    assert result.init_operators == ["OpBox"]
    # 6 data points (3 per theory), 2 observable operators, 1 coefficient
    assert result.stacked_mats.shape == (6, 2, 1)
    assert result.settings == SETTINGS

    # first three points sit at scale 100 → only Op1 is non-zero
    assert result.stacked_mats[0, 0, 0] == pytest.approx(1.0)
    assert result.stacked_mats[0, 1, 0] == pytest.approx(0.0)
    # last three sit at scale 200 → only Op2 is non-zero
    assert result.stacked_mats[5, 0, 0] == pytest.approx(0.0)
    assert result.stacked_mats[5, 1, 0] == pytest.approx(2.0)


def test_build_rge_matrix_rejects_a_pickle_from_different_settings(tmp_path, theory_a):
    """The settings guard fires through the top-level entry point too."""
    tg = TheoryGroup([theory_a])
    payload = {
        "rge_settings": {**SETTINGS, "yukawa": "full"},
        100.0: pd.DataFrame([[1.0]], index=["Op1"], columns=["OpBox"]),
    }
    path = tmp_path / RGEMatrix.FILENAME
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)

    rge_dict = {"init_scale": 1000.0, "obs_scale": "dynamic", "rg_matrix": str(path)}
    with pytest.raises(ValueError, match="do not match"):
        build_rge_matrix(rge_dict, ["OpBox"], tg)
