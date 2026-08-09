"""Unit tests for the smefit/blackjax_samplers package — blackjax itself is mocked.

The real samplers are exercised in tests/test_blackjax_nuts.py (slow) and
tests/test_bayes_update_blackjax.py (slow).
"""

import json
from unittest.mock import MagicMock, patch

import anesthetic
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smefit.blackjax_samplers import (
    _SAMPLER_REGISTRY,
    BJ_ALGORITHM_SETTINGS,
    BJ_ALGORITHMS,
    BJ_SHARED_SETTINGS,
    get_sampler,
)
from smefit.blackjax_samplers._common import _HealthReport
from smefit.blackjax_samplers.nested_sampling import _nested_sampling_diagnostics
from smefit.blackjax_samplers.nuts import _thin_chains
from smefit.blackjax_samplers.nuts import run as _run_nuts
from smefit.priors import Prior, _UniformDist

N_CHAINS = 3
N_DRAWS = 20
N_DIMS = 2
PARAM_NAMES = ["OpA", "OpB"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_and_settings_map_agree():
    assert set(_SAMPLER_REGISTRY) == set(BJ_ALGORITHM_SETTINGS) == set(BJ_ALGORITHMS)


def test_algorithm_settings_are_disjoint_from_shared():
    for keys in BJ_ALGORITHM_SETTINGS.values():
        assert not (keys & BJ_SHARED_SETTINGS)


def test_get_sampler_returns_callable():
    assert callable(get_sampler("nested_sampling"))
    assert callable(get_sampler("nuts"))


def test_get_sampler_unknown_raises():
    with pytest.raises(ValueError, match="Unknown BlackJAX algorithm"):
        get_sampler("metropolis")

    try:
        get_sampler("metropolis")
    except ValueError as exc:  # the message must enumerate the real options
        assert "nested_sampling" in str(exc)
        assert "nuts" in str(exc)


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------


def _ramp_positions():
    """(C, S, d) block whose values encode (chain, draw) for identity checks.

    Note the chains are separated by construction, so this block always has a
    large R-hat — use `_mixed_positions` for anything asserting a healthy run.
    """
    chain = jnp.arange(N_CHAINS)[:, None, None]
    draw = jnp.arange(N_DRAWS)[None, :, None]
    dim = jnp.arange(N_DIMS)[None, None, :]
    return chain * 1000.0 + draw * 10.0 + dim


def _mixed_positions(scale=0.3):
    """(C, S, d) draws from one common distribution, so R-hat sits near 1."""
    return scale * jax.random.normal(jax.random.PRNGKey(0), (N_CHAINS, N_DRAWS, N_DIMS))


def test_thin_chains_keeps_everything_when_it_fits():
    positions = _ramp_positions()
    out = _thin_chains(positions, N_CHAINS * N_DRAWS)
    assert out.shape == (N_CHAINS * N_DRAWS, N_DIMS)


def test_thin_chains_truncates_to_n_samples():
    positions = _ramp_positions()
    out = _thin_chains(positions, 12)
    assert out.shape == (12, N_DIMS)


def test_thin_chains_interleaves_chains():
    """Truncation must cost every chain equally, not gut the last one."""
    positions = _ramp_positions()
    out = _thin_chains(positions, 12)
    chains = jnp.floor(out[:, 0] / 1000.0)
    counts = [int(jnp.sum(chains == c)) for c in range(N_CHAINS)]
    assert counts == [4, 4, 4]
    # the first sweep is one draw from each chain, in order
    assert [int(c) for c in chains[:N_CHAINS]] == list(range(N_CHAINS))


def test_thin_chains_warns_when_fewer_draws_than_requested(caplog):
    positions = _ramp_positions()
    with caplog.at_level("WARNING"):
        out = _thin_chains(positions, 10_000)
    assert out.shape == (N_CHAINS * N_DRAWS, N_DIMS)
    assert any("fewer than the requested" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# _run_nuts, with blackjax mocked out
# ---------------------------------------------------------------------------


@pytest.fixture
def nuts_settings(tmp_path):
    return {
        "algorithm": "nuts",
        "seed": 0,
        "log_dir": str(tmp_path),
        "num_chains": N_CHAINS,
        "num_warmup": 5,
        "num_samples": N_DRAWS,
        "target_acceptance_rate": 0.8,
        "max_num_doublings": 10,
    }


@pytest.fixture
def uniform_prior():
    return Prior(
        [_UniformDist(-1.0, 1.0), _UniformDist(-2.0, 2.0)],
        PARAM_NAMES,
        specs={
            "OpA": {"dist": "uniform", "low": -1.0, "high": 1.0},
            "OpB": {"dist": "uniform", "low": -2.0, "high": 2.0},
        },
    )


def _run_mocked_nuts(
    prior,
    settings,
    n_samples,
    positions,
    is_divergent=None,
    acceptance=None,
    step_sizes=None,
    leapfrogs=None,
    tree_depth=None,
):
    """Call _run_nuts with window_adaptation/run_inference_algorithm mocked.

    The mock bypasses jax.vmap by returning the whole (C, S, d) block from the
    per-chain function, which is what vmap would have stacked anyway.
    """
    if is_divergent is None:
        is_divergent = jnp.zeros((N_CHAINS, N_DRAWS), dtype=bool)
    if acceptance is None:
        acceptance = jnp.full((N_CHAINS, N_DRAWS), 0.9)
    if step_sizes is None:
        step_sizes = jnp.full((N_CHAINS,), 0.5)
    if leapfrogs is None:
        leapfrogs = jnp.full((N_CHAINS, N_DRAWS), 7, dtype=jnp.int32)
    if tree_depth is None:
        tree_depth = jnp.full((N_CHAINS, N_DRAWS), 3, dtype=jnp.int32)

    mock_warmup = MagicMock()
    mock_warmup.run.return_value = (MagicMock(parameters={}, state=MagicMock()), None)

    log_likelihood = lambda x: -jnp.sum(x**2) / 2.0

    # The real kernel records logdensity = log_prior_unconstrained + loglik at
    # each draw, and _run_nuts recovers the likelihood by subtracting the prior
    # back off. Build it consistently here so that subtraction is exercised
    # rather than papered over.
    logdensity = _real_vmap(
        lambda u: prior.log_prob_unconstrained(u)
        + log_likelihood(prior.from_unconstrained(u))
    )(positions.reshape(-1, N_DIMS)).reshape(N_CHAINS, N_DRAWS)

    with (
        patch(
            "smefit.blackjax_samplers.nuts.blackjax.window_adaptation",
            return_value=mock_warmup,
        ),
        patch("smefit.blackjax_samplers.nuts.blackjax.nuts", return_value=MagicMock()),
        patch(
            "smefit.blackjax_samplers.nuts.run_inference_algorithm",
            return_value=(
                None,
                (
                    positions,
                    logdensity,
                    leapfrogs,
                    tree_depth,
                    is_divergent,
                    acceptance,
                ),
            ),
        ),
        patch(
            "smefit.blackjax_samplers.nuts.jax.vmap",
            side_effect=lambda fn, *a, **k: _mocked_vmap(
                fn,
                positions,
                logdensity,
                leapfrogs,
                tree_depth,
                is_divergent,
                acceptance,
                step_sizes,
                *a,
                **k,
            ),
        ),
    ):
        return _run_nuts(
            jnp.zeros(2, dtype="uint32"),
            prior,
            log_likelihood,
            n_samples,
            settings,
        )


# captured before the patch so the mock can delegate for every other vmap call
_real_vmap = __import__("jax").vmap


def _mocked_vmap(
    fn,
    positions,
    logdensity,
    leapfrogs,
    tree_depth,
    is_divergent,
    acceptance,
    step_sizes,
    *args,
    **kwargs,
):
    """Stand in for jax.vmap over the two per-chain stages of _run_nuts.

    Returns the whole (C, ...) block that vmap would have stacked, and delegates
    every other vmap call (the bijector maps) to the real implementation.
    """
    if fn.__name__ == "_run_warmup":
        dummy_state = jnp.zeros((N_CHAINS, N_DIMS))
        return lambda *_: (
            dummy_state,
            step_sizes,
            jnp.ones((N_CHAINS, N_DIMS)),
        )
    if fn.__name__ == "_run_sampling":
        return lambda *_: (
            positions,
            logdensity,
            leapfrogs,
            tree_depth,
            is_divergent,
            acceptance,
        )
    return _real_vmap(fn, *args, **kwargs)


def test_run_nuts_output_shape_and_no_logz(uniform_prior, nuts_settings):
    positions = _ramp_positions() * 0.001  # keep inside a sane u-range
    out = _run_mocked_nuts(uniform_prior, nuts_settings, 12, positions)

    assert out.samples.shape == (12, N_DIMS)
    assert out.best_point.shape == (N_DIMS,)
    assert out.logz is None, "NUTS provides no evidence estimate"
    assert out.diagnostics["algorithm"] == "nuts"


def test_run_nuts_samples_are_in_constrained_space(uniform_prior, nuts_settings):
    """Draws come back as physical/sampler-space values inside the prior box."""
    positions = _ramp_positions() * 0.001
    out = _run_mocked_nuts(uniform_prior, nuts_settings, 12, positions)

    assert bool(jnp.all(out.samples[:, 0] >= -1.0))
    assert bool(jnp.all(out.samples[:, 0] <= 1.0))
    assert bool(jnp.all(out.samples[:, 1] >= -2.0))
    assert bool(jnp.all(out.samples[:, 1] <= 2.0))


def test_run_nuts_writes_diagnostics_and_samples(
    uniform_prior, nuts_settings, tmp_path
):
    positions = _ramp_positions() * 0.001
    _run_mocked_nuts(uniform_prior, nuts_settings, 12, positions)

    assert (tmp_path / "nuts_samples.csv").exists()
    diag_file = tmp_path / "nuts_diagnostics.json"
    assert diag_file.exists()

    diag = json.loads(diag_file.read_text())
    assert set(diag["rhat"]) == set(PARAM_NAMES)
    assert set(diag["ess"]) == set(PARAM_NAMES)
    assert diag["divergences"] == 0
    assert diag["num_chains"] == N_CHAINS

    header = (tmp_path / "nuts_samples.csv").read_text().splitlines()[0]
    assert header.split(",") == PARAM_NAMES


def test_run_nuts_warns_on_bad_rhat(uniform_prior, nuts_settings, caplog):
    """Chains parked at wildly different places must trip the R-hat warning."""
    offsets = jnp.array([-6.0, 0.0, 6.0])[:, None, None]
    positions = offsets + 0.01 * _ramp_positions()

    with caplog.at_level("WARNING"):
        _run_mocked_nuts(uniform_prior, nuts_settings, 12, positions)

    assert any("R-hat" in m for m in caplog.messages)


def test_run_nuts_rhat_detects_within_chain_drift(uniform_prior, nuts_settings, caplog):
    """Every chain drifting the same way is non-stationary, but the chains agree
    with each other — the classic Gelman-Rubin statistic reads ~0.98 here and
    sees nothing. Only the split-chain R-hat catches it, so this pins the
    choice of `blackjax.diagnostics.rhat` over `potential_scale_reduction`.
    """
    drift = jnp.linspace(0.0, 5.0, N_DRAWS)[None, :, None] * jnp.ones(
        (N_CHAINS, 1, N_DIMS)
    )
    positions = drift + 0.1 * jax.random.normal(
        jax.random.PRNGKey(1), (N_CHAINS, N_DRAWS, N_DIMS)
    )

    with caplog.at_level("WARNING"):
        out = _run_mocked_nuts(uniform_prior, nuts_settings, 12, positions)

    assert out.diagnostics["max_rhat"] > 1.1
    assert out.diagnostics["converged"] is False
    assert any("R-hat" in m for m in caplog.messages)


def test_run_nuts_warns_on_divergences(uniform_prior, nuts_settings, caplog):
    positions = _ramp_positions() * 0.001
    is_divergent = jnp.zeros((N_CHAINS, N_DRAWS), dtype=bool).at[0, :3].set(True)

    with caplog.at_level("WARNING"):
        out = _run_mocked_nuts(
            uniform_prior, nuts_settings, 12, positions, is_divergent
        )

    assert out.diagnostics["divergences"] == 3
    assert any("divergent" in m for m in caplog.messages)


def test_run_nuts_max_loglikelihood_is_likelihood_not_posterior(
    uniform_prior, nuts_settings
):
    """FitResult.chi2_val is -2 * max_loglikelihood, so the prior must not leak in.

    _run_nuts recovers the likelihood as ``logdensity - log_prior_unconstrained``.
    The draws are offset away from the origin on purpose: with draws at u ~ 0 the
    likelihood here is O(1e-7) against an O(1) prior, and the subtraction is then
    pure float32 cancellation noise rather than a test of anything.
    """
    positions = 0.5 + _ramp_positions() * 0.001
    out = _run_mocked_nuts(uniform_prior, nuts_settings, 12, positions)

    expected = -float(jnp.sum(out.best_point**2)) / 2.0
    assert out.max_loglikelihood == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Cost model and health verdict
# ---------------------------------------------------------------------------


def test_run_nuts_reports_cost_model(uniform_prior, nuts_settings, tmp_path):
    """Runtime = draws x leapfrogs/draw x ms/gradient — all three must be
    readable off a single run, which is what makes 'is this speed reasonable?'
    answerable without a bespoke probe."""
    positions = _mixed_positions()
    out = _run_mocked_nuts(uniform_prior, nuts_settings, 12, positions)

    d = out.diagnostics
    assert d["leapfrogs_per_draw_mean"] == pytest.approx(7.0)
    assert d["leapfrogs_per_draw_max"] == 7
    assert d["tree_depth_mean"] == pytest.approx(3.0)
    assert d["tree_depth_max"] == 3
    assert d["treedepth_saturation"] == 0.0
    assert d["total_gradient_evaluations"] == 7 * N_CHAINS * N_DRAWS
    assert d["ms_per_gradient"] > 0
    assert d["converged"] is True

    written = json.loads((tmp_path / "nuts_diagnostics.json").read_text())
    assert written["total_gradient_evaluations"] == d["total_gradient_evaluations"]


def test_run_nuts_healthy_run_logs_no_error(uniform_prior, nuts_settings, caplog):
    positions = _mixed_positions()
    with caplog.at_level("ERROR"):
        out = _run_mocked_nuts(uniform_prior, nuts_settings, 12, positions)
    assert out.diagnostics["converged"] is True
    assert not caplog.messages


def test_run_nuts_flags_step_size_collapse(uniform_prior, nuts_settings, caplog):
    """The failure mode that looks like 'slow' but is actually 'stuck'."""
    positions = _ramp_positions() * 0.001
    with caplog.at_level("ERROR"):
        out = _run_mocked_nuts(
            uniform_prior,
            nuts_settings,
            12,
            positions,
            step_sizes=jnp.full((N_CHAINS,), 1e-200),
        )
    assert out.diagnostics["converged"] is False
    assert any("step size collapsed" in m for m in caplog.messages)
    assert any("do not use this posterior" in m for m in caplog.messages)


def test_run_nuts_flags_universal_divergence(uniform_prior, nuts_settings, caplog):
    positions = _ramp_positions() * 0.001
    with caplog.at_level("ERROR"):
        out = _run_mocked_nuts(
            uniform_prior,
            nuts_settings,
            12,
            positions,
            is_divergent=jnp.ones((N_CHAINS, N_DRAWS), dtype=bool),
        )
    assert out.diagnostics["converged"] is False
    assert out.diagnostics["divergence_rate"] == 1.0
    assert any("diverged on" in m for m in caplog.messages)


def test_run_nuts_flags_zero_acceptance(uniform_prior, nuts_settings, caplog):
    positions = _ramp_positions() * 0.001
    with caplog.at_level("ERROR"):
        out = _run_mocked_nuts(
            uniform_prior,
            nuts_settings,
            12,
            positions,
            acceptance=jnp.zeros((N_CHAINS, N_DRAWS)),
        )
    assert out.diagnostics["converged"] is False
    assert any("acceptance rate" in m for m in caplog.messages)


def test_run_nuts_warns_on_treedepth_saturation(uniform_prior, nuts_settings, caplog):
    """Saturation is not a failure, but it is the dominant runtime term and the
    reason a fit takes hours rather than minutes."""
    cap = nuts_settings["max_num_doublings"]
    positions = _mixed_positions()
    with caplog.at_level("WARNING"):
        out = _run_mocked_nuts(
            uniform_prior,
            nuts_settings,
            12,
            positions,
            tree_depth=jnp.full((N_CHAINS, N_DRAWS), cap, dtype=jnp.int32),
            leapfrogs=jnp.full((N_CHAINS, N_DRAWS), 2**cap, dtype=jnp.int32),
        )
    assert out.diagnostics["treedepth_saturation"] == 1.0
    assert out.diagnostics["converged"] is True  # slow, but not broken
    assert any("maximum tree depth" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# Nested sampling diagnostics
# ---------------------------------------------------------------------------


def _nested_samples(n=4000, n_live=50):
    """A synthetic anesthetic.NestedSamples good enough for the summary stats."""
    rng = np.random.default_rng(0)
    logL = np.sort(rng.normal(0.0, 3.0, n))
    logL_birth = np.concatenate([np.full(n_live, -np.inf), logL[: n - n_live]])
    return anesthetic.NestedSamples(
        data=rng.normal(size=(n, 3)),
        logL=logL,
        logL_birth=logL_birth,
        columns=["a", "b", "c"],
    )


def _ns_diagnostics(caplog=None, **overrides):
    kwargs = {
        "nested_samples": _nested_samples(),
        "n_free": 2,
        "n_live": 50,
        "n_dead": 4000,
        "ess_value": 2000,
        "n_requested": 1000,
        "n_stored": 1000,
        "logzs": jnp.array([-6.4, -6.5, -6.6]),
        "termination_margin": -2.5,
        "log_precision": -2.0,
    }
    kwargs.update(overrides)
    return _nested_sampling_diagnostics(**kwargs)


def test_nested_diagnostics_healthy_run_converges(caplog):
    with caplog.at_level("ERROR"):
        diag = _ns_diagnostics()

    assert diag["converged"] is True
    assert caplog.messages == []
    # Handley-Lemos statistics come through with their error bars.
    for key in ("logz", "logz_std", "D_KL", "D_KL_std", "logL_P", "d_G", "d_G_std"):
        assert key in diag
    assert diag["ess_efficiency"] == pytest.approx(2000 / 4000)


def test_nested_diagnostics_flags_unusable_ess(caplog):
    """A posterior of a few distinct points is a failure, not an imprecision."""
    with caplog.at_level("ERROR"):
        diag = _ns_diagnostics(ess_value=10, n_stored=10)

    assert diag["converged"] is False
    assert any("effective sample size" in m.lower() for m in caplog.messages)


def test_nested_diagnostics_warns_on_unconstrained_directions(caplog):
    """d_G counts the directions the data constrains; well below n_free means
    flat directions, the same pathology whitening reports from the Hessian."""
    with caplog.at_level("WARNING"):
        diag = _ns_diagnostics(n_free=6)

    assert diag["d_G"] < 6
    assert any("unconstrained" in m for m in caplog.messages)


def test_nested_diagnostics_quiet_when_dimensionality_matches(caplog):
    with caplog.at_level("WARNING"):
        _ns_diagnostics(n_free=2)

    assert not any("unconstrained" in m for m in caplog.messages)


def test_nested_diagnostics_warns_when_fewer_draws_than_requested(caplog):
    with caplog.at_level("WARNING"):
        diag = _ns_diagnostics(ess_value=400, n_requested=1000, n_stored=400)

    assert diag["n_stored"] == 400
    assert any("posterior samples were stored" in m for m in caplog.messages)


def test_nested_diagnostics_warns_on_early_termination(caplog):
    """n_live * D_KL is the run length needed to compress prior to posterior."""
    with caplog.at_level("WARNING"):
        diag = _ns_diagnostics(n_dead=5)

    assert diag["expected_n_dead"] > 5
    assert any("well short of" in m for m in caplog.messages)


def test_nested_diagnostics_flags_non_finite_evidence(caplog):
    with caplog.at_level("ERROR"):
        diag = _ns_diagnostics(logzs=jnp.array([jnp.nan, jnp.nan]))

    assert diag["converged"] is False
    assert any("non-finite" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# Shared verdict scaffolding
# ---------------------------------------------------------------------------


def test_health_report_clean_run_converges(caplog):
    with caplog.at_level("WARNING"):
        diag = _HealthReport("NUTS").finish({})
    assert diag["converged"] is True
    assert caplog.messages == []


def test_health_report_warning_does_not_fail_the_run(caplog):
    """A warning says the posterior is imprecise, not unusable — the two must
    stay distinguishable or every soft finding would invalidate a fit."""
    report = _HealthReport("NUTS")
    with caplog.at_level("WARNING"):
        report.warn("something is a bit off: %d", 3)
        diag = report.finish({})

    assert diag["converged"] is True
    assert "something is a bit off: 3" in caplog.messages


def test_health_report_collects_every_failure_and_names_them(caplog):
    report = _HealthReport("Nested sampling")
    with caplog.at_level("ERROR"):
        report.fail("first thing", "detail one")
        report.fail("second thing", "detail two")
        diag = report.finish({})

    assert diag["converged"] is False
    summary = caplog.messages[-1]
    assert "Nested sampling run FAILED" in summary
    assert "first thing, second thing" in summary
