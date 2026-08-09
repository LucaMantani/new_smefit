"""
smefit.blackjax_samplers.nuts

The ``nuts`` algorithm behind ``blackjax_settings.algorithm``: No-U-Turn
Hamiltonian Monte Carlo, run in the prior's unconstrained reparametrisation.

A plain helper module, NOT registered in ``smefit.app.smefit_providers``. It
must not import ``smefit.blackjax_fit`` or ``smefit.config``, which import the
registry that imports this.
"""

import logging
import math
import os
import time

import blackjax
import jax
import jax.numpy as jnp
import pandas as pd
from blackjax.diagnostics import ess_bulk, ess_tail, rhat
from blackjax.util import run_inference_algorithm

from smefit.blackjax_samplers._common import (
    SamplerOutput,
    _HealthReport,
    _write_diagnostics,
)

log = logging.getLogger(__name__)

#: Keys of ``blackjax_settings`` this algorithm owns.
SETTINGS = frozenset(
    {
        "num_chains",
        "num_warmup",
        "num_samples",
        "target_acceptance_rate",
        "max_num_doublings",
    }
)


#: A step size at or below this means the chain never moved.
_STEP_SIZE_COLLAPSE = 1e-12


def _nuts_diagnostics(
    prior,
    positions,
    leapfrogs,
    tree_depth,
    is_divergent,
    acceptance,
    step_sizes,
    max_num_doublings,
    sampling_seconds=None,
    warmup_seconds=None,
):
    """Convergence and cost summary for a (C, S, d) block of NUTS draws.

    Also decides whether the run is usable at all: ``diagnostics["converged"]``
    is False when the sampler failed outright, as opposed to merely mixing
    poorly. Those cases warrant an error rather than a warning, because the
    posterior that comes out is meaningless rather than imprecise.
    """
    rhat_per_param = rhat(positions, chain_axis=0, sample_axis=1)
    ess_bulk_per_param = ess_bulk(positions, chain_axis=0, sample_axis=1)
    ess_tail_per_param = ess_tail(positions, chain_axis=0, sample_axis=1)
    num_chains, num_draws, _ = positions.shape
    total = num_chains * num_draws
    n_div = int(jnp.sum(is_divergent))
    total_grads = int(jnp.sum(leapfrogs))
    min_step = min(float(s) for s in step_sizes)

    diagnostics = {
        "algorithm": "nuts",
        "num_chains": int(num_chains),
        "num_samples": int(num_draws),
        "rhat": {
            name: float(rhat_per_param[i]) for i, name in enumerate(prior.param_names)
        },
        "ess": {
            name: float(ess_bulk_per_param[i])
            for i, name in enumerate(prior.param_names)
        },
        "ess_tail": {
            name: float(ess_tail_per_param[i])
            for i, name in enumerate(prior.param_names)
        },
        "max_rhat": float(jnp.max(rhat_per_param)),
        "min_ess": float(jnp.min(ess_bulk_per_param)),
        "min_ess_tail": float(jnp.min(ess_tail_per_param)),
        "divergences": n_div,
        "divergence_rate": n_div / total,
        "mean_acceptance_rate": float(jnp.mean(acceptance)),
        "step_size": [float(s) for s in step_sizes],
        # Cost model: runtime = draws x leapfrogs_per_draw x ms_per_gradient.
        "leapfrogs_per_draw_mean": float(jnp.mean(leapfrogs)),
        "leapfrogs_per_draw_median": float(jnp.median(leapfrogs)),
        "leapfrogs_per_draw_max": int(jnp.max(leapfrogs)),
        "tree_depth_mean": float(jnp.mean(tree_depth)),
        "tree_depth_max": int(jnp.max(tree_depth)),
        "treedepth_saturation": float(jnp.mean(tree_depth >= max_num_doublings)),
        "total_gradient_evaluations": total_grads,
    }
    if warmup_seconds is not None:
        diagnostics["warmup_seconds"] = float(warmup_seconds)
    if sampling_seconds is not None and total_grads:
        diagnostics["sampling_seconds"] = float(sampling_seconds)
        # Sampling phase only, matching total_gradient_evaluations. Warmup
        # gradients are not observable (blackjax discards the per-step info),
        # so folding warmup time in here would inflate this by ~num_warmup/num_draws.
        diagnostics["ms_per_gradient"] = 1e3 * sampling_seconds / total_grads

    # --- outright failure: the posterior is unusable ------------------------
    report = _HealthReport("NUTS", log)
    if not math.isfinite(min_step) or min_step < _STEP_SIZE_COLLAPSE:
        report.fail(
            "step-size collapse",
            "NUTS step size collapsed to %.3e: the chains never moved. This "
            "usually means the log-density or its gradient returned NaN/inf "
            "somewhere in the prior's support. Check the `whitening:` block "
            "(a near-singular Hessian makes the whitened prior box span a huge "
            "physical range), lower `sigma_prior`, or use "
            "`blackjax_settings.algorithm: nested_sampling`, which does not "
            "need gradients.",
            min_step,
        )
    if diagnostics["divergence_rate"] > 0.5:
        report.fail(
            "universal divergence",
            "NUTS diverged on %.0f%% of draws. The posterior geometry is beyond "
            "what the sampler can integrate; the draws are not from the target "
            "distribution. Raise target_acceptance_rate towards 0.95, reduce "
            "`whitening.sigma_prior`, or switch to "
            "`blackjax_settings.algorithm: nested_sampling`.",
            100 * diagnostics["divergence_rate"],
        )
    if diagnostics["mean_acceptance_rate"] < 0.01:
        report.fail(
            "zero acceptance",
            "NUTS mean acceptance rate is %.4f: essentially every proposal was "
            "rejected, so the chains are stuck at their starting points.",
            diagnostics["mean_acceptance_rate"],
        )
    if diagnostics["max_rhat"] > 1.1:
        # A sampler can look healthy by every per-step measure — target
        # acceptance met, no divergences — and still not have explored the
        # posterior, which is what tree-depth saturation produces. R-hat this
        # far from 1 is not "imprecise", it is "the chains sampled different
        # distributions".
        report.fail(
            "chains did not mix",
            "NUTS max R-hat is %.3f (min ESS %.0f): the chains did not explore "
            "the same distribution, so the combined draws are not a posterior "
            "sample. With %.0f%% of draws at the maximum tree depth this means "
            "the step size is far too small for the posterior's extent — the "
            "geometry, not the sampler settings, is the limit.",
            diagnostics["max_rhat"],
            diagnostics["min_ess"],
            100 * diagnostics["treedepth_saturation"],
        )
    report.finish(diagnostics)

    # --- poor but not fatal --------------------------------------------------
    if diagnostics["max_rhat"] > 1.01:
        worst = max(diagnostics["rhat"], key=diagnostics["rhat"].get)
        report.warn(
            "NUTS did not converge: max R-hat = %.4f > 1.01 (worst: %s). Increase "
            "num_warmup/num_samples, or enable `whitening:` to decorrelate the posterior.",
            diagnostics["max_rhat"],
            worst,
        )
    if min(diagnostics["min_ess"], diagnostics["min_ess_tail"]) < 100 * num_chains:
        # Vehtari et al. want >= 100 per chain for both: the bulk number governs
        # the central estimate, the tail one governs the credible interval, and
        # a posterior can pass on one while failing on the other.
        report.warn(
            "Low effective sample size: min bulk ESS = %.0f, min tail ESS = %.0f "
            "over %d chains (want >= 100 per chain for both). Increase "
            "num_samples, or enable `whitening:`.",
            diagnostics["min_ess"],
            diagnostics["min_ess_tail"],
            num_chains,
        )
    if 0 < n_div and diagnostics["divergence_rate"] <= 0.5:
        report.warn(
            "NUTS reported %d divergent transitions (%.2f%% of draws). Raise "
            "target_acceptance_rate towards 0.95, or enable `whitening:`.",
            n_div,
            100 * diagnostics["divergence_rate"],
        )
    if diagnostics["treedepth_saturation"] > 0.2:
        report.warn(
            "NUTS hit the maximum tree depth (%d, i.e. %d leapfrog steps) on "
            "%.0f%% of draws. Each draw then costs the maximum, which is usually "
            "the dominant term in the runtime. This signals a poorly conditioned "
            "posterior rather than a bug; enable/raise `whitening:`, or lower "
            "max_num_doublings to cap the cost per draw.",
            max_num_doublings,
            2**max_num_doublings,
            100 * diagnostics["treedepth_saturation"],
        )

    log.info(
        "NUTS diagnostics: max R-hat = %.4f, min ESS = %.0f, mean acceptance = %.3f, "
        "divergences = %d, step size = %.3e.",
        diagnostics["max_rhat"],
        diagnostics["min_ess"],
        diagnostics["mean_acceptance_rate"],
        n_div,
        min_step,
    )
    cost = (
        f", {diagnostics['ms_per_gradient']:.2f} ms/gradient"
        if "ms_per_gradient" in diagnostics
        else ""
    )
    log.info(
        "NUTS cost: %.0f leapfrog steps per draw (max %d, tree depth %.1f/%d, "
        "%.0f%% saturating), %d gradient evaluations total%s.",
        diagnostics["leapfrogs_per_draw_mean"],
        diagnostics["leapfrogs_per_draw_max"],
        diagnostics["tree_depth_mean"],
        max_num_doublings,
        100 * diagnostics["treedepth_saturation"],
        total_grads,
        cost,
    )
    return diagnostics


def _thin_chains(positions, n_samples):
    """Thin a (C, S, d) block down to at most ``n_samples`` draws, shape (N, d).

    Chains are interleaved before the final truncation so that dropping a
    partial sweep costs every chain equally, rather than gutting the last one.
    """
    num_chains, num_draws, n_dims = positions.shape
    total = num_chains * num_draws
    if total <= n_samples:
        if total < n_samples:
            log.warning(
                "NUTS produced %d draws (num_chains x num_samples), fewer than the "
                "requested n_samples=%d. Using all %d.",
                total,
                n_samples,
                total,
            )
        thinned = positions
    else:
        per_chain = -(-n_samples // num_chains)  # ceil
        stride = max(1, num_draws // per_chain)
        thinned = positions[:, ::stride, :][:, :per_chain, :]
    flat = jnp.swapaxes(thinned, 0, 1).reshape(-1, n_dims)
    return flat[:n_samples]


def run(rng_key, prior, log_likelihood, n_samples, settings):
    """No-U-Turn Hamiltonian Monte Carlo (``blackjax.nuts``).

    Runs ``num_chains`` chains in parallel under ``jax.vmap``, each preceded by
    a ``window_adaptation`` warmup that tunes the step size and a diagonal mass
    matrix. Sampling happens in the prior's unconstrained reparametrisation
    (see `Prior.log_prob_unconstrained`), so bounded priors do not stall the
    integrator at their walls.

    Returns ``logz=None``: NUTS provides no evidence estimate.
    """
    n_dims = len(prior.param_names)
    num_chains = int(settings["num_chains"])
    num_warmup = int(settings["num_warmup"])
    num_draws = int(settings["num_samples"])

    if num_chains * num_draws * n_dims > 5e7:
        log.warning(
            "NUTS will hold %d x %d x %d draws in memory (~%.1f GB in float64). "
            "Reduce num_chains or num_samples if this is too much.",
            num_chains,
            num_draws,
            n_dims,
            8 * num_chains * num_draws * n_dims / 1e9,
        )

    @jax.jit
    def logdensity(u):
        return prior.log_prob_unconstrained(u) + log_likelihood(
            prior.from_unconstrained(u)
        )

    rng_key, init_key = jax.random.split(rng_key)
    # One over-dispersed prior draw per chain, in unconstrained space, shape
    # (C, d). Over-dispersed starts are what make R-hat meaningful: chains
    # started at a common point agree by construction and hide non-convergence.
    u0 = prior.sample_unconstrained(init_key, num_chains)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=float(settings["target_acceptance_rate"]),
        max_num_doublings=int(settings["max_num_doublings"]),
    )

    max_doublings = int(settings["max_num_doublings"])

    def _run_warmup(key, u_init):
        adapt, _ = warmup.run(key, u_init, num_steps=num_warmup)

        return (
            adapt.state,
            adapt.parameters["step_size"],
            adapt.parameters["inverse_mass_matrix"],
        )

    def _run_sampling(key, state, step_size, inverse_mass_matrix):
        kernel = blackjax.nuts(
            logdensity,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            max_num_doublings=max_doublings,
        )
        # The custom transform is what keeps memory bounded: the default keeps
        # whole NUTSState/NUTSInfo trees, momenta included, for every step.
        _, history = run_inference_algorithm(
            rng_key=key,
            inference_algorithm=kernel,
            num_steps=num_draws,
            initial_state=state,
            transform=lambda state, info: (
                state.position,
                state.logdensity,
                info.num_integration_steps,
                info.num_trajectory_expansions,
                info.is_divergent,
                info.acceptance_rate,
            ),
        )
        return history

    log.info(
        "Compiling and running %d NUTS chains (%d warmup + %d samples each) over "
        "%d parameters...",
        num_chains,
        num_warmup,
        num_draws,
        n_dims,
    )

    warmup_key, sample_key = jax.random.split(rng_key)

    t0 = time.time()
    warm_states, step_sizes, inverse_mass_matrices = jax.block_until_ready(
        jax.vmap(_run_warmup)(jax.random.split(warmup_key, num_chains), u0)
    )
    warmup_seconds = time.time() - t0
    log.info(
        "NUTS warmup finished in %.2f minutes (%d steps x %d chains); adapted "
        "step size %.3e.",
        warmup_seconds / 60.0,
        num_warmup,
        num_chains,
        min(float(s) for s in step_sizes),
    )

    t0 = time.time()
    (
        positions,
        logdensity_draws,
        leapfrogs,
        tree_depth,
        is_divergent,
        acceptance,
    ) = jax.block_until_ready(
        jax.vmap(_run_sampling)(
            jax.random.split(sample_key, num_chains),
            warm_states,
            step_sizes,
            inverse_mass_matrices,
        )
    )
    sampling_seconds = time.time() - t0
    log.info(
        "NUTS sampling finished in %.2f minutes (%d draws x %d chains).",
        sampling_seconds / 60.0,
        num_draws,
        num_chains,
    )

    t0 = time.time()
    diagnostics = _nuts_diagnostics(
        prior,
        positions,
        leapfrogs,
        tree_depth,
        is_divergent,
        acceptance,
        step_sizes,
        int(settings["max_num_doublings"]),
        sampling_seconds=sampling_seconds,
        warmup_seconds=warmup_seconds,
    )
    log.info("NUTS: diagnostics computed in %.1f s.", time.time() - t0)

    t0 = time.time()
    posterior_free = jax.block_until_ready(
        jax.vmap(prior.from_unconstrained)(_thin_chains(positions, n_samples))
    )
    log.info("NUTS: posterior draws mapped in %.1f s.", time.time() - t0)

    # Best point over the FULL chains, not just the thinned draws — nested
    # sampling likewise maximises over all its live and dead points.
    all_u = positions.reshape(-1, n_dims)
    logl = jax.block_until_ready(
        logdensity_draws.reshape(-1) - jax.vmap(prior.log_prob_unconstrained)(all_u)
    )
    best_index = int(jnp.argmax(logl))
    best_point = jax.block_until_ready(prior.from_unconstrained(all_u[best_index]))

    log_dir = settings["log_dir"]
    pd.DataFrame(posterior_free, columns=prior.param_names).to_csv(
        os.path.join(log_dir, "nuts_samples.csv"), index=False
    )
    _write_diagnostics(log_dir, "nuts_diagnostics.json", diagnostics)

    log.info("NUTS provides no evidence estimate; FitResult.logz is None.")

    return SamplerOutput(
        samples=posterior_free,
        best_point=best_point,
        max_loglikelihood=float(logl[best_index]),
        logz=None,
        diagnostics=diagnostics,
    )
