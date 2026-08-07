"""
smefit.blackjax_samplers

Per-algorithm BlackJAX runners behind ``blackjax_settings.algorithm``.

This is a plain helper module, deliberately NOT registered in
``smefit.app.smefit_providers``: none of its functions are reportengine nodes,
and registering it would publish every public function into the generated
``actions.md`` and every defaulted parameter into the legal runcard keys.

It is imported by ``smefit.blackjax_fit`` (the provider) and by
``smefit.config`` (for the settings key ownership map below), so it must not
import either of them back.
"""

import dataclasses
import json
import logging
import os
import time
from typing import Optional

import anesthetic
import blackjax
import jax
import jax.numpy as jnp
import pandas as pd
import tqdm
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from blackjax.ns.utils import ess, finalise, log_weights, sample
from blackjax.util import run_inference_algorithm
from jax.scipy.special import logsumexp

from smefit.priors import UnconstrainedPrior

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SamplerOutput:
    """Algorithm-independent result of one BlackJAX run.

    Attributes
    ----------
    samples : jnp.ndarray, shape (n_draws, n_free)
        Posterior draws in SAMPLER space (whitened coordinates when whitening
        is active), already thinned/truncated to at most ``n_samples``.
    best_point : jnp.ndarray, shape (n_free,)
        Sampler-space point of maximum LIKELIHOOD (not maximum posterior).
    max_loglikelihood : float
        Maximum of ``-chi2/2`` over the sampler's draws. ``FitResult.chi2_val``
        is ``-2 * max_loglikelihood``, so this must never include the prior term.
    logz : float or None
        Log evidence; None for algorithms that do not estimate one.
    diagnostics : dict
        JSON-serialisable summary, written to ``log_dir`` and logged.
    """

    samples: jnp.ndarray
    best_point: jnp.ndarray
    max_loglikelihood: float
    logz: Optional[float] = None
    diagnostics: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Nested sampling
# ---------------------------------------------------------------------------


def _run_nested_sampling(
    rng_key, prior, log_likelihood, n_samples, settings, init_point
):
    """BlackJAX nested sampling (``blackjax.nss``).

    The only algorithm here that estimates the log evidence. ``init_point`` is
    unused: nested sampling starts from live points drawn from the prior.
    """
    n_dims = len(prior.param_names)
    n_live = settings["n_live"]
    n_delete = int(settings["delete_fraction"] * n_live)

    inital_particles = prior.sample(rng_key, n_live)

    algo = blackjax.nss(
        logprior_fn=prior.log_prob,
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=int(settings["repeats"] * n_dims),
    )

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    state = algo.init(inital_particles)

    dead = []

    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not (
            state.integrator.logZ_live - state.integrator.logZ
            < settings["log_precision"]
        ):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    final_states = finalise(state, dead)
    rng_key, ess_key, weights_key, sample_key = jax.random.split(rng_key, 4)

    ess_value = int(ess(ess_key, final_states))
    logw = log_weights(rng_key, final_states)
    logzs = logsumexp(logw, axis=0)
    full_samples = sample(sample_key, final_states, ess_value).position

    # Get number of posterior samples to resample
    n_posterior_samples = n_samples

    # Check if we have enough samples
    if n_posterior_samples > full_samples.shape[0]:
        n_posterior_samples = full_samples.shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior "
            f"samples computed by BlackJAX. Setting the number of resampled posterior "
            f"samples to {n_posterior_samples}"
        )

    # Take first n_posterior_samples samples from the full posterior samples for resampling
    posterior_free = full_samples[:n_posterior_samples]

    # write out an anesthetic dataframe
    nested_samples = anesthetic.NestedSamples(
        data=final_states.particles.position,
        logL=final_states.particles.loglikelihood,
        logL_birth=final_states.particles.loglikelihood_birth,
        columns=prior.param_names,
    )
    log_dir = settings["log_dir"]
    nested_samples.to_csv(log_dir + "/nested_samples.csv")

    # Compute bayesian metrics (similar to UltraNest)
    # Find maximum likelihood point
    best_free_index = jnp.argmax(final_states.particles.loglikelihood)
    max_logl = float(final_states.particles.loglikelihood[best_free_index])
    best_free = final_states.particles.position[best_free_index]

    return SamplerOutput(
        samples=posterior_free,
        best_point=best_free,
        max_loglikelihood=max_logl,
        logz=float(logzs.mean()),
        diagnostics={
            "algorithm": "nested_sampling",
            "n_live": int(n_live),
            "n_dead": len(dead) * n_delete,
            "ess": int(ess_value),
            "logz": float(logzs.mean()),
            "logz_std": float(logzs.std()),
        },
    )


# ---------------------------------------------------------------------------
# NUTS
# ---------------------------------------------------------------------------


def _nuts_initial_points(rng_key, uprior, settings, init_point, num_chains, n_dims):
    """Starting positions for the chains, in unconstrained space, shape (C, d).

    ``init: prior`` draws one over-dispersed prior sample per chain, which is
    what makes R-hat meaningful — chains started at a common point agree by
    construction and hide non-convergence. ``init: baseline`` starts every
    chain at ``init_point`` (sampler space) plus a small jitter, as an escape
    hatch for badly-scaled un-whitened problems where a prior draw lands at an
    astronomically large chi2.
    """
    if settings.get("init", "prior") == "prior":
        return uprior.sample_unconstrained(rng_key, num_chains)
    u_base = uprior.to_unconstrained(jnp.asarray(init_point))
    return u_base + 0.1 * jax.random.normal(rng_key, (num_chains, n_dims))


def _nuts_diagnostics(prior, positions, is_divergent, acceptance, step_sizes):
    """Convergence summary for a (C, S, d) block of NUTS draws, and its warnings."""
    rhat = potential_scale_reduction(positions, chain_axis=0, sample_axis=1)
    n_eff = effective_sample_size(positions, chain_axis=0, sample_axis=1)
    num_chains, num_draws, _ = positions.shape
    total = num_chains * num_draws
    n_div = int(jnp.sum(is_divergent))

    diagnostics = {
        "algorithm": "nuts",
        "num_chains": int(num_chains),
        "num_samples": int(num_draws),
        "rhat": {name: float(rhat[i]) for i, name in enumerate(prior.param_names)},
        "ess": {name: float(n_eff[i]) for i, name in enumerate(prior.param_names)},
        "max_rhat": float(jnp.max(rhat)),
        "min_ess": float(jnp.min(n_eff)),
        "divergences": n_div,
        "divergence_rate": n_div / total,
        "mean_acceptance_rate": float(jnp.mean(acceptance)),
        "step_size": [float(s) for s in step_sizes],
    }

    if diagnostics["max_rhat"] > 1.01:
        worst = max(diagnostics["rhat"], key=diagnostics["rhat"].get)
        log.warning(
            "NUTS did not converge: max R-hat = %.4f > 1.01 (worst: %s). Increase "
            "num_warmup/num_samples, or enable `whitening:` to decorrelate the posterior.",
            diagnostics["max_rhat"],
            worst,
        )
    if diagnostics["min_ess"] < 100 * num_chains:
        log.warning(
            "Low effective sample size: min ESS = %.0f over %d chains (want >= 100 "
            "per chain). Increase num_samples, or enable `whitening:`.",
            diagnostics["min_ess"],
            num_chains,
        )
    if n_div > 0:
        log.warning(
            "NUTS reported %d divergent transitions (%.2f%% of draws). Raise "
            "target_acceptance_rate towards 0.95, or enable `whitening:`.",
            n_div,
            100 * diagnostics["divergence_rate"],
        )
    log.info(
        "NUTS diagnostics: max R-hat = %.4f, min ESS = %.0f, mean acceptance = %.3f, "
        "divergences = %d.",
        diagnostics["max_rhat"],
        diagnostics["min_ess"],
        diagnostics["mean_acceptance_rate"],
        n_div,
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


def _run_nuts(rng_key, prior, log_likelihood, n_samples, settings, init_point):
    """No-U-Turn Hamiltonian Monte Carlo (``blackjax.nuts``).

    Runs ``num_chains`` chains in parallel under ``jax.vmap``, each preceded by
    a ``window_adaptation`` warmup that tunes the step size and a diagonal mass
    matrix. Sampling happens in an unconstrained space (see `UnconstrainedPrior`),
    so bounded priors do not stall the integrator at their walls.

    Returns ``logz=None``: NUTS provides no evidence estimate.
    """
    # Raised before anything is compiled, so a bad runcard fails in under a second.
    uprior = UnconstrainedPrior(prior)

    n_dims = len(prior.param_names)
    num_chains = int(settings["num_chains"])
    num_warmup = int(settings["num_warmup"])
    num_draws = int(settings["num_samples"])

    if not jax.config.jax_enable_x64:
        log.warning(
            "Running NUTS in float32 (-f32). Gradient MCMC is prone to divergences "
            "and NaN step sizes at this precision; prefer the default float64."
        )
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
        return uprior.log_prob_unconstrained(u) + log_likelihood(
            uprior.from_unconstrained(u)
        )

    rng_key, init_key = jax.random.split(rng_key)
    u0 = _nuts_initial_points(
        init_key, uprior, settings, init_point, num_chains, n_dims
    )

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=float(settings["target_acceptance_rate"]),
        max_num_doublings=int(settings["max_num_doublings"]),
    )

    def _run_chain(key, u_init):
        warmup_key, sample_key = jax.random.split(key)
        adapt, _ = warmup.run(warmup_key, u_init, num_steps=num_warmup)
        kernel = blackjax.nuts(logdensity, **adapt.parameters)
        # The custom transform is what keeps memory bounded: the default keeps
        # whole NUTSState/NUTSInfo trees, momenta included, for every step.
        _, history = run_inference_algorithm(
            rng_key=sample_key,
            inference_algorithm=kernel,
            num_steps=num_draws,
            initial_state=adapt.state,
            transform=lambda state, info: (
                state.position,
                info.is_divergent,
                info.acceptance_rate,
            ),
        )
        return (*history, adapt.parameters["step_size"])

    log.info(
        "Compiling and running %d NUTS chains (%d warmup + %d samples each) over "
        "%d parameters...",
        num_chains,
        num_warmup,
        num_draws,
        n_dims,
    )
    t0 = time.time()
    # If vmap over the warmup ever breaks, a Python loop over chains is
    # numerically identical at num_chains times the wall clock.
    positions, is_divergent, acceptance, step_sizes = jax.vmap(_run_chain)(
        jax.random.split(rng_key, num_chains), u0
    )
    log.info("NUTS sampling finished in %.2f minutes.", (time.time() - t0) / 60.0)

    diagnostics = _nuts_diagnostics(
        prior, positions, is_divergent, acceptance, step_sizes
    )

    posterior_free = jax.vmap(uprior.from_unconstrained)(
        _thin_chains(positions, n_samples)
    )

    # Best point over the FULL chains, not just the thinned draws — nested
    # sampling likewise maximises over all its live and dead points.
    all_x = jax.vmap(uprior.from_unconstrained)(positions.reshape(-1, n_dims))
    logl = jax.vmap(log_likelihood)(all_x)
    best_index = int(jnp.argmax(logl))

    log_dir = settings["log_dir"]
    pd.DataFrame(posterior_free, columns=prior.param_names).to_csv(
        os.path.join(log_dir, "nuts_samples.csv"), index=False
    )
    with open(os.path.join(log_dir, "nuts_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    log.info("NUTS provides no evidence estimate; FitResult.logz is None.")

    return SamplerOutput(
        samples=posterior_free,
        best_point=all_x[best_index],
        max_loglikelihood=float(logl[best_index]),
        logz=None,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SAMPLER_REGISTRY = {
    "nested_sampling": _run_nested_sampling,
    "nuts": _run_nuts,
}

BJ_ALGORITHMS = tuple(_SAMPLER_REGISTRY)

#: Accepted values of ``blackjax_settings.init`` — see ``_nuts_initial_points``.
BJ_INIT_MODES = frozenset({"prior", "baseline"})

#: Keys of ``blackjax_settings`` that every algorithm uses.
BJ_SHARED_SETTINGS = frozenset({"algorithm", "seed", "log_dir"})

#: Keys of ``blackjax_settings`` owned by one algorithm. Consumed by
#: ``smefitConfig.parse_blackjax_settings`` to warn about settings that the
#: selected algorithm will ignore. Keep in step with the literal ``known_keys``
#: set there — ``tests/test_config.py`` enforces that.
BJ_ALGORITHM_SETTINGS = {
    "nested_sampling": frozenset(
        {"n_live", "repeats", "delete_fraction", "log_precision"}
    ),
    "nuts": frozenset(
        {
            "num_chains",
            "num_warmup",
            "num_samples",
            "target_acceptance_rate",
            "max_num_doublings",
            "init",
        }
    ),
}

assert set(BJ_ALGORITHM_SETTINGS) == set(_SAMPLER_REGISTRY)


def get_sampler(algorithm):
    """Look up the runner for *algorithm*."""
    if algorithm not in _SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown BlackJAX algorithm '{algorithm}'. "
            f"Available: {sorted(_SAMPLER_REGISTRY)}"
        )
    return _SAMPLER_REGISTRY[algorithm]
