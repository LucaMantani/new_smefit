"""
smefit.blackjax_samplers.nested_sampling

The ``nested_sampling`` algorithm behind ``blackjax_settings.algorithm``:
BlackJAX nested sampling, the only algorithm here that estimates the log
evidence.

A plain helper module, NOT registered in ``smefit.app.smefit_providers``. It
must not import ``smefit.blackjax_fit`` or ``smefit.config``, which import the
registry that imports this.
"""

import logging
import math
import time

import anesthetic
import blackjax
import jax
import jax.numpy as jnp
import tqdm
from blackjax.ns.utils import ess, finalise, log_weights, sample
from jax.scipy.special import logsumexp

from smefit.blackjax_samplers._common import (
    SamplerOutput,
    _HealthReport,
    _write_diagnostics,
)

log = logging.getLogger(__name__)

#: Keys of ``blackjax_settings`` this algorithm owns.
SETTINGS = frozenset({"n_live", "repeats", "delete_fraction", "log_precision"})

#: Below this many effectively independent draws the posterior is a handful of
#: distinct points, whatever n_samples was requested.
_MIN_USABLE_ESS = 50

#: Evidence uncertainty (nats) above which logZ cannot support model comparison:
#: differences smaller than this are noise.
_LOGZ_STD_LIMIT = 1.0


def _nested_sampling_diagnostics(
    nested_samples,
    n_free,
    n_live,
    n_dead,
    ess_value,
    n_requested,
    n_stored,
    logzs,
    termination_margin,
    log_precision,
    sampling_seconds=None,
):
    """Convergence and cost summary for a nested-sampling run.

    Mirrors ``_nuts_diagnostics``: assembles the JSON-serialisable summary and
    decides ``diagnostics["converged"]``, which is False only when the run
    failed outright rather than merely being imprecise.

    The Handley-Lemos statistics come from anesthetic, which recomputes them on
    the same NestedSamples object already built for the CSV, so they are close
    to free. ``d_G`` (Bayesian model dimensionality) counts the directions the
    data actually constrains: well below ``n_free`` means flat directions — the
    same pathology ``smefit.whitening`` reports from the Hessian, measured here
    independently and after the fit rather than before it.

    Note there is no insertion-index test (Fowlie et al. 2020), the nested
    sampling analogue of R-hat: blackjax's ``NSInfo`` does not expose the
    insertion index of replacement live points.
    """
    stats = nested_samples.stats(nsamples=50)

    def _stat(name):
        col = stats[name]
        return float(col.mean()), float(col.std())

    logz, logz_std = _stat("logZ")
    d_kl, d_kl_std = _stat("D_KL")
    logl_p, _ = _stat("logL_P")
    d_g, d_g_std = _stat("d_G")

    diagnostics = {
        "algorithm": "nested_sampling",
        "n_live": int(n_live),
        "n_dead": int(n_dead),
        "n_free": int(n_free),
        "logz": float(jnp.mean(logzs)),
        "logz_std": float(jnp.std(logzs)),
        # anesthetic's own estimate, from the same draws by a different route
        "logz_anesthetic": logz,
        "logz_anesthetic_std": logz_std,
        "D_KL": d_kl,
        "D_KL_std": d_kl_std,
        "logL_P": logl_p,
        "d_G": d_g,
        "d_G_std": d_g_std,
        # Named apart from the NUTS "ess": that one is a per-parameter dict of
        # autocorrelation-based sample sizes, this is a single count of
        # effective weighted particles.
        "ess_posterior": int(ess_value),
        # Fraction of dead points that survive as independent posterior draws.
        "ess_efficiency": float(ess_value) / n_dead if n_dead else 0.0,
        "n_requested": int(n_requested),
        # Draws actually stored: the importance resample yields only
        # ess_posterior particles, so asking for more than that returns fewer.
        "n_stored": int(n_stored),
        # How far past the stopping rule the run went; <= log_precision means
        # the loop exited on its own terms rather than on an iteration cap.
        "termination_margin": float(termination_margin),
        "log_precision": float(log_precision),
        # n_live * D_KL is the expected number of iterations to compress the
        # prior down to the posterior; far fewer means the run stopped early.
        "expected_n_dead": float(n_live * d_kl),
    }
    if sampling_seconds is not None:
        diagnostics["sampling_seconds"] = float(sampling_seconds)

    # --- outright failure: the posterior is unusable ------------------------
    report = _HealthReport("Nested sampling", log)
    if ess_value < _MIN_USABLE_ESS:
        report.fail(
            "no effective samples",
            "Nested sampling produced an effective sample size of %d (< %d): the "
            "posterior consists of a handful of distinct points, so any interval "
            "from it is meaningless. Raise n_live, or raise `repeats` so the "
            "inner MCMC decorrelates the live points.",
            ess_value,
            _MIN_USABLE_ESS,
        )
    if not math.isfinite(diagnostics["logz"]):
        report.fail(
            "non-finite evidence",
            "Nested sampling returned a non-finite log evidence: the likelihood "
            "returned NaN/inf somewhere in the prior's support.",
        )
    report.finish(diagnostics)

    # --- poor but not fatal --------------------------------------------------
    if diagnostics["logz_std"] > _LOGZ_STD_LIMIT:
        report.warn(
            "Log evidence is uncertain: logZ = %.2f +/- %.2f nats. Model "
            "comparison needs this error well below the logZ difference of "
            "interest; raise n_live (the error scales as sqrt(D_KL/n_live)).",
            diagnostics["logz"],
            diagnostics["logz_std"],
        )
    if d_g < n_free - 1.0:
        report.warn(
            "Bayesian model dimensionality d_G = %.1f +/- %.1f over %d free "
            "coefficients: roughly %.0f directions are unconstrained by the "
            "data. Expect a posterior that follows the prior along them.",
            d_g,
            d_g_std,
            n_free,
            n_free - d_g,
        )
    if n_stored < n_requested:
        report.warn(
            "Effective sample size %d is below the requested n_samples = %d, so "
            "only %d posterior samples were stored. Raise n_live, or raise "
            "`repeats` so the inner MCMC decorrelates the live points.",
            ess_value,
            n_requested,
            n_stored,
        )
    if n_dead < 0.5 * diagnostics["expected_n_dead"]:
        report.warn(
            "Nested sampling stopped after %d dead points, well short of the "
            "~%.0f (n_live x D_KL) expected to compress the prior to this "
            "posterior. The evidence is likely underestimated; lower "
            "`log_precision`.",
            n_dead,
            diagnostics["expected_n_dead"],
        )

    log.info(
        "Nested sampling diagnostics: logZ = %.2f +/- %.2f, D_KL = %.2f nats, "
        "d_G = %.1f/%d, ESS = %d (%.0f%% of %d dead points).",
        diagnostics["logz"],
        diagnostics["logz_std"],
        d_kl,
        d_g,
        n_free,
        ess_value,
        100 * diagnostics["ess_efficiency"],
        n_dead,
    )
    return diagnostics


def run(rng_key, prior, log_likelihood, n_samples, settings):
    """BlackJAX nested sampling (``blackjax.nss``).

    The only algorithm here that estimates the log evidence.
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

    t0 = time.time()
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not (
            state.integrator.logZ_live - state.integrator.logZ
            < settings["log_precision"]
        ):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)
    sampling_seconds = time.time() - t0
    # Evidence still sitting in the live points when the loop stopped; the
    # stopping rule is this dropping below log_precision.
    termination_margin = float(state.integrator.logZ_live - state.integrator.logZ)

    final_states = finalise(state, dead)
    rng_key, ess_key, weights_key, sample_key = jax.random.split(rng_key, 4)

    ess_value = int(ess(ess_key, final_states))
    logw = log_weights(rng_key, final_states)
    logzs = logsumexp(logw, axis=0)
    full_samples = sample(sample_key, final_states, ess_value).position

    # Get number of posterior samples to resample
    n_posterior_samples = n_samples

    # The importance resample above yields `ess_value` particles, so a larger
    # request simply cannot be met; _nested_sampling_diagnostics warns about it.
    n_posterior_samples = min(n_posterior_samples, full_samples.shape[0])

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

    diagnostics = _nested_sampling_diagnostics(
        nested_samples,
        n_free=len(prior.param_names),
        n_live=n_live,
        n_dead=len(dead) * n_delete,
        ess_value=ess_value,
        n_requested=n_samples,
        n_stored=n_posterior_samples,
        logzs=logzs,
        termination_margin=termination_margin,
        log_precision=settings["log_precision"],
        sampling_seconds=sampling_seconds,
    )
    _write_diagnostics(log_dir, "nested_diagnostics.json", diagnostics)

    return SamplerOutput(
        samples=posterior_free,
        best_point=best_free,
        max_loglikelihood=max_logl,
        logz=float(logzs.mean()),
        diagnostics=diagnostics,
    )
