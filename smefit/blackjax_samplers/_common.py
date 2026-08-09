"""
smefit.blackjax_samplers._common

Pieces shared by every BlackJAX sampler runner: the result container and the
health-verdict scaffolding.

Like the per-algorithm modules, this is a plain helper module and is NOT
registered in ``smefit.app.smefit_providers`` — none of its contents are
reportengine nodes.
"""

import dataclasses
import json
import logging
import os
from typing import Optional

import jax.numpy as jnp

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
# Shared diagnostics scaffolding
# ---------------------------------------------------------------------------
#
# The per-algorithm diagnostics have almost nothing in common as *numbers* —
# NUTS asks whether the chains explored the target, nested sampling asks
# whether the prior was compressed onto the posterior and the evidence
# converged, and no statistic transfers between them. What they do share is the
# verdict machinery below: accumulate fatal and soft findings, decide
# ``converged``, and say so once at the end.


class _HealthReport:
    """Collects a run's findings and turns them into a ``converged`` verdict.

    ``fail`` marks the posterior unusable (an error the user must act on);
    ``warn`` marks it imprecise but real. Keeping the two apart is the whole
    point: "the draws are not from the target distribution" and "you could use
    more of them" call for different reactions.
    """

    def __init__(self, label, logger=log):
        self.label = label
        # Log through the algorithm's own logger, so a message about NUTS is
        # still attributed to smefit.blackjax_samplers.nuts rather than to this module.
        self.log = logger
        self.failures = []

    def fail(self, name, msg, *args):
        self.failures.append(name)
        self.log.error(msg, *args)

    def warn(self, msg, *args):
        self.log.warning(msg, *args)

    def finish(self, diagnostics):
        """Stamp the verdict onto *diagnostics* and announce a failed run."""
        diagnostics["converged"] = not self.failures
        if self.failures:
            self.log.error(
                "%s run FAILED (%s) — do not use this posterior. The draws and "
                "diagnostics have still been written to the log directory for "
                "debugging.",
                self.label,
                ", ".join(self.failures),
            )
        return diagnostics


def _write_diagnostics(log_dir, filename, diagnostics):
    """Persist a diagnostics dict next to the algorithm's draws."""
    with open(os.path.join(log_dir, filename), "w") as f:
        json.dump(diagnostics, f, indent=2)
