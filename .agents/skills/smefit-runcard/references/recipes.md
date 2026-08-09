# Advanced runcard recipes

Hand-written companion to the auto-generated `runcard-keys.md` (authoritative
for key names/defaults). Each recipe shows the intent behind a block and the
pitfalls that key tables cannot convey.

## RGE running

Wilson coefficients defined at a high scale are evolved to the observable
scales before predictions are computed.

```yaml
rge:
  init_scale: 10000.0      # GeV — scale where the coefficients are defined (required)
  obs_scale: dynamic       # per-data-point scales from the theory files, or a fixed float
  smeft_accuracy: integrate  # 'integrate' (exact) or 'leadinglog' (faster)
  yukawa: top              # 'top', 'full', or 'none'
  adm_QCD: false           # true = QCD-only anomalous dimensions (EW couplings zeroed)
```

- Computing the RGE matrix can be slow; it is saved automatically under the
  run's output directory. To reuse one across runs, point `rg_matrix:` at the
  saved pickle — preferably via the `smefit_results` prefix:
  `rg_matrix: smefit_results/fits/<fit_name>/rge_matrix.pkl`. If the fit is not
  present locally, smefit automatically tries to download it from the server
  before failing.
- Drop the whole block only when coefficients are defined directly at the
  observable scale.

## Whitening (Hessian-preconditioned sampling)

```yaml
whitening:
  sigma_prior: 5.0
  eps: 1e-8
  shift: baseline           # 'baseline' (default) or 'gradient_descent'
```

Replaces per-coefficient priors: sampling happens in the whitened space where
the chi2 Hessian at the shift point is the identity, with a
`uniform[-sigma_prior, sigma_prior]` prior per direction. Useful when
coefficients have wildly different sensitivities (badly conditioned
posteriors). Individual `prior:` entries on free coefficients are ignored
while whitening is active, but free coefficients still need `free: True`.
Incompatible with `bayesian_update_path`.

`shift` picks the point the whitening transform is centred on:
- `baseline` (default) — the coefficients' baseline point (`baseline_value`,
  zero unless set).
- `gradient_descent` — the `gd_best_fit` point instead; this additionally
  requires a `gradient_descent_settings` block in the runcard, since it makes
  the graph depend on `gd_best_fit`.

## Choosing a BlackJAX algorithm

`run_blackjax_fit` names the backend, not the algorithm. Which sampler runs is
`blackjax_settings.algorithm`:

```yaml
blackjax_settings:
  algorithm: nuts            # nested_sampling (default) or nuts
  num_chains: 4
  num_warmup: 1000
  num_samples: 2500          # PER CHAIN; thinned down to the top-level n_samples
  target_acceptance_rate: 0.8
```

Pick `nested_sampling` when you need the log evidence for model comparison,
when the posterior may be multimodal, or when the runcard uses
`bayesian_update_path` (whose exact-posterior prior has no per-parameter
bijectors, so `nuts` refuses it).

Pick `nuts` for smooth, unimodal, high-dimensional posteriors: it exploits the
JAX gradient of the chi2 and typically reaches a given effective sample size far
faster than nested sampling. It writes `"logz": null` — use `bic`/`aic` for
model comparison instead. Pair it with `whitening:` (the posterior it explores
is then decorrelated and unit-scale) and keep the default float64 precision;
gradient MCMC under `-f32` is prone to divergences.

Uniform priors — including the `uniform[-sigma_prior, sigma_prior]` that
`whitening:` imposes — are sampled through a logit bijector, so prior bounds
never stall the sampler at a wall. No runcard change is needed for that.

After a `nuts` run, read `<output>/blackjax_logs/nuts_diagnostics.json`. Start
with `converged`: when it is `false` the run failed outright and the posterior
is meaningless — the log carries an `ERROR` naming which check tripped
(step-size collapse, >50% divergences, or ~zero acceptance). Otherwise:
- `max_rhat` >= 1.01 → chains have not mixed. Raise `num_warmup`/`num_samples`,
  or enable `whitening:`.
- `divergences` > 0 → the step size is too large for the posterior's curvature.
  Raise `target_acceptance_rate` towards 0.95, or enable `whitening:`.
- `min_ess` (bulk) or `min_ess_tail` below ~100 per chain → correlated draws;
  same remedies. Bulk governs the central estimate and tail the credible
  interval, so a run can pass one and fail the other. A posterior pressed
  against a prior bound also shows up here, and is fixed by widening the prior
  (or `sigma_prior`).

`rhat` and `ess` are the rank-normalised split-chain versions (Vehtari et al.
2021), which is what the 1.01 threshold is calibrated for; they detect drift
*within* a chain, which the classic Gelman-Rubin statistic cannot.

A `nested_sampling` run writes `<output>/blackjax_logs/nested_diagnostics.json`,
with the same `converged`-first layout:
- `d_G` (Bayesian model dimensionality) counts the directions the *likelihood*
  constrains. Well below `n_free` means flat directions — the same pathology
  `whitening:` reports from the Hessian, measured after the fit instead of
  before it.
- `D_KL` is the prior→posterior compression in nats, and `n_live * D_KL`
  (`expected_n_dead`) is how long the run should take. `n_dead` far below it
  means the run stopped early; lower `log_precision`.
- `logz_std` above ~1 nat means the evidence cannot support model comparison —
  raise `n_live`, since the error scales as `sqrt(D_KL/n_live)`.
- `ess_posterior` below the requested `n_samples` caps how many draws are
  stored (`n_stored`); raise `n_live` or `repeats`. It is named apart from the
  NUTS `ess` on purpose — that one is a per-parameter dict, this is a single
  count of effective weighted particles.

There is no insertion-index test (the nested-sampling analogue of R-hat):
blackjax does not expose the insertion index of replacement live points.

**Why a NUTS fit takes as long as it does.** The runtime is essentially

    time  =  draws  x  leapfrogs_per_draw  x  ms_per_gradient

and all three are reported. `ms_per_gradient` is a property of your likelihood
(dataset count, `use_quad`, RGE, external chi2) and is not something the sampler
can improve. `leapfrogs_per_draw_mean` is the multiplier that decides whether a
fit takes minutes or hours: a well-conditioned posterior needs 8-64 steps, while
`treedepth_saturation` near 1 means every draw is paying the
`2**max_num_doublings` maximum. Saturation is a statement about the posterior's
geometry, not a bug — enable or strengthen `whitening:`, or lower
`max_num_doublings` to cap the cost per draw at the price of shorter moves.

A whitening block on a badly conditioned Hessian is the usual culprit: watch for
the "regularised Hessian is ill-conditioned" warning and the count of
unconstrained directions in the `Hessian whitening:` line. Directions the data
does not constrain get their scale from `whitening.eps` rather than from the
fit, which is exactly the geometry gradient samplers handle worst. Nested
sampling is the more robust choice there.

## Sequential Bayesian updating

```yaml
bayesian_update_path: /path/to/previous_fit_output
```

Uses the exact posterior of a previous fit as the prior of this one. The
directory must contain `fit_results.json` and `input/runcard.yaml` (both are
written automatically by every fit run). Datasets already used in the previous
fit must NOT be repeated in this runcard, or they are double-counted.

## External chi2 modules

For likelihoods that cannot be expressed as commondata+theory files
(e.g. Drell-Yan with analytic likelihoods, optimal observables at future
colliders). Modules live in an `external_chi2/` folder (in the new_smefit repo
and in smefit_database — reference them with the matching path prefix).

```yaml
external_chi2:
  CMS_DYMee_13TeV:                     # class name inside the module
    path: new_smefit/external_chi2/drell_yan/CMS_DYMee_13TeV.py  # prefix-resolved
    use_quad: True                     # extra keys are forwarded to the class
    order: NLO_QCD
    group: drell_yan                   # optional: report/Fisher grouping (stripped before forwarding)
```

`path` and `rg_matrix` entries are prefix-resolved like all runcard paths
(`new_smefit/...`, `smefit_database/...`, `smefit_results/...`); an `rg_matrix`
under `smefit_results/` is auto-downloaded from the server when missing.

Contract for the class: constructor accepts `coefficients=`, `rge_dict=`, plus
the extra keys verbatim; instance exposes `compute_chi2(coeffs)`, `num_data`,
and `param_names`. A runcard may have `external_chi2` with no `datasets:` at
all — then the fit runs on external likelihoods alone.

## Pseudodata / projections

```yaml
pseudodata_settings:
  lumi_new: 3000          # rescale statistical errors to this luminosity (fb^-1)
  noise: L0               # L0 = central values on theory, L1 = fluctuated
  seed: 42                # only used for L1
  fred_tot: 0.5           # reduce total systematics by this factor
  fred_sys: 1.0           # reduce correlated systematics by this factor
actions_:
  - write_pseudodata
```

Writes projected datasets to `<output>/pseudodata/*.yaml` (plus theory JSON
copies for `_proj` datasets), ready to be used as `data_path` in a future-
collider fit.

## Reports and Fisher information

Use a `template_text` block with `{@action@}` tags and the `report` action:

```yaml
template_text: |
  # Fisher information
  {@fisher_diagonals_normalised@}
  {@plot_fisher_diagonals_heatmap@}
actions_:
  - report
```

- The Fisher chain evaluates at the gradient-descent best fit, so
  `optimizer_settings` / `gradient_descent_settings` apply (defaults are fine;
  `gradient_descent_settings: {sm_solution: True}` skips optimization and uses
  the SM point).
- Add `group: <label>` to dataset entries (and external chi2 blocks) to
  aggregate Fisher matrices per group instead of per dataset.

## Precision and performance

- `smefit <runcard> -f32` switches JAX to float32 — faster, but check
  convergence.
- `chi2_timing` as an action benchmarks the likelihood before committing to an
  expensive sampler; `time_likelihood.yaml` is the template.
- For samplers, start from the template settings; increase
  `min_num_live_points` / `n_live` only when posteriors look under-resolved.
