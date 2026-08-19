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
Incompatible with `bayesian_update`.

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
bayesian_update:
  name: my_previous_fit          # the fit directory name
  path: smefit_results/fits      # optional: where to look for it (this is the default)
```

The shorthand `bayesian_update: my_previous_fit` is equivalent. The fit is
looked up exactly like a `fits:` entry — under `smefit_results/fits/` unless
`path` says otherwise, and downloaded from the server if it is not there yet.

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

## chi2 scans

Two different scans share one settings block, `chi2_scan_settings:
{n_points: 50}` — the number of points per scanned direction. Neither is a
fit: nothing is sampled or optimized, the chi2 is simply evaluated on a grid,
so no `prior` is consumed as a prior. **The scan range is read from the
coefficient's `prior:` block** — it must be `dist: uniform`, whose `low`/`high`
become the grid endpoints. Any other distribution (or no prior at all) falls
back to `[-1, 1]` with a warning, which is almost never what you want.

### 1D chi2 scan per coefficient

No template runcard ships for this one — take the structure from
`mass_scan.yaml` (or any fit template) and replace the coefficients and
`actions_:` with the block below.

```yaml
coefficients:
  OpWB: {free: True, prior: {dist: uniform, low: -0.5, high: 0.5}}
  OpD:  {free: True, prior: {dist: uniform, low: -0.5, high: 0.5}}
chi2_scan_settings:
  n_points: 10
actions_:
  - chi2_scan_table       # and/or plot_chi2_scan
```

Scans **each free coefficient in turn**, holding the other free ones at zero
(fixed-value coefficients keep their values) — the same one-at-a-time
`single_free` machinery as `run_individual_*_fits`, so the cost is
`n_free × n_points` chi2 evaluations. `chi2_scan_table` writes a table with
`(coefficient, {value, chi2})` columns; `plot_chi2_scan` writes one figure per
coefficient. Both work as `{@…@}` tags in a report `template_text`.

### Mass scan (`mass_scan.yaml`)

```yaml
rge:
  init_scale: 10000.0      # overridden per scan point
  obs_scale: 91.0
coefficients:
  OtG:  {free: False, vars: [m], expr: "1/m**2"}
  OpWB: {free: False, vars: [m], expr: "1/m**2"}
  m:    {free: True, prior: {dist: uniform, low: 1, high: 10}}   # TeV
chi2_scan_settings:
  n_points: 10
actions_:
  - mass_scan_table
```

Scans the chi2 as a function of a new-physics **mass scale**: exactly one free
coefficient (the mass), with every Wilson coefficient tied to it through an
`expr:` constraint. Requirements and costs, none of them obvious:

- **Exactly one free coefficient**, or the run fails at config time with
  `mass_scan requires exactly one free coefficient`.
- The scanned value is also used as `rge.init_scale` at each point, so the
  RGE matrix (and any external chi2's) is **recomputed per scan point** — the
  scan is genuinely `n_points` × the startup cost of one fit. Keep `n_points`
  small at first. A cached `rge.rg_matrix:` is *not* reused here, since the
  whole point is a different matching scale per point.
- The prior range is a mass range **in TeV**, not a Wilson-coefficient range.
  TeV is the convention because the `expr:` output *is* a Wilson coefficient,
  and those are in TeV^-2 — so `expr: "1/m**2"` works with no conversion
  factor. `rge.init_scale` is in GeV, but the scan point is converted for you
  (`gev_per_tev` in `produce_individual_mass_rge_matrix` /
  `produce_individual_mass_ext_chi2_func`), so `rge.init_scale` in the runcard
  stays in GeV like everywhere else.

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
- `plot_fisher_diagonals_heatmap` takes the same `cmap`, `value_fmt` and
  `colorbar` as the correlation heatmap — see "Configuring an action" below.
  `value_fmt: "{:.2f}"` is the one worth knowing: the default single decimal
  rounds a small but non-zero share to `0.0`.

## Reporting on fits that have already been run

`fits:` loads finished fit directories, so a runcard can report on them without
refitting anything — no `datasets`, `coefficients` or covariance flags needed.
Template: `posterior_correlations.yaml`.

```yaml
fits:
  - {name: analytical_fit, path: '.', label: '$\mathrm{Analytic}$'}
  - {name: ultranest_fit, path: '.', label: '$\mathrm{UltraNest}$'}

template_text: |
  # Posterior correlations
  {@with fits@}
  ## {@fit@}
  {@plot_posterior_correlations@}
  {@endwith@}
actions_:
  - report(main=True)
```

- `path` is the directory *containing* the fit; drop it to look the fit up in
  `smefit_results/fits`, downloading it from the server if it is not there.
- A `{@with fits@}`…`{@endwith@}` block runs everything inside it once per fit,
  and `{@fit@}` renders that fit's name — use it for a heading, or the report
  stacks every fit's output with nothing saying which is which. `{@fits
  <action>@}` is the one-line form, and gives up the heading.
- Outside a report, prefix the action with the namespace in `actions_`,
  space-separated: `- fits plot_posterior_correlations`. Either way outputs are
  named after the fit (`<fit_name>_plot_posterior_correlations.pdf`), and
  heatmaps are headed with the fit's `label` (its name when it has none).
- `plot_posterior_correlations` reads the posterior samples in each fit's
  `fit_results.json` and correlates the **free** coefficients. Every cell is
  annotated with its value, so there is no separate table. It rejects a fit run
  with `run_individual_*_fits`: its coefficients were never sampled together.

### Configuring an action

There is no settings block for how a figure looks, and none is needed: an
action's keyword parameters are resolved by reportengine like any other
resource, so a runcard sets them three ways.

As a **top-level key**, applying everywhere the parameter appears — the same
mechanism as `n_samples`, `seed` or `tol`:

```yaml
cmap: PuOr
colorbar: False
```

As an **action argument**, in a template or in `actions_`:

```yaml
template_text: |
  {@fits plot_posterior_correlations(cmap="PuOr", colorbar=False)@}
actions_:
  - fits plot_posterior_correlations(value_fmt="{:.3f}")
```

Argument values are parsed as YAML, so strings, numbers and booleans work.
**An action argument beats a top-level key** when both name the same
parameter — the specific wins.

Both heatmaps take `cmap`, `value_fmt` and `colorbar`, and they share those
names, so one top-level `cmap:` sets both; use an action argument where they
should differ. What is *not* settable is what would stop two plots being
comparable: the colour scale ([-1, 1] for correlations, 0–100% for Fisher),
square cells, and the correlation heatmap's heading, which is always the fit's
label so that every heatmap says which fit it is.

### Choosing which coefficients a report shows

`params_to_plot` restricts every routine that takes it to the coefficients
named, in the order named — a global fit's correlation heatmap is unreadable
with tens of operators on a side, and the interesting block is a handful of
them.

```yaml
params_to_plot: [OtG, OtW, OpQM, OpQ3]
```

It is the mechanism above, but the one key meant to be shared: sharing keeps a
report's figures and tables talking about the same operators in the same order.
Taken today by `plot_posterior_correlations` and `fisher_diagonals_normalised`
— so the Fisher table's CSV and the heatmap drawn from it are restricted
together. An action argument still overrides it for one figure.

- Write it as a YAML list of coefficient names — the raw names, not their LaTeX
  labels. Nothing validates the list, so a name is either matched or skipped.
- **Order is the runcard's**, not alphabetical: write operators in the order
  they should be read, grouped by sector if that helps.
- **Each fit keeps the largest subset it has.** A name a given fit never fitted
  is left out of that fit's heatmap and logged, so one list can head several
  fits with different coefficients. A list matching *nothing* raises instead —
  that is a misspelling, not a subset.

Three things to know about the mechanism:

- **An argument that is not a parameter of the action is silently ignored.**
  reportengine matches arguments against the action's signature
  (`_make_callspec`, `reportengine/resourcebuilder.py`) and drops the rest, so
  `cmpa="PuOr"` does nothing and says nothing. Check the spelling against the
  signature in `actions.md`.
- **The same action cannot appear twice with different arguments.** The node's
  namespace key is built from the action's *name* only
  (`_create_default_key`), so both calls collapse to one node and one output
  file: the first set of arguments wins, the second is dropped silently. Two
  variants of one figure need two providers.
- **A parameter has to be a parameter.** Only what a provider declares can be
  set this way, which is the deliberate limit on how much of a plot a runcard
  controls — see `smefit-dev` before adding one.

## Precision and performance

- `smefit <runcard> -f32` switches JAX to float32 — faster, but check
  convergence.
- `chi2_timing` as an action benchmarks the likelihood before committing to an
  expensive sampler; `time_likelihood.yaml` is the template.
- For samplers, start from the template settings; increase
  `min_num_live_points` / `n_live` only when posteriors look under-resolved.
