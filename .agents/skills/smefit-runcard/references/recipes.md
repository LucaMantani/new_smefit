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

### What can be drawn from a finished fit

Everything below reads the posterior samples in `fit_results.json`; none of it
refits. `actions.md` has the full signatures.

| What | Overlaid, called bare | One per fit, under `{@with fits@}` | Template |
|---|---|---|---|
| 1D posteriors | `plot_fits_posterior_histograms` | `plot_posterior_histograms` | `posterior_histograms.yaml` |
| Central value + C.I. | `plot_fits_coefficient_bounds` | `plot_coefficient_bounds` | `coefficient_bounds.yaml` |
| Mass reach `Λ/√cᵢ` | `plot_fits_mass_reach` | `plot_mass_reach` | `mass_reach.yaml` |
| 2D contours | `plot_fits_posterior_contours` | `plot_posterior_contours` | `posterior_contours.yaml` |
| Correlations | — | `plot_posterior_correlations` | `posterior_correlations.yaml` |
| Bounds table | `coefficient_bounds_table` | — | `coefficient_bounds_table.yaml` |

Two things decide which member of a pair to call, and neither is an option:

- **How many fits reportengine hands the action**, which its first parameter
  name fixes. `fits` (plural) takes the whole list and draws one figure with
  every fit in it, so it is called bare; `fit` takes one and is called under
  `{@with fits@}`, giving one figure per fit.
- **Whether the action can read a one-at-a-time fit.** The 1D routines —
  histograms, bounds, reach, table — accept a `run_individual_*_fits` output,
  because a 1D posterior per coefficient is all they need. So the
  "marginalised vs individual" pair of figures is two `fits:` entries pointing
  at two fit directories, never a flag. The contour and correlation actions
  reject one: those coefficients were never sampled together, and pairing
  independent posteriors would draw a correlation set by the seed.

Options worth knowing before reaching for the key tables:

- `confidence_level` is a percentage — 95, not 0.95. The routines reading 1D
  bounds (histograms aside: the bounds plot, the reach plot, the table) raise
  on a value below 1 rather than quietly drawing a 0.95% interval; **the
  contour actions do not check it**, and 0.95 there silently draws a contour
  nobody wants. The contour and bounds plots take a two-element list to draw
  two levels at once; the reach plot takes one, since a bar has one height.
- `double_solution: [OtG]` declares a coefficient whose posterior has two
  disjoint solutions, as quadratic corrections produce. It is never detected:
  a posterior is bimodal because of the physics. Without it, equal-tailed
  percentiles span the empty gap between the modes and put the central value
  where there is no posterior mass — so the bounds plot and the table need it,
  while the contours do not (their KDE level is calibrated on the samples).
  A dict keyed by fit name sets it per fit.
- `coefficient_bounds_table` chooses its columns: `show_bounds` (default true)
  gives `best` plus one column per entry of `bounds_levels` (default `95`
  alone; `[68, 95]` for the pair the old report quoted, any level for
  anything else), and `show_reach` adds `Λ/√cᵢ` at `confidence_level`. Both
  off is an error.
- Lists and dicts have to be **top-level keys**: the template argument parser
  splits on commas, so `{@action(confidence_level=[68, 95])@}` does not parse.
  Scalars and booleans go either way.

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
Taken by every report routine that is about coefficients — the posterior
figures and table listed above, `plot_posterior_correlations` and
`fisher_diagonals_normalised` — so the Fisher table's CSV and the heatmap drawn
from it are restricted together, and a bounds figure and the table beside it
cannot end up about different operators. An action argument still overrides it
for one figure.

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
