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

- Computing the RGE matrix can be slow. A fit action saves it under the run's
  output directory automatically, next to `fit_results.json`. To reuse one
  across runs, point `rg_matrix:` at the saved pickle — preferably via the
  `smefit_results` prefix:
  `rg_matrix: smefit_results/fits/<fit_name>/rge_matrix.pkl`. If the fit is not
  present locally, smefit automatically tries to download it from the server
  before failing.
- To precompute a matrix without running a fit — or to persist it up front,
  before a long sampling run rather than after it — use the `write_rge_matrix`
  action. It needs only the `rge:` block, `datasets:` and `coefficients:`:

  ```yaml
  actions_:
    - write_rge_matrix   # first, so a killed run still leaves the matrix behind
    - run_ultranest_fit
  ```
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

## Precision and performance

- `smefit <runcard> -f32` switches JAX to float32 — faster, but check
  convergence.
- `chi2_timing` as an action benchmarks the likelihood before committing to an
  expensive sampler; `time_likelihood.yaml` is the template.
- For samplers, start from the template settings; increase
  `min_num_live_points` / `n_live` only when posteriors look under-resolved.
