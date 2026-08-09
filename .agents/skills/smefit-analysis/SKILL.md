---
name: smefit-analysis
description: Use this skill when running smefit fits or analyses from the command line, interpreting fit output (fit_results.json, posterior samples, chi2, log-evidence, best-fit tables), producing reports with Fisher information tables and heatmaps, comparing or post-processing fits, debugging failed or slow smefit runs, or timing the likelihood.
---

# Running and interpreting smefit analyses

Assumes a runcard exists (to create or fix one, use the **smefit-runcard**
skill). References here:

- `references/actions.md` — auto-generated list of every action and the
  fit-type → action → settings-block table. Trust it over memory.
- `references/output-layout.md` — what each run writes where, and the
  `fit_results.json` schema.
- `references/troubleshooting.md` — common failures and their causes.

## Running

```bash
smefit <runcard.yaml>            # output dir = runcard filename stem
smefit <runcard.yaml> -o my_fit  # explicit output dir
smefit <runcard.yaml> -f32       # float32 precision (faster, less accurate)
```

The `smefit` CLI must run in an environment with the package installed (in the
development repo: `conda activate new_smefit`). The run executes every entry in
`actions_:`; reportengine resolves the dependency graph automatically.

First-time machines need the local path setup once (`smefit_setup_local`,
interactive — ask the user before running it): it writes `.config/paths.yaml`,
which resolves the prefix paths (`smefit_database/...`, `smefit_results/...`)
used in shareable runcards. Fits/reports shared via the server live under the
configured `smefit_results/` directory (`smefit_ls` to list, `smefit_get` to
download); a runcard referencing `smefit_results/fits/<name>/...` triggers an
automatic download when the fit is missing locally.

## Reading results

Every fit action prints a summary table (best fit, std, prior per coefficient;
chi2, chi2/dof, log Z when sampling) and writes `fit_results.json` in the
output directory. Key fields: `best_fit_point`, `std`, `samples` (full
posterior, per coefficient — null for pure best-fit runs), `chi2`,
`chi2_ndof`, `logz`, `bic`, `aic`. Details and the individual-fits layout:
`references/output-layout.md`.

- To quote a constraint on a coefficient: prefer quantiles of `samples`
  (e.g. 68%/95% central intervals via numpy) over `best_fit_point ± std` when
  samples exist — posteriors can be non-Gaussian, especially with
  `use_quad: True`.
- `chi2_ndof` far above 1 signals tension or a mis-configured covariance
  (check `use_t0` / `use_theory_covmat`).
- Compare models with `logz` (nested sampling only — it is `null` for
  `blackjax_settings.algorithm: nuts` and for the analytic/Hessian fits) or
  `bic`/`aic`, which every fit provides.
- After a `nuts` fit, read `blackjax_logs/nuts_diagnostics.json` before trusting
  the posterior: `max_rhat` should be < 1.01 and `divergences` should be 0.
  After a `nested_sampling` fit, read `blackjax_logs/nested_diagnostics.json`:
  `converged` should be true, and `d_G` (constrained directions) close to
  `n_free` — well below it means the data leaves directions unconstrained.
- A posterior that looks identical to the prior means the chosen data does not
  constrain that coefficient (check with the smefit-datasets skill:
  `smefit_db.py info <dataset>` shows the operators a dataset is sensitive to).

## Reports and Fisher information

Report runcards render `template_text` with `{@action@}` tags via the `report`
action into `<output>/index.html` (tables and figures alongside). The Fisher
chain (`fisher_diagonals_normalised`, `plot_fisher_diagonals_heatmap`)
evaluates at the gradient-descent best fit; group datasets with `group:` labels
to aggregate rows. See the smefit-runcard skill's `recipes.md` for the runcard
side.

## Performance checklist

1. Benchmark first: `chi2_timing` action (template `time_likelihood.yaml`).
2. RGE matrix computation can dominate startup — it is cached per run at
   `<output>/rge_matrix.pkl`; reuse it in later runs via `rge.rg_matrix:`.
3. `-f32` roughly halves memory and speeds up sampling; verify results against
   a float64 run before trusting it.
4. Prefer `run_analytic_fit` (exact, seconds) whenever the model is linear
   (`use_quad: False`, no nonlinear constraints) — samplers are overkill there.
5. UltraNest cost scales with `min_num_live_points` and the number of free
   coefficients; for quick sanity checks lower it in `ultranest_settings`.
