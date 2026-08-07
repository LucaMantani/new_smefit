---
name: smefit-analysis
description: Use this skill when running smefit fits or analyses from the command line, interpreting fit output (fit_results.json, posterior samples, chi2, log-evidence, best-fit tables), producing reports with Fisher information tables and heatmaps, scanning the chi2 over a coefficient or a new-physics mass scale, comparing or post-processing fits, debugging failed or slow smefit runs, or timing the likelihood.
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
- Compare models with `logz` (nested sampling) or `bic`/`aic`.
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

## chi2 scans

`chi2_scan_table` / `plot_chi2_scan` (per free coefficient, others held at
zero) and `mass_scan_table` (chi2 vs. a new-physics mass scale) evaluate the
chi2 on a grid rather than fitting. They write into `tables/` and `figures/`
and produce **no `fit_results.json`** — do not look for one. Grid size is
`chi2_scan_settings.n_points`, and the range comes from each scanned
coefficient's `uniform` prior; see the smefit-runcard skill's `recipes.md`.

Reading them:
- The scan minimum is a best-fit estimate only along that one direction; it
  does not equal a joint best fit unless the coefficients are uncorrelated.
- `Δchi2 = 1` / `3.84` around the minimum give the 68% / 95% 1D intervals for
  a Gaussian likelihood — a scan whose curve is visibly non-parabolic
  (common with `use_quad: True`) invalidates that reading, and a real fit is
  needed.
- A flat scan means the data does not constrain that direction.
- A mass scan is the expensive one: the scanned scale is used as
  `rge.init_scale`, so the RGE matrix is recomputed at every point.

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
