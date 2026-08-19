# Troubleshooting smefit runs

Common failures, their real causes, and fixes. Error texts below are excerpts
of the actual exceptions raised by the code.

## Configuration / startup errors

| Error | Cause → fix |
|---|---|
| `Path '<p>' starts with '<prefix>' but '<prefix>' is not set in .../.config/paths.yaml` | The runcard uses a prefix path (`smefit_database/...`, `smefit_results/...`) but the machine's local setup is missing/incomplete → run `smefit_setup_local` (ask the user first). |
| `smefit_results is not configured. Run 'smefit_setup_local' to set it up.` | Server-related scripts need the `smefit_results` key in `.config/paths.yaml` → run `smefit_setup_local`. |
| `'<path>' does not exist locally and could not be downloaded from the server: ...` | An `rg_matrix` (or fit) under `smefit_results/fits/<name>/` is neither local nor on the server — check the fit name (`smefit_ls`), network, and server credentials. |
| `data_path ... does not exist` / `theory_path ... does not exist` | The (resolved) paths don't point into a smefit_database clone. Run the smefit-datasets skill's `smefit_db.py locate`; check `.config/paths.yaml`. |
| `Dataset <name> not found in <data_path>` | Typo in `datasets[].name`, or the dataset lives in `commondata_projections_L0/` while `data_path` points at `commondata/`. Check with `smefit_db.py search`. |
| `Theory predictions for dataset <name> not found` | Missing `<name>.json` under `theory_path` — same causes as above. |
| `KeyError: '<order>'` while loading theory | The requested `order` is not a key of that dataset's theory JSON. `smefit_db.py info <name>` lists the allowed orders. |
| `Theory covariance type <type> not found for dataset <name>` | `theory_cov:` value not available for that dataset (types are per-dataset). |
| `rge block requires 'init_scale'` | Add `init_scale:` (GeV) to the `rge` block. |
| `Free coefficient '<name>' has no prior defined` | Every `free: True` coefficient needs a `prior` (unless whitening / bayesian_update is active). |
| `<name>: free=True forbids 'value' and 'expr'` (and similar) | Coefficient kind invariants violated — see the smefit-runcard skill's `coefficients.md`. |
| `Module <stem> not found in <path>` | `external_chi2` `path:` doesn't point at the module file (usually a stale `/path/to/smefit_database` placeholder). |
| `No data provided and no external_chi2 configured` | Runcard has neither `datasets` nor `external_chi2`. |
| `Could not find previous ultranest fit at <log_dir>` | `ReactiveNS_settings.resume: True` but no previous run in the output dir — drop `resume` or point `-o` at the old output. |
| `fit_results.json not found at <path>` / `input/runcard.yaml not found` | The `bayesian_update` fit must be a *completed* smefit output directory. |
| reportengine `ConfigError` about an unknown action | Action name not in `references/actions.md` (typo, or the runcard predates a rename). |
| `whitening is not compatible with bayesian_update` | Exactly that — remove one of the two. |
| `mass_scan requires exactly one free coefficient, got N: [...]` | `mass_scan_table` scans a single mass parameter; every Wilson coefficient must be tied to it with `expr:`/`vars:`, i.e. `free: False`. |

## Silent misbehavior (no crash)

- **Misspelled keys are only warnings.** smefit logs
  `Unknown key '<k>' in <block> settings` and continues with defaults. Grep the
  log output for `Unknown key` after every run; validate runcards beforehand
  with the smefit-runcard skill's `validate_runcard.py`.
- **Posterior equals prior** for a coefficient: the selected datasets are not
  sensitive to it (`smefit_db.py info <dataset>` shows which operators enter).
- **chi2/dof ≫ 1**: dataset tension or covariance misconfiguration — typical
  culprits are `use_t0: False` with multiplicative systematics, or missing
  `use_theory_covmat: True`.
- **A chi2/mass scan covers the wrong range**: the grid endpoints come from the
  scanned coefficient's `uniform` prior. Anything else logs
  `does not have a uniform prior with 'low'/'high'` / `lacks a uniform prior`
  and silently falls back to `[-1, 1]` — grep for it, and give the coefficient
  a `prior: {dist: uniform, low: …, high: …}` spanning the range you want.
- **A run produced no `fit_results.json`**: expected when `actions_:` contains
  only table/figure actions (Fisher, `chi2_scan_table`, `mass_scan_table`) —
  results are under `tables/` and `figures/`, not in a fit result file.

## Performance / resources

- **Long startup before sampling**: the RGE matrix is being computed
  (per-datapoint scales). Reuse `<output>/rge_matrix.pkl` via `rge.rg_matrix:`
  in subsequent runs — e.g. `rg_matrix: smefit_results/fits/<fit>/rge_matrix.pkl`
  (auto-downloaded from the server if the fit is not local).
- **Mass scan is `n_points` times slower than expected**: by design — the
  scanned scale *is* `rge.init_scale`, so the RGE matrix (and any external
  chi2) is rebuilt at every point and no cached `rg_matrix` can be reused.
  Lower `chi2_scan_settings.n_points` for exploratory runs.
- **Out of memory with `use_quad: True`**: quadratic predictions are
  `[ndata, n_ops, n_ops]` arrays — reduce coefficients/datasets or run `-f32`.
- **Sampler runs forever**: lower `min_num_live_points`/`min_ess` (UltraNest)
  or `n_live` (BlackJAX) for exploratory fits; benchmark with `chi2_timing`
  first; consider `run_analytic_fit` if the model is linear.
- **float64 vs float32**: results differing between `-f32` and default runs
  indicate conditioning problems — trust float64, or use `whitening`.
