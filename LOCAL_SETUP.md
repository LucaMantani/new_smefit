# Local Setup

After installing the package, run the interactive setup command once to configure your local paths:

```bash
smefit_setup_local
```

This creates `.config/paths.yaml` in the repository root with the three standard directories. The file is machine-specific and listed in `.gitignore` — it is never committed.

## What the setup does

1. **Infers defaults** from the package location, assuming the three directories are siblings:

   ```
   <parent>/
   ├── new_smefit/       ← this repository
   ├── smefit_database/  ← commondata and theory files
   └── smefit_results/   ← fit and report outputs
   ```

2. **Prompts** for each path, showing the inferred default. Press Enter to accept.
3. **Clones `smefit_database`** from GitHub if the directory does not exist (optional, requires `git`).
4. **Creates `smefit_results`** if it does not exist.

## The config file

`.config/paths.yaml` is plain YAML and can be edited directly in VSCode at any time:

```yaml
new_smefit: /path/to/new_smefit
smefit_database: /path/to/smefit_database
smefit_results: /path/to/smefit_results

# You can add any number of custom path aliases below.
# Use the key name as a prefix in runcards, just like the standard paths.
# Example:
#   my_extra_database: /path/to/my_extra_database
# Then in a runcard:
#   data_path: my_extra_database/commondata
```

Re-run `smefit_setup_local` at any time to update the standard paths interactively.

## Using paths in runcards

Any key in `.config/paths.yaml` can be used as a path prefix in runcard YAML files:

```yaml
data_path: smefit_database/commondata
theory_path: smefit_database/theory
bayesian_update:
  name: my_previous_fit
  path: smefit_results/fits   # optional, this is the default

external_chi2:
  CMS_DYMee_13TeV:
    path: new_smefit/external_chi2/drell_yan/CMS_DYMee_13TeV.py
```

Absolute paths still work unchanged, so existing runcards do not need to be modified.

## `smefit_results` directory structure

Fits and reports downloaded from the server (via `smefit_get`) or produced locally are stored under `smefit_results/`:

```
smefit_results/
├── registry.json          ← local index of all tracked resources
├── fits/
│   └── <fit_name>/
│       ├── <fit_name>.yaml      ← runcard used to produce the fit
│       ├── fit_results.json     ← posterior samples and summary statistics
│       ├── rge_matrix.pkl       ← RGE matrix (optional)
│       ├── blackjax_logs/
│       ├── figures/
│       ├── individual_fits/
│       ├── input/
│       └── tables/
└── reports/
    └── <report_name>/
```

Use `smefit_ls` to list the contents of the local registry, and `smefit_get` to download resources from the server into this directory.

### Loading an RGE matrix from a stored fit

Once a fit containing an `rge_matrix.pkl` has been downloaded or produced locally, you can point a new runcard at it using the `smefit_results` prefix.

**Vanilla case** — `rg_matrix` lives inside the `rge:` block:

```yaml
rge:
  init_scale: 1000
  obs_scale: 91.2
  yukawa: top
  rg_matrix: smefit_results/fits/my_fit/rge_matrix.pkl
```

**External likelihood case** — `rg_matrix` lives inside the relevant `external_chi2` entry:

```yaml
external_chi2:
  CMS_DYMee_13TeV:
    path: new_smefit/external_chi2/drell_yan/CMS_DYMee_13TeV.py
    rg_matrix: smefit_results/fits/my_fit/CMS_DYMee_13TeV/rge_matrix.pkl #(needs to be saved there manually)
```

In both cases the `smefit_results/` prefix is resolved automatically against the path configured in `.config/paths.yaml`.

### Automatic fit download

If the fit is not found locally, smefit will attempt to download the full fit archive from the server automatically before failing.

The local `registry.json` is updated after a successful download, exactly as if you had run `smefit_get` manually.

If the download also fails (fit not on server, no network, wrong name), a clear error is raised:

```
FileNotFoundError: '<path>' does not exist locally and could not be downloaded from the server: Fit '<name>' not found on server.
```
