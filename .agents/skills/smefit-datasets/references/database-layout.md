# smefit_database layout and file schemas

Repository: https://github.com/LHCfitNikhef/smefit_database

```
smefit_database/
  data_summary.yaml            # CATALOG: all datasets, grouped by experiment
  operators_implemented.yaml   # CATALOG: Wilson coefficients + WCxf definitions
  ext_likelihood_summary.yaml  # CATALOG: external likelihoods (runcard-ready blocks)
  commondata/                  # <dataset>.yaml — measurements (one file per dataset)
  commondata_projections_L0/   # projected datasets for future colliders
  theory/                      # <dataset>.json — SM + EFT predictions (mirrors commondata names)
  external_chi2/               # Python modules for external likelihoods (+ their data)
  runcards/                    # example runcards maintained with the database
  Kappa_framework/, scripts/, tests/
```

A valid local clone is recognized by `smefit_db.py` when it contains
`commondata/`, `theory/`, and `data_summary.yaml`. Its location is recorded in
smefit's machine-specific `.config/paths.yaml` (key `smefit_database`, created
by `smefit_setup_local`), which lets runcards use the shareable prefix form
`data_path: smefit_database/commondata` instead of absolute paths.

## data_summary.yaml

Top-level keys are experiment groups (`ATLAS`, `CMS`, `ATLAS_CMS`, `LEP`,
`HLLHC`, `FCCee_*`, `CEPC_*`, …) plus an `Info` string. Each group maps to a
list of entries:

```yaml
ATLAS:
  - {name: ATLAS_STXS_runII_13TeV, order: NLO_QCD, allowed_orders: [LO, NLO_QCD]}
```

- `name` — matches `commondata/<name>.yaml` and `theory/<name>.json`.
- `order` — recommended default order for runcards.
- `allowed_orders` — every order key present in the theory file (orders are
  dataset-specific; some exotic ones exist, e.g. `NLO_EW_only_for_ZH`).

## commondata/<name>.yaml

| Key | Content |
|---|---|
| `dataset_name` | must equal the filename stem |
| `doi`, `arxiv`, `hepdata`, `location`, `description` | provenance metadata |
| `units`, `luminosity` | units string; luminosity in fb⁻¹ (scalar or per-point list) |
| `num_data`, `num_sys` | number of data points / systematic sources |
| `data_central` | central values, length `num_data` |
| `statistical_error` | statistical errors, length `num_data` |
| `systematics` | matrix, shape `(num_sys, num_data)` |
| `sys_names` | per-systematic correlation labels (`CORR`, `UNCORR`, or shared names) |
| `sys_type` | per-systematic type (`ADD` or `MULT` — MULT entries drive the t0 prescription) |

## theory/<name>.json

| Key | Content |
|---|---|
| `best_sm` | SM predictions, length `num_data` |
| `scales` | per-data-point scale in GeV (used by `rge.obs_scale: dynamic`) |
| `theory_cov_current` / `_conservative` / `_aggressive` | theory covariance variants (`theory_cov` runcard key) |
| `LO`, `NLO_QCD`, … | one dict per available order |

Each order dict maps prediction keys to per-data-point lists:

- `"SM"` — SM prediction at that order,
- `"<Op>"` — linear EFT correction for operator `<Op>`,
- `"<OpA>*<OpB>"` — quadratic (dim-6)² cross terms (upper triangular; only
  present for datasets with quadratic corrections; needed for `use_quad: True`).

## external_chi2/

Self-contained Python likelihood modules (e.g. `drell_yan/`,
`optimal_observables/` with `invcov_*.dat` inverse covariances for future
colliders). Consume via `ext_likelihood_summary.yaml`, whose entries are
complete `external_chi2:` runcard blocks with a `/path/to/smefit_database`
placeholder (`smefit_db.py ext` rewrites it to the located clone). The module
class contract is documented in the smefit-runcard skill's `recipes.md`.
