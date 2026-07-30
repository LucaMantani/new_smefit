---
name: smefit-datasets
description: Use this skill when discovering or looking up SMEFiT datasets, Wilson coefficients/operators, or external likelihoods — e.g. "what datasets are available for top/Higgs/diboson?", "which operators does dataset X constrain?", "is OtG implemented?" — and when locating, cloning, or setting up the smefit_database repository (commondata, theory tables, projections, external chi2).
---

# SMEFiT dataset discovery

All experimental data and theory predictions live in a separate repository:
[LHCfitNikhef/smefit_database](https://github.com/LHCfitNikhef/smefit_database).
Runcards reference a local clone of it through the shareable prefix form
(`data_path: smefit_database/commondata`, `theory_path: smefit_database/theory`),
resolved via the machine-specific `.config/paths.yaml` that the interactive
`smefit_setup_local` command creates (it also offers to clone the database).
Layout and file schemas: `references/database-layout.md`.

**Never answer catalog questions from memory.** Always run the bundled script —
it reads the database's machine-readable catalogs (`data_summary.yaml`,
`operators_implemented.yaml`, `ext_likelihood_summary.yaml`), preferring a
local clone and transparently falling back to GitHub:

```bash
# absolute path to this skill's own scripts/ directory — the script is not on PATH
python /abs/path/to/skills/smefit-datasets/scripts/smefit_db.py <subcommand> [--json] [--offline]
```

| Subcommand | Purpose |
|---|---|
| `locate` | Find the local clone (via `.config/paths.yaml` first); prints the `data_path`/`theory_path` values to use in runcards |
| `search KEYWORD` | Datasets matching a keyword (name or experiment group), with allowed orders |
| `info DATASET` | Metadata (description, arxiv, num_data) + operators entering the predictions |
| `operators [PATTERN]` | Implemented Wilson coefficients with their WCxf Warsaw-basis definitions |
| `ext [PATTERN]` | External likelihoods, printed as ready-to-paste `external_chi2:` runcard blocks |
| `clone [DEST]` | Print the setup/clone commands (never executes them) |

Exit codes: `0` ok, `1` nothing matched, `3` no database reachable (no clone,
no network).

## Workflow

1. Run `smefit_db.py locate`. If it finds a clone, use the printed runcard
   values (prefix form when `.config/paths.yaml` is configured, absolute
   otherwise).
2. If it exits with code 3 (or suggests it), **ask the user** whether to run
   `smefit_setup_local` — smefit's interactive setup that records the paths in
   `.config/paths.yaml` and offers to clone the database (~large repo). Do not
   run it or clone without confirmation.
3. Use `search`/`info` to pick datasets. A dataset's `order` in a runcard must
   be one of its `allowed_orders`; `info` also shows which Wilson coefficients
   actually enter its predictions — useful to pick coefficients that the chosen
   data can constrain (a coefficient no dataset is sensitive to gives an
   unconstrained, prior-shaped posterior).
4. Hand the chosen names to the runcard (see the **smefit-runcard** skill,
   which references this script for paths and dataset validation).

## Notes

- The keyword search covers dataset names and experiment groups (ATLAS, CMS,
  LEP, FCCee, CEPC, HLLHC, …). Physics-process vocabulary maps onto name
  fragments: try `tt` (top pairs), `AC` (asymmetries), `STXS`/`ggF`/`WH`/`ZH`
  (Higgs), `WW`/`WZ` (diboson), `EWPO` (electroweak precision), `_proj`
  (HL-LHC projections).
- Future-collider projections live in `commondata_projections_L0/` of the
  database; `info` checks there too.
- GitHub fallback fetches only the small catalog files (cached ~1 day in the
  system temp dir); theory JSONs are large, so per-dataset operator listings
  need a local clone.
