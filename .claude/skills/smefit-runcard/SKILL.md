---
name: smefit-runcard
description: Use this skill when creating, editing, or validating a smefit runcard — the YAML configuration for a SMEFT fit. Covers choosing the fit action (analytic, ultranest, blackjax, hessian, individual fits, projections, reports), defining Wilson coefficients and priors, RGE running, covariance flags (use_t0, use_theory_covmat, use_quad), external chi2, whitening, and every settings block (ultranest_settings, blackjax_settings, optimizer_settings, gradient_descent_settings, hessian_settings, pseudodata_settings).
version: 0.1.0
---

# smefit runcard authoring

A runcard is a YAML file executed with `smefit <runcard.yaml> [-o output_dir]`.
It declares data/theory paths, datasets, Wilson coefficients with priors, and
the actions to run. reportengine resolves everything else automatically.

**Trust the bundled references over memory** — they are auto-generated from the
code and kept in sync by CI:

- `references/runcard-keys.md` — every recognized key, sub-key, and default.
  Only keys listed there exist; unknown keys are silently warned, not rejected.
- `references/actions.md` — every action usable in `actions_:`, plus a table
  mapping fit type → action → settings blocks.
- `references/priors.md` — the allowed prior distributions (only these).
- `references/coefficients.md` — free/fixed/constrained coefficient rules (hand-written).
- `references/recipes.md` — RGE, whitening, Bayesian updating, external chi2, projections.

## Workflow

1. **Start from a template** in `templates/` — one per fit type
   (`analytical_fit.yaml`, `ultranest_fit.yaml`, `blackjax_fit.yaml`,
   `blackjax_individual_fit.yaml`, `hessian_fit.yaml`, `projections.yaml`,
   `time_likelihood.yaml`, `report.yaml`). Do not write a runcard from scratch.

2. **Resolve `data_path` / `theory_path`**. They must point into a local clone of
   the [smefit_database](https://github.com/LHCfitNikhef/smefit_database) repo
   (`<clone>/commondata` and `<clone>/theory`). Find the clone by running the
   smefit-datasets skill's locate script:
   ```bash
   python "$(dirname <this skill dir>)/smefit-datasets/scripts/smefit_db.py" locate
   ```
   (i.e. `../smefit-datasets/scripts/smefit_db.py` relative to this SKILL.md).
   If it exits with "no database found", follow its instructions to clone.

3. **Choose datasets** with the same script (`smefit_db.py search <keyword>`,
   `smefit_db.py info <dataset>`). Never invent dataset names: every entry in
   `datasets:` must have a matching `<name>.yaml` in commondata and
   `<name>.json` in theory, and its `order` must be allowed for that dataset.

4. **Define coefficients** following `references/coefficients.md`. Check
   operator names with `smefit_db.py operators <pattern>` — never guess them.

5. **Pick the action + settings block** from the decision table in
   `references/actions.md`. The single most important choice: `actions_:` and
   the matching settings block must agree (e.g. `run_ultranest_fit` ↔
   `ultranest_settings`).

6. **Validate** before handing the runcard to the user:
   ```bash
   python <this skill dir>/scripts/validate_runcard.py <runcard.yaml>
   ```
   Fix every error it reports. Then the authoritative check is simply running
   `smefit <runcard.yaml>` (see the smefit-analysis skill).

## Minimal skeleton

```yaml
data_path: /path/to/smefit_database/commondata
theory_path: /path/to/smefit_database/theory
use_theory_covmat: True
use_t0: True
use_quad: False        # include (dim-6)^2 terms?

rge:                   # needed when coefficients are defined at a scale != observables
  init_scale: 10000.0  # GeV, required
  obs_scale: dynamic

datasets:
  - {name: LEP1_EWPOs_2006, order: LO}

coefficients:
  OpWB: {free: True, prior: {dist: uniform, low: -0.5, high: 0.5}}

actions_:
  - run_analytic_fit
```

## Rules of thumb

- `run_analytic_fit` requires a purely linear model: `use_quad: False` and no
  expression-constrained coefficients (the posterior must be Gaussian).
- Every free coefficient needs a `prior`; only `uniform` and `gaussian`/`normal`
  exist (`references/priors.md`).
- `use_t0: True` is the statistically sound choice when multiplicative
  systematics are present; pair it with `use_theory_covmat: True` when theory
  errors matter.
- The `rge` block is present in all templates; drop it only if coefficients are
  defined at the observable scale.
- Output goes to a directory named after the runcard stem unless `-o` is given.
