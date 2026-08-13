---
name: smefit-runcard
description: Use this skill when creating, editing, or validating a smefit runcard — the YAML configuration for a SMEFT fit. Covers choosing the fit action (analytic, ultranest, blackjax, hessian, individual fits, chi2/mass scans, projections, reports), defining Wilson coefficients and priors, RGE running, covariance flags (use_t0, use_theory_covmat, use_quad), external chi2, whitening, and every settings block (ultranest_settings, blackjax_settings, optimizer_settings, gradient_descent_settings, hessian_settings, pseudodata_settings, chi2_scan_settings).
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
   `blackjax_individual_fit.yaml`, `hessian_fit.yaml`), `mass_scan.yaml` for a
   scan evaluating the chi2 on a grid instead, plus the runcards that fit
   nothing: `projections.yaml`, `time_likelihood.yaml`, `report.yaml`, and
   `posterior_correlations.yaml` (reports on fits already on disk — it needs no
   `datasets` or `coefficients` at all, so steps 2–5 do not apply to it). Do
   not write a runcard from scratch. The 1D chi2 scan has no template of its
   own: start from `mass_scan.yaml` and follow the recipe in `recipes.md`.
   Take the **structure** from them — settings blocks, key names, which blocks
   pair with which action — but treat their `datasets`, `coefficients` and
   `external_chi2` entries as placeholders: they are the repo's smoke-test
   values (e.g. `expr: "y**2 + 0.5*OpWB**2"`), not a physics starting point.
   Replace them using steps 3–4.

2. **Resolve `data_path` / `theory_path`**. Prefer the shareable prefix form —
   `data_path: smefit_database/commondata`, `theory_path: smefit_database/theory` —
   which smefit resolves through the machine-specific `.config/paths.yaml`
   (created once by the interactive `smefit_setup_local` command; any key in
   that file works as a prefix, and absolute paths still work too). Check the
   setup by running the smefit-datasets skill's locate script — it lives in the
   sibling skill directory, `../smefit-datasets/scripts/smefit_db.py` relative
   to this file, so build an absolute path from this skill's directory rather
   than running it relative to the working directory:
   ```bash
   python /abs/path/to/skills/smefit-datasets/scripts/smefit_db.py locate
   ```
   It prints the recommended runcard values. If it exits with "no database
   found", ask the user before running `smefit_setup_local` (it also offers to
   clone the database).

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

6. **Validate** before handing the runcard to the user, using an absolute path
   to this skill's own `scripts/` directory:
   ```bash
   python /abs/path/to/skills/smefit-runcard/scripts/validate_runcard.py <runcard.yaml>
   ```
   Fix every error it reports. Then the authoritative check is simply running
   `smefit <runcard.yaml>` (see the smefit-analysis skill).

## Minimal skeleton

```yaml
data_path: smefit_database/commondata    # prefix resolved via .config/paths.yaml
theory_path: smefit_database/theory
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

## Confirm with the user before finalizing

These change the physics content of the fit, not just its robustness —
always surface them as an explicit question rather than defaulting silently:
- `use_quad` (linear vs. linear+quadratic — no universally correct choice)
- `rge` presence and `init_scale` (observable-scale vs. high-scale matching)
- prior distributions/ranges when not supplied by the user (placeholders bias
  interpretation even if the fit "runs fine")
- any dataset chosen to constrain a coefficient not requested by the user
  (e.g. adding a Higgs dataset so a requested coefficient is constrained at all)

`use_t0`/`use_theory_covmat` remain safe to default to `True` when
systematics/theory errors are present — just say so, don't ask.

## Rules of thumb

- Prefer prefix paths (`smefit_database/...`, `new_smefit/...`,
  `smefit_results/...`) over absolute ones — the runcard then works unchanged
  on any machine that ran `smefit_setup_local`. See the "Path resolution"
  section of `references/runcard-keys.md`.
- `run_analytic_fit` requires a purely linear model: `use_quad: False` and no
  expression-constrained coefficients (the posterior must be Gaussian).
- A free coefficient needs a `prior` only for sampler actions (`run_ultranest_fit`,
  `run_blackjax_fit`, or their `individual_*` variants) — `run_analytic_fit`,
  `run_hessian_fit`, and gradient-descent fits never consume it. `whitening`/
  `bayesian_update_path` synthesize a prior automatically either way. Only
  `uniform` and `gaussian`/`normal` exist (`references/priors.md`).
- `use_t0: True` is the statistically sound choice when multiplicative
  systematics are present; pair it with `use_theory_covmat: True` when theory
  errors matter.
- The `rge` block is present in all templates; drop it only if coefficients are
  defined at the observable scale.
- Output goes to a directory named after the runcard stem unless `-o` is given.
