# Claude Code skills for smefit

Four skills assist users and developers working with smefit:

| Skill | Purpose |
|---|---|
| `smefit-runcard` | Author and validate runcards (keys, coefficients, priors, settings) |
| `smefit-datasets` | Discover datasets/operators/external likelihoods in smefit_database |
| `smefit-analysis` | Run fits, interpret output, reports/Fisher, troubleshooting |
| `smefit-server` | Develop the server/registry infrastructure (maintainers) |

## Generated vs hand-written files

Files carrying an `AUTO-GENERATED` / `AUTO-COPIED` banner are produced by

```bash
python scripts/generate_skill_reference.py          # regenerate
python scripts/generate_skill_reference.py --check  # CI freshness check
```

which introspects the package (runcard keys from `smefit/config.py`, actions
from the provider modules in `smefit/app.py`, priors from `smefit/priors.py`,
path prefixes from `smefit/paths.py`) and copies `template_runcards/*.yaml`
verbatim (they already use shareable prefix paths). Generated files:

- `smefit-runcard/references/runcard-keys.{md,json}`, `actions.md`, `priors.md`
- `smefit-runcard/templates/*.yaml`
- `smefit-analysis/references/actions.md`

Never edit these by hand — change the code/templates and regenerate. The
`.github/workflows/skills.yml` CI job fails PRs whose generated files are
stale, so user-facing changes (new runcard keys, actions, priors) keep the
skills up to date automatically. Everything else (SKILL.md files, the other
references, the bundled scripts) is hand-maintained.

## Plugin-readiness rules

These skills are written so they can be packaged later as a distributable
Claude Code plugin (`.claude-plugin/plugin.json` + this directory as
`skills/`) for users who pip-install smefit without cloning this repo. When
editing skills, preserve these invariants:

1. **Self-contained skill dirs** — everything a skill needs at runtime
   (references, templates, scripts) lives inside its own directory; generated
   files are committed, not built on demand.
2. **No repo assumptions in skill bodies** — no absolute paths, no references
   to repo-root files (`CLAUDE.md`, `template_runcards/`, `smefit/…`), no
   `conda activate new_smefit` (except as an aside for the dev repo).
3. **Cross-skill references** — only via sibling-directory wording/paths
   (`../smefit-datasets/scripts/smefit_db.py` relative to a SKILL.md); the
   `skills/<name>/` layout survives plugin packaging.
4. **Scripts never import smefit** — stdlib + pyyaml only, so they work before
   the environment is set up.
5. **No skill-owned user state** — machine-specific configuration belongs to
   smefit itself (the git-ignored `.config/paths.yaml` managed by
   `smefit_setup_local`); skill scripts only read it, never write it.
