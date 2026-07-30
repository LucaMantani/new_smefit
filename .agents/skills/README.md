# Agent skills for smefit

Five skills assist users and developers working with smefit:

| Skill | Scope | Purpose |
|---|---|---|
| `smefit-runcard` | user-facing | Author and validate runcards (keys, coefficients, priors, settings) |
| `smefit-datasets` | user-facing | Discover datasets/operators/external likelihoods in smefit_database |
| `smefit-analysis` | user-facing | Run fits, interpret output, reports/Fisher, troubleshooting |
| `smefit-dev` | maintainer-only | Extend smefit itself: reportengine nodes, runcard keys, actions, priors |
| `smefit-server` | maintainer-only | Develop the server/registry infrastructure |

The **scope** column matters for the plugin-readiness rules below: they apply to
the user-facing skills, which must work for someone who pip-installed smefit
without cloning this repo. The maintainer-only skills document the internals of
this repository — repo paths (`smefit/config.py`, `smefit/server_utils.py`,
`pyproject.toml`) are their subject matter, so rules 1–2 do not apply to them
and they would be excluded from a distributable plugin. They are the single
source of truth for their areas; `AGENTS.md` only points at them.

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

## Tests

`tests/test_skill_scripts.py` is part of the normal `pytest` suite (run by the
unconditional `tests.yml` workflow, not by `skills.yml`, which only gates
freshness). It runs both bundled scripts as subprocesses — the way a skill
invokes them — covering the validator's error paths against
`tests/fixtures/`, and asserting every `template_runcards/*.yaml` validates
with **zero warnings**. That last assertion is the drift alarm: a new runcard
or dataset-entry key used in a template but missing from the generated
`runcard-keys.json` shows up there as an "unknown key" warning. The
database-dependent tests skip when no `smefit_database` clone is configured.

## Portability

This directory is the canonical location. `.agents/skills/` (workspace) and
`~/.agents/skills/` (user) are the discovery paths of the [Agent Skills open
standard](https://agentskills.io), so **Codex CLI and Gemini CLI find these
skills with no configuration** — as do Cursor and other tools implementing the
standard. Claude Code has not adopted the standard directory yet
([anthropics/claude-code#31005](https://github.com/anthropics/claude-code/issues/31005)),
so a committed symlink `.claude/skills` → `../.agents/skills` covers it. Add new
skills here, never under `.claude/`.

Only `name` and `description` frontmatter is used, which is the portable subset;
progressive disclosure (short SKILL.md, detail in `references/`) is what every
adopting tool expects. The plugin-readiness invariants below therefore double as
the cross-tool portability contract.

Two consequences worth knowing:

- The generator, `tests/test_skill_scripts.py` and
  `.github/workflows/skills.yml` all address `.agents/skills`. If the symlink
  direction is ever inverted, those are the four places to change.
- A Windows clone made without symlink support turns `.claude/skills` into a
  plain text file; the real files are still here, so Codex/Gemini users are
  unaffected. See the root `README.md` for the user-facing fix.

## Plugin-readiness rules

These skills are written so they can be packaged later as a distributable
Claude Code plugin (`.claude-plugin/plugin.json` + this directory as
`skills/`) or a Gemini CLI extension, for users who pip-install smefit without
cloning this repo. When editing skills, preserve these invariants:

1. **Self-contained skill dirs** — everything a skill needs at runtime
   (references, templates, scripts) lives inside its own directory; generated
   files are committed, not built on demand. Some content is therefore
   *deliberately* duplicated across skills — `references/actions.md` is
   byte-identical in `smefit-runcard` and `smefit-analysis`, and both bundled
   scripts carry their own copy of `find_paths_config()`. Do not "deduplicate"
   these by hand: the generator writes both copies of `actions.md`, and the
   script helper is mirrored on purpose so each script runs standalone.
2. **No repo assumptions in skill bodies** — no absolute paths, no references
   to repo-root files (`AGENTS.md`, `template_runcards/`, `smefit/…`), no
   `conda activate new_smefit` (except as an aside for the dev repo).
3. **Cross-skill references** — only via sibling-directory wording/paths
   (`../smefit-datasets/scripts/smefit_db.py` relative to a SKILL.md); the
   `skills/<name>/` layout survives plugin packaging.
4. **Scripts never import smefit** — stdlib + pyyaml only, so they work before
   the environment is set up.
5. **No skill-owned user state** — machine-specific configuration belongs to
   smefit itself (the git-ignored `.config/paths.yaml` managed by
   `smefit_setup_local`); skill scripts only read it, never write it.
