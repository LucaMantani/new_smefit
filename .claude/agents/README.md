# Claude Code agents for smefit

One custom subagent so far:

| Agent | Purpose |
|---|---|
| `smefit-fit-doctor` | Diagnose a failing/hanging/misbehaving `smefit` run: reproduce cheaply, cross-check against known failure modes, report root cause + fix |

Unlike the skills in `.claude/skills/`, which are single-shot reference
lookups (dataset discovery, runcard templating), a fit-diagnosis session is
multi-step and can generate a lot of noisy intermediate output (tracebacks,
sampler logs, repeated reproduction attempts). Isolating that in a subagent
keeps the main conversation clean — only the final root cause and fix come
back.

## Plugin-readiness rules

These agents are meant to ship alongside `.claude/skills/` in a future
distributable Claude Code plugin, as a sibling `agents/` directory. The same
invariants that `.claude/skills/README.md` documents for skills apply here:

1. **Self-contained** — an agent file should not depend on state outside
   what it's given as input at invocation time.
2. **No repo assumptions in agent bodies** — no absolute paths, no
   references to repo-root files (`CLAUDE.md`, `template_runcards/`), no
   `conda activate new_smefit` except as a dev-repo aside.
3. **Cross-skill references** — describe the *layout* (`skills/<name>/...`, a
   sibling of `agents/`), never a literal relative path. An agent's tool calls
   resolve against the user's working directory, **not** against its own
   definition file, so `../skills/...` silently points outside the repo. Have
   the agent resolve the location once with `Glob` (see step 0 of
   `smefit-fit-doctor.md`) and fall back to the `Skill` tool. Agents that need
   skill content must therefore carry `Glob` — and preferably `Skill` — in
   their allowlist.
4. **No agent-owned user state** — machine-specific configuration belongs to
   smefit itself (`.config/paths.yaml`); agents only read it, never write it.

## Tools-allowlist convention

Diagnostic agents (read-only investigation, recommend rather than act) get
`Bash, Read, Grep, Glob, Skill` — deliberately no `Edit`/`Write`, so a
diagnosis session can never modify the user's runcard or output files.
`Glob` and `Skill` are what make rule 3 above workable: `Glob` to resolve the
skills directory, `Skill` as the fallback when it isn't co-located. If a
future agent's job is genuinely to make changes, give it `Edit`/`Write`
explicitly and say so in its description; don't grant it by default.

These files are hand-maintained — there is no auto-generation or CI
freshness check for `.claude/agents/` (unlike several files under
`.claude/skills/`, which are generated from the code by
`scripts/generate_skill_reference.py`).
