# SMEFiT

![Tests badge](https://github.com/LucaMantani/new_smefit/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/LucaMantani/new_smefit/graph/badge.svg?token=168VE5BKM9)](https://codecov.io/gh/LucaMantani/new_smefit)

## Installation

```
conda create -n new_smefit
conda activate new_smefit
conda install python
conda install pre-commit
conda install pandoc
pip install -e .
```

### Local setup and server access

After installing the package, follow the instructions in [`LOCAL_SETUP.md`](LOCAL_SETUP.md) to configure the default paths for datasets and results.

For instructions on using the server, see [`SERVER.md`](SERVER.md).

### GPU (CUDA) Support

To enable GPU acceleration, install JAX with CUDA 12 (or 13 if available) support:

```bash
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

> **Note:** This step is optional. If no GPU is available, JAX will fall back to CPU automatically.

## Toggling experimental uncertainties per dataset

Two optional keys, `stat_unc` and `syst_unc`, can be set on any dataset entry
in a runcard to opt out of statistical or systematic uncertainties for that
dataset. This mirrors the existing `theory_cov: zero` opt-out for the theory
covariance matrix.

```yaml
datasets:
  - {name: DATASET_A, order: LO, stat_unc: zero}                                    # stat off
  - {name: DATASET_B, order: LO, syst_unc: zero}                                    # syst off
  - {name: DATASET_C, order: LO, stat_unc: zero, syst_unc: zero, theory_cov: zero}  # everything off
```

- Default (key absent, or explicit `current`): the uncertainty is used as loaded
  from the dataset file — unchanged behavior.
- `zero`: the corresponding uncertainty is set to zero for that dataset only.
  A warning is logged whenever this happens.
- Any other value raises a `ValueError`.

### Singular covariance matrix guard

Zeroing out both `stat_unc` and `syst_unc` (and `theory_cov`) for a dataset
makes its diagonal block of the fit covariance matrix all zeros, i.e.
singular — which would kill the fit. `produce_fit_covmat` checks every
dataset's diagonal block individually and raises a single `ValueError`
listing **all** datasets with a singular block, rather than stopping at the
first one found.


## Using SMEFiT with an AI assistant (no coding required)

This repository ships instructions that teach an AI coding assistant how SMEFiT
works — which datasets exist, what every runcard key means, how to run a fit and
read the result. With them you can set up and run an analysis by describing what
you want in plain English, instead of learning the YAML format and the command
line first.

The instructions are written as [Agent
Skills](https://agentskills.io), an open format, so they work with
**[Claude Code](https://code.claude.com/docs/en/overview)** (the reference
setup), **[OpenAI Codex CLI](https://developers.openai.com/codex/cli)** and
**[Gemini CLI](https://geminicli.com)** alike. The guide below says "the
assistant" throughout; substitute whichever one you use.

You still need the physics judgement. The assistant handles the mechanics.

### 1. One-time setup

1. **Install SMEFiT** following the [Installation](#installation) section above,
   then run `smefit_setup_local` once (see
   [`LOCAL_SETUP.md`](LOCAL_SETUP.md)). This tells SMEFiT where the datasets
   live. If you skip it, the assistant will offer to walk you through it.
2. **Install an assistant** and sign in. Any one of:

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash   # Claude Code
   npm install -g @openai/codex                     # OpenAI Codex CLI
   npm install -g @google/gemini-cli                # Gemini CLI
   ```

   For Windows, Homebrew, or the app versions of Claude Code, see the [official
   install guide](https://code.claude.com/docs/en/overview).
3. **Open this repository with it.** The instructions live in this folder, so
   they are only picked up when the assistant is pointed at the folder.
   Anywhere else, it knows nothing about SMEFiT.

#### Which tool to use

The skills are stored in this repository (under `.agents/skills/`, with
`.claude/skills` symlinked to it), so what matters is whether the tool you use
reads them.

| Tool | Works here? | Notes |
|---|---|---|
| **Claude Code — terminal** (`claude` in the repo folder) | Yes | The reference setup. Everything works, including running fits. |
| **Claude Code — desktop app** (Mac/Windows) | Yes | Open this folder as your project. Good if you would rather not use a terminal. |
| **Claude Code — VS Code / JetBrains** | Yes | Same as the terminal, inside your editor. |
| **OpenAI Codex CLI** (`codex` in the repo folder) | Yes | Reads `.agents/skills/` and `AGENTS.md` with no configuration. The `smefit-fit-doctor` subagent is Claude Code-only; everything else is identical. |
| **Gemini CLI** (`gemini` in the repo folder) | Yes | Same — `.agents/skills/` and `AGENTS.md` are picked up automatically. Same subagent caveat. |
| **Claude Code — web** ([claude.ai/code](https://claude.ai/code)) | Partly | Cloud sessions do load this repository's skills, so runcard authoring and dataset questions work. But the cloud machine is not *your* machine: your conda environment, your local database clone and your paths configuration are not there, so use it for preparing and understanding, not for running fits on your own data. |
| **[Claude Cowork](https://claude.com/product/cowork)**, ChatGPT and Gemini **web apps** | **No** | These load only the skills enabled on your own account — they do not read a repository's skills directory, and they have no access to your machine, your conda environment or your database clone. The assistant will fall back on guesswork about datasets and runcard keys. Use one of the terminal options above instead. |

For the terminal, that means:

```bash
cd /path/to/new_smefit
conda activate new_smefit
claude          # or: codex   /   gemini
```

Activating the conda environment first is what lets the assistant actually run
`smefit` commands for you.

To check it worked, ask: *"What SMEFiT skills do you have available?"* — it
should list `smefit-runcard`, `smefit-datasets` and `smefit-analysis`. If it
does not, see [If it is not working](#4-if-it-is-not-working) below.

**If you would rather not work inside this clone**, install the skills for your
user account instead — they are then available in any directory:

```bash
# Codex CLI and Gemini CLI
mkdir -p ~/.agents/skills && cp -r /path/to/new_smefit/.agents/skills/smefit-* ~/.agents/skills/

# Claude Code reads ~/.claude/skills/ instead
mkdir -p ~/.claude/skills && cp -r /path/to/new_smefit/.agents/skills/smefit-* ~/.claude/skills/
```

Re-copy after pulling this repository — the copies do not update themselves. Note
that the dataset and validator scripts still need SMEFiT installed and
`smefit_setup_local` run.

### 2. What you can ask for

You do not need to name the skills or use special syntax. Ask in ordinary
language and the relevant instructions load themselves.

**Finding data and operators**

> Which datasets are available for top quark pair production?
>
> What does the dataset `ATLAS_tt_13TeV_asy_2022` measure, and which operators
> does it constrain?
>
> Is the operator `OtG` implemented?

**Setting up a fit**

> Build me a runcard that fits `OtG` and `OtW` to the ATLAS and CMS top-pair
> data at NLO, including quadratic terms.
>
> Take my existing runcard and add the LEP electroweak precision observables.
>
> I want to use nested sampling instead of the analytic fit — what do I need to
> change?

The assistant picks the datasets from the real catalogue rather than inventing
names, and checks the finished runcard with a validator before handing it to you.

**Running and understanding results**

> Run this runcard and tell me what the results mean.
>
> What is the 95% credible interval on `OtG` from this fit?
>
> My chi2 per degree of freedom is 3.5 — is that a problem?
>
> Compare these two fits and tell me which model the data prefers.

**When something goes wrong**

> My fit crashed with this error: [paste the error]
>
> The fit has been running for two hours and seems stuck.
>
> The posterior for this coefficient looks identical to the prior.

In Claude Code, these get handed to a specialist helper (the
`smefit-fit-doctor`) that reproduces the issue cheaply, finds the cause and
reports back a suggested fix. It is deliberately unable to edit your files, so
a diagnosis can never overwrite your runcard or results. Codex CLI and Gemini
CLI have no equivalent, so they diagnose in the main conversation instead —
working from the same troubleshooting reference, with more intermediate output
on screen.

### 3. Good habits

- **Say what you want, not how to do it.** "Fit the top sector with quadratics"
  works better than guessing at key names.
- **Ask it to explain.** *"Why did you choose `use_t0: True`?"* or *"What does
  `logz` mean here?"* — the explanation is as much a deliverable as the runcard.
- **Fits cost time and CPU.** A nested-sampling run can take hours. Ask for a
  cheap check first: *"Do a quick analytic fit to see if the setup is sane
  before running the sampler."*
- **It asks before doing anything disruptive** — cloning the database,
  changing your path configuration, starting a long run. If you are unsure what
  a step does, ask before approving it.
- **Check the physics yourself.** These tools get the mechanics right far more
  often than the judgement. Datasets, operator basis, priors and whether a result
  is believable are still your call. If a number looks surprising, ask how it was
  computed.

### 4. If it is not working

| Symptom | Fix |
|---|---|
| It does not seem to know anything about SMEFiT | Either it was started outside the repository — quit, `cd` into `new_smefit` and start it again — or you are using a web app, which does not read this repository's skills (see the table above). |
| Claude Code specifically finds no skills, but Codex/Gemini do | Your clone did not create symlinks (common on Windows). `.claude/skills` should be a link to `../.agents/skills`, not a text file. Fix with `git config --global core.symlinks true` and re-clone, or use the `~/.claude/skills/` copy shown above. |
| `smefit: command not found` | The conda environment is not active. Run `conda activate new_smefit` before starting the assistant. |
| Errors about paths, or "no database found" | The local setup has not been run — ask it to help you through `smefit_setup_local`. |
| Answers about datasets look made up | Ask it to check against the database. It has a script for exactly that and should never answer catalogue questions from memory. |

For contributors: the instructions themselves live in
[`.agents/skills/`](.agents/skills/README.md) (portable across tools) and
[`.claude/agents/`](.claude/agents/README.md) (Claude Code-only subagents);
[`AGENTS.md`](AGENTS.md) is the project context file, with `CLAUDE.md`
symlinked to it.
