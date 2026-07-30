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

## Using SMEFiT with Claude (no coding required)

This repository ships instructions that teach [Claude
Code](https://code.claude.com/docs/en/overview) how SMEFiT works —
which datasets exist, what every runcard key means, how to run a fit and read
the result. With them you can set up and run an analysis by describing what you
want in plain English, instead of learning the YAML format and the command line
first.

You still need the physics judgement. Claude handles the mechanics.

### 1. One-time setup

1. **Install SMEFiT** following the [Installation](#installation) section above,
   then run `smefit_setup_local` once (see
   [`LOCAL_SETUP.md`](LOCAL_SETUP.md)). This tells SMEFiT where the datasets
   live. If you skip it, Claude will offer to walk you through it.
2. **Install Claude Code** and sign in with your Claude account. On
   macOS/Linux the one-line install is

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   ```

   For Windows, Homebrew, or the app versions, see the [official install
   guide](https://code.claude.com/docs/en/overview).
3. **Open this repository with it.** The instructions live in this folder, so
   Claude only picks them up when it is pointed at the folder. Anywhere else, it
   knows nothing about SMEFiT.

#### Which version of Claude to use

The skills are stored in this repository (under `.claude/`), so what matters is
whether the tool you use reads them.

| How you use Claude | Works here? | Notes |
|---|---|---|
| **Terminal** (`claude` in the repo folder) | Yes | The reference setup. Everything works, including running fits. |
| **Desktop app** (Mac/Windows) | Yes | Open this folder as your project. Good if you would rather not use a terminal. |
| **VS Code / JetBrains extension** | Yes | Same as the terminal, inside your editor. |
| **Web** ([claude.ai/code](https://claude.ai/code)) | Partly | Cloud sessions do load this repository's skills, so runcard authoring and dataset questions work. But the cloud machine is not *your* machine: your conda environment, your local database clone and your paths configuration are not there, so use it for preparing and understanding, not for running fits on your own data. |
| **[Claude Cowork](https://claude.com/product/cowork)** | **No** | Cowork loads only the skills enabled on your claude.ai account — [it does not read a repository's `.claude/skills/`](https://code.claude.com/docs/en/skills). The SMEFiT skills will not be there, so Claude will fall back on guesswork about datasets and runcard keys. Use one of the options above instead. |

For the terminal, that means:

```bash
cd /path/to/new_smefit
conda activate new_smefit
claude
```

Activating the conda environment first is what lets Claude actually run `smefit`
commands for you.

To check it worked, ask: *"What SMEFiT skills do you have available?"* — it
should list `smefit-runcard`, `smefit-datasets` and `smefit-analysis`. If it
does not, see [If it is not working](#4-if-it-is-not-working) below.

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

Claude picks the datasets from the real catalogue rather than inventing names,
and checks the finished runcard with a validator before handing it to you.

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

For these, Claude hands the problem to a specialist helper (the
`smefit-fit-doctor`) that reproduces the issue cheaply, finds the cause and
reports back a suggested fix. It is deliberately unable to edit your files, so
a diagnosis can never overwrite your runcard or results.

### 3. Good habits

- **Say what you want, not how to do it.** "Fit the top sector with quadratics"
  works better than guessing at key names.
- **Ask it to explain.** *"Why did you choose `use_t0: True`?"* or *"What does
  `logz` mean here?"* — the explanation is as much a deliverable as the runcard.
- **Fits cost time and CPU.** A nested-sampling run can take hours. Ask for a
  cheap check first: *"Do a quick analytic fit to see if the setup is sane
  before running the sampler."*
- **Claude asks before doing anything disruptive** — cloning the database,
  changing your path configuration, starting a long run. If you are unsure what
  a step does, ask before approving it.
- **Check the physics yourself.** Claude gets the mechanics right far more often
  than the judgement. Datasets, operator basis, priors and whether a result is
  believable are still your call. If a number looks surprising, ask how it was
  computed.

### 4. If it is not working

| Symptom | Fix |
|---|---|
| Claude does not seem to know anything about SMEFiT | Either it was started outside the repository — quit, `cd` into `new_smefit` and run `claude` again — or you are in Cowork, which does not read this repository's skills (see the table above). |
| `smefit: command not found` | The conda environment is not active. Run `conda activate new_smefit` before `claude`. |
| Errors about paths, or "no database found" | The local setup has not been run — ask Claude to help you through `smefit_setup_local`. |
| Answers about datasets look made up | Ask it to check against the database. It has a script for exactly that and should never answer catalogue questions from memory. |

For contributors: the instructions themselves live under `.claude/` — see
[`.claude/skills/README.md`](.claude/skills/README.md) and
[`.claude/agents/README.md`](.claude/agents/README.md).
