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

# Toggling experimental uncertainties per dataset

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

## Singular covariance matrix guard

Zeroing out both `stat_unc` and `syst_unc` (and `theory_cov`) for a dataset
makes its diagonal block of the fit covariance matrix all zeros, i.e.
singular — which would kill the fit. `produce_fit_covmat` checks every
dataset's diagonal block individually and raises a single `ValueError`
listing **all** datasets with a singular block, rather than stopping at the
first one found.
