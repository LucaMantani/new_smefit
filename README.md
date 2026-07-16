# SMEFiT

![Tests badge](https://github.com/LucaMantani/new_smefit/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
