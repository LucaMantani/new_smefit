"""RGE running of Wilson coefficients between the initial and observable scale.

Split into:

- ``_patches``: monkey patches applied to ``wilson``/``ckmutil`` at import time.
- ``runner``: the :class:`RGE` runner — pure computation, no file I/O.
- ``matrix``: the data model — :class:`RGEMatrix`, the per-data-point matrices a
  run computes, and :class:`RGECache`, the scale-keyed ``rge_matrix.pkl`` they
  are stored as — and both directions of that payload layout.
- ``build``: :func:`build_rge_matrix`, the top-level entry point tying the
  runner, the data model, and the scale-keyed cache together.

Everything below is re-exported here so ``from smefit.rge import <name>``
keeps working regardless of which submodule actually defines it.
"""

from . import _patches  # noqa: F401  (import solely for its monkey-patch side effect)
from .build import build_rge_matrix, resolve_rge_matrices
from .matrix import RGECache, RGEMatrix
from .runner import ALLOWED_SMEFT_ACCURACY, ALLOWED_YUKAWA, RGE

__all__ = [
    "RGE",
    "RGECache",
    "RGEMatrix",
    "ALLOWED_YUKAWA",
    "ALLOWED_SMEFT_ACCURACY",
    "build_rge_matrix",
    "resolve_rge_matrices",
]
