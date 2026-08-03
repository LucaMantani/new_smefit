"""RGE running of Wilson coefficients between the initial and observable scale.

Split into:

- ``_patches``: monkey patches applied to ``wilson``/``ckmutil`` at import time.
- ``runner``: the :class:`RGE` runner — pure computation, no file I/O.
- ``matrix``: the on-disk data model, :class:`RGEMatrix`, and both directions of
  the ``rge_matrix.pkl`` payload layout.
- ``loading``: :func:`build_rge_matrix`, the top-level entry point tying the
  runner, the data model, and the scale-keyed cache together.

Everything below is re-exported here so ``from smefit.rge import <name>``
keeps working regardless of which submodule actually defines it.
"""

from . import _patches  # noqa: F401  (import solely for its monkey-patch side effect)
from .loading import build_rge_matrix, resolve_rge_matrices
from .matrix import RGEMatrix
from .runner import ALLOWED_SMEFT_ACCURACY, ALLOWED_YUKAWA, RGE

__all__ = [
    "RGE",
    "RGEMatrix",
    "ALLOWED_YUKAWA",
    "ALLOWED_SMEFT_ACCURACY",
    "build_rge_matrix",
    "resolve_rge_matrices",
]
