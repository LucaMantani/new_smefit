"""Monkey patches applied to ``wilson``/``ckmutil`` on import of :mod:`smefit.rge`.

Imported first by :mod:`smefit.rge` (before ``runner``, ``matrix``, ``loading``)
so every patch below is in effect before any RGE computation runs.
"""

from copy import deepcopy
from functools import partial, wraps

import ckmutil.ckm
import numpy as np
import wilson

### Patch of a CKM function, so that the CP violating
### phase is set to gamma and not computed explicitly
### See https://github.com/wilson-eft/wilson/issues/113#issuecomment-2179273979
### This needs to be done before the import of wilson

# copying so we keep the original function
ckm_tree = deepcopy(ckmutil.ckm.ckm_tree)
ckmutil.ckm.ckm_tree = partial(ckm_tree, delta_expansion_order=0)
### End of patch

##################### MONKEY PATCH
# switch off the SM - EFT mixing, since SMEFiT assumes that the
# RGE solution is linearised
# Keep a reference to the original beta function
original_beta = wilson.run.smeft.beta.beta


def beta_wrapper(C, HIGHSCALE=np.inf, *args, **kwargs):
    return original_beta(C, HIGHSCALE, *args, **kwargs)


wilson.run.smeft.beta.beta = beta_wrapper
wilson.run.smeft.beta.beta_array = partial(
    wilson.run.smeft.beta.beta_array, HIGHSCALE=np.inf
)
##################### END OF MONKEY PATCH

##################### MONKEY PATCH
# Patch smeftpar: we remove all dependence on SMEFT parameters
# in SM paramaters, otherwise the linear approximation is not valid
C_patch = {
    "phi": 0.0,
    "phiBox": 0.0,
    "phiD": 0.0,
    "phiWB": 0.0,
    "phiG": 0.0,
    "phiW": 0.0,
    "phiB": 0.0,
    "dphi": 0.0,
    "uphi": 0.0,
    "ephi": 0.0,
}
# Reference to the original smeftpar function
original_smeftpar = wilson.run.smeft.smpar.smeftpar


# Define the monkey-patched function
@wraps(original_smeftpar)
def patched_smeftpar(*args, **kwargs):
    # check if C is passed as a keyword argument
    if "C" in kwargs:
        kwargs["C"] = C_patch
    # otherwise, check if it is passed as a positional argument
    else:
        args = list(args)
        args[1] = C_patch

    return original_smeftpar(*args, **kwargs)


# Apply the monkey patch
wilson.run.smeft.smpar.smeftpar = patched_smeftpar
##################### END OF MONKEY PATCH


##################### MONKEY PATCH
# Monkey patch flavour rotation
# Define the new method
def _to_wcxf_no_rotation(self, C_out, scale_out):
    """Return the Wilson coefficients `C_out` as a wcxf.WC instance, without rotation."""
    # Skip the self._rotate_defaultbasis line
    d = wilson.util.smeftutil.arrays2wcxf_nonred(C_out)
    d = wilson.wcxf.WC.dict2values(d)
    wc = wilson.wcxf.WC("SMEFT", "Warsaw", scale_out, d)
    return wc


# Monkey-patch the method
wilson.run.smeft.classes.SMEFT._to_wcxf = _to_wcxf_no_rotation
##################### END OF MONKEY PATCH
