"""Shared physical constants at MZ."""

import numpy as np

alpha_s = 0.118
mw = 80.387
mz = 91.1876
gs = np.sqrt(4 * np.pi * alpha_s)
sw = np.sqrt(1 - mw**2 / mz**2)
cw = np.sqrt(1 - sw**2)
