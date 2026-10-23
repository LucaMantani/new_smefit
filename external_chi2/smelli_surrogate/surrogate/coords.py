r"""Coordinate bookkeeping: the ordered real vector `x` <-> flavio inputs.

A coordinate is one real number.  Two kinds exist, and the order is fixed:

* **WC coordinates** (``sectors:``) -- Re/Im parts of flavio-basis WET Wilson
  coefficients.  They always come first, because `jms.py` builds its
  JMS -> internal matrix by indexing `Coords.wcs` against `len(Coords)`.
* **parameter coordinates** (``par_sectors:``) -- Re/Im parts of *flavio
  parameters*, appended after the WC block.  They exist because the subleading
  non-factorisable hadronic shifts (``B+->pi+ deltaC9 a1 Re`` and friends) enter
  flavio's amplitudes through the very same ``wc['v']`` slot as the WET C9, so
  the predictions stay exact quadratic forms in them -- they are coordinates of
  the surrogate in every sense except that they are set in `par`, not in the
  Wilson object.  See PLAN 11.

  Only parameters whose flavio default is an **exact zero with no uncertainty**
  belong here.  A parameter that still carries a sigma is marginalised by
  smelli into `sm_cov`, and making it a coordinate as well would double-count
  it.

Both kinds live in the same `sectors` dict, so `groups:` and every family's
`sector_index` work on them without knowing the difference.
"""

import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
COORD_DIR = os.path.join(os.path.dirname(HERE), "coords")


class Coords:
    """An ordered list of real coordinates (Re/Im parts of WET coefficients)."""

    def __init__(self, spec):
        self.spec = spec
        self.name = spec["name"]
        self.scale = float(spec["scale"])
        self.box = float(spec.get("box", 1.2))
        self.gamma_nodes = np.array(spec.get("gamma_nodes", [0.0]), dtype=float)
        self.names, self.wcs, self.units = [], [], []
        self.pars = []
        self.sectors = {}
        for sector, sspec in (spec["sectors"] or {}).items():
            unit = float(sspec.get("unit", 1.0))
            idx = []
            for wc in sspec["wcs"]:
                for part in ("R", "I"):
                    idx.append(len(self.names))
                    self.names.append("{}_{}".format(wc, part))
                    self.wcs.append((wc, part))
                    self.units.append(unit)
            self.sectors[sector] = idx
        # WC coordinates are now complete; `n_wc` is the boundary `jms.py` and
        # `x_to_wc` rely on.  Parameter coordinates follow.
        self.n_wc = len(self.names)
        for sector, sspec in (spec.get("par_sectors") or {}).items():
            default = float(sspec.get("unit", 1.0))
            entries = sspec["pars"]
            if not isinstance(entries, dict):
                entries = {p: default for p in entries}
            idx = []
            for base, unit in entries.items():
                for part in ("R", "I"):
                    idx.append(len(self.names))
                    self.names.append("{}_{}".format(base, part))
                    self.pars.append((base, part))
                    self.units.append(float(unit))
            if sector in self.sectors:
                raise ValueError("sector %r declared twice" % sector)
            self.sectors[sector] = idx
        self.units = np.array(self.units)
        # `groups` maps a family group name to the sectors that family must carry.
        # It is data, not code: PLAN 8.2 says measure the grouping, and the result
        # of that measurement lives in the coordinate file.  Sectors named in a
        # group but absent from `sectors` are silently dropped, so one family
        # table serves several supports.
        self.groups = {
            g: tuple(s for s in ss if s in self.sectors)
            for g, ss in (spec.get("groups") or {}).items()
        }

    def __len__(self):
        return len(self.names)

    def index(self, name):
        return self.names.index(name)

    def group(self, name, default=()):
        """Sectors of a family group, or `default` if the file declares none."""
        if name in self.groups:
            return self.groups[name]
        return tuple(s for s in default if s in self.sectors)

    def sector_index(self, *sectors):
        out = []
        for s in sectors:
            out.extend(self.sectors[s])
        return np.array(out, dtype=int)

    @property
    def par_index(self):
        """Indices of the parameter coordinates, in order."""
        return np.arange(self.n_wc, len(self.names), dtype=int)

    def x_to_wc(self, x):
        """Internal-unit vector -> dict of flavio-basis WET coefficients (physical)."""
        x = np.asarray(x, dtype=float)[: self.n_wc] * self.units[: self.n_wc]
        d = {}
        for (wc, part), v in zip(self.wcs, x):
            d[wc] = d.get(wc, 0j) + (v if part == "R" else 1j * v)
        return d

    def x_to_par(self, x):
        """Internal-unit vector -> dict of flavio *parameter* overrides (physical).

        Empty when the support declares no ``par_sectors``, so every caller can
        apply it unconditionally.  flavio spells the two parts ``... Re`` and
        ``... Im``, which is what the keys are.
        """
        if not self.pars:
            return {}
        v = np.asarray(x, dtype=float)[self.n_wc :] * self.units[self.n_wc :]
        return {
            "{} {}".format(base, "Re" if part == "R" else "Im"): float(val)
            for (base, part), val in zip(self.pars, v)
        }

    def sub(self, x, *sectors):
        return np.asarray(x, dtype=float)[self.sector_index(*sectors)]

    def random(self, rng, n=None, box=None):
        box = self.box if box is None else box
        shape = (len(self),) if n is None else (n, len(self))
        return rng.uniform(-box, box, size=shape)


def load(name="S1min"):
    path = name if os.path.sep in name else os.path.join(COORD_DIR, name + ".yaml")
    with open(path) as f:
        return Coords(yaml.safe_load(f))
