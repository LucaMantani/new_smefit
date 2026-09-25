r"""smelli's likelihoods for smefit, from the precomputed LEFT/EW surrogate.

Four smelli blocks, one class each, named after the likelihood it reproduces.

    external_chi2:
      fast_likelihood_quarks:      # 352 obs
        path: <this file>
      likelihood_lfu_fcnc:         #  17 obs (R_K, R_K*, R_phi, D_P', tau tau; q2 <= 6)
        path: <this file>
      likelihood_ewpt:             #  28 obs
        path: <this file>
      likelihood_eell:             # 148 obs (LEP2 e+e- -> ll)
        path: <this file>


It loads the pre-computed interpolated smelli predictions from:

    surrogate/      the evaluator (uv, evaluate, ew + coords, design, df2,
                    families, jms); numpy and jax only
    tables/         S2 (quarks), S2L (LFU), EW1 (EWPT + LEP2), ~39 MB

It is currently designed for a specific UV fit:
    Zp_model.yaml   the NFU Z' model card these tables were built for

It needs be extended for the SMEFT.
"""

import hashlib
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")
MODEL = os.path.join(HERE, "Zp_model.yaml")

if HERE not in sys.path:  # `surrogate/` is a sibling of this file
    sys.path.insert(0, HERE)

from surrogate import design, ew, uv  # noqa: E402
from surrogate.evaluate import Surrogate  # noqa: E402


def _abs(path, default):
    """A runcard path, absolute or relative to this directory."""
    path = os.path.expanduser(path or default)
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


# ---------------------------------------------------------------------------
# shared: runcard coefficients -> UV parameters -> Warsaw vector
# ---------------------------------------------------------------------------


class _ZprimeBlock:
    """UV parameters from the runcard, and the Warsaw (Re, Im) vector they give."""

    def __init__(self, coefficients, rge_dict=None, model=None, **_):
        self._charges, self._qq1, self._defaults, _ = uv.load_model(_abs(model, MODEL))
        self.M = float(self._defaults["M"])
        self.param_names = list(coefficients.free_names)  # smefit's contract
        # the Warsaw coefficients this model generates, in a fixed order
        probe = uv.smeft_wcs(
            {**self._defaults, "g": 0.2, "tsb": 0.1, "tdb": 0.1, "tds": 0.1},
            self._charges,
            self._qq1,
            np,
        )
        self.warsaw = sorted(k for k, v in probe.items() if abs(v) > 0)
        # how each UV parameter is obtained from smefit's free-parameter vector
        by_name = {c.name: c for c in coefficients.coefficients}
        pos = {n: i for i, n in enumerate(coefficients.free_names)}
        self._rules = []
        for n in uv.UV_PARAMS:
            c = by_name.get(n)
            if c is None:
                self._rules.append((n, "const", float(self._defaults[n])))
            elif c.free:
                self._rules.append((n, "free", pos[n]))
            elif c.value is not None:
                self._rules.append((n, "const", float(c.value)))
            else:
                self._rules.append((n, "expr", (c, [pos[v] for v in c.vars])))

    def _uv(self, values):
        p = {"M": self.M}
        for n, kind, arg in self._rules:
            if kind == "free":
                p[n] = values[arg]
            elif kind == "const":
                p[n] = arg
            else:
                c, idx = arg
                p[n] = c.constrain(*[values[i] for i in idx])
        return p

    def _warsaw_vector(self, values):
        wcs = uv.smeft_wcs(self._uv(jnp.asarray(values)), self._charges, self._qq1, jnp)
        return jnp.stack([f(wcs[n]) for n in self.warsaw for f in (jnp.real, jnp.imag)])


# ---------------------------------------------------------------------------
# the two flavour blocks: Warsaw(M) -> JMS(M_Z) -> LEFT surrogate tables
# ---------------------------------------------------------------------------


class _FlavourBlock(_ZprimeBlock):
    SUPPORT = None

    def __init__(
        self,
        coefficients,
        rge_dict=None,
        model=None,
        tables=None,
        penalty=1e4,
        dg_tol=2.0,
        **kw
    ):
        super().__init__(coefficients, rge_dict, model, **kw)
        path = _abs(tables, os.path.join(TABLES, self.SUPPORT, "tables.pkl"))
        self.S = Surrogate(path)
        self.num_data = len(self.S.obs)
        self._W2J = jnp.asarray(self._warsaw_to_jms(path))
        self._jT = jnp.asarray(self.S.jms_T)
        self._jTd = (
            None if self.S.jms_T_dlnv is None else jnp.asarray(self.S.jms_T_dlnv)
        )
        self._penalty = float(penalty)
        self._dg_max = float(dg_tol) * float(np.max(np.abs(self.S.coords.gamma_nodes)))

    def _warsaw_to_jms(self, tables_path):
        """Warsaw(M) -> JMS(M_Z) matrix, from wilson; cached next to the tables.

        The key holds everything the matrix depends on (PLAN 12.4, item 5: a cache
        keyed on names alone was once reused stale).
        """
        import wilson

        m = self.S.df2.m
        ckm = {
            "Vus": m["Vus"],
            "Vcb": m["Vcb"],
            "Vub": m["Vub"],
            "gamma": m["gamma_sm"],
        }
        out = list(self.S.jms_names)
        key = hashlib.md5(
            repr(
                (
                    self.warsaw,
                    self.M,
                    out,
                    wilson.__version__,
                    "integrate",
                    sorted(ckm.items()),
                )
            ).encode()
        ).hexdigest()[:16]
        cache = os.path.join(os.path.dirname(tables_path), "w2j_%s.npy" % key)
        if os.path.exists(cache):
            return np.load(cache)
        probe = 1e-9  # linear map, so any small size; O(1) GeV^-2 breaks wilson
        W = np.zeros((len(out), 2 * len(self.warsaw)))
        for j, (n, part) in enumerate((n, p) for n in self.warsaw for p in (1.0, 1j)):
            w = wilson.Wilson(
                {n: probe * part}, scale=self.M, eft="SMEFT", basis="Warsaw"
            )
            w.set_option("smeft_accuracy", "integrate")
            opt = dict(w.get_option("parameters"))  # a copy: it is shared by reference
            opt.update(ckm)
            w.set_option("parameters", opt)
            d = w.match_run(self.S.coords.scale, "WET", "JMS").dict
            for i, name in enumerate(out):
                base, p = name.rsplit("_", 1)
                v = complex(d.get(base, 0.0)) / probe
                W[i, j] = v.real if p == "R" else v.imag
        try:
            np.save(cache, W)
        except OSError:
            pass
        return W

    def compute_chi2(self, values):
        y = self._W2J @ self._warsaw_vector(values)  # JMS at M_Z
        d2 = self.S._df2(jnp)
        x0 = design.poly_eval(self._jT, 0.0, jnp) @ y
        nockm = self.S.nockm(x0, jnp)
        # smelli re-extracts the CKM from the NP point: delta gamma, and the
        # JMS -> coordinate map moves with it (flavio's lambda_t normalisation)
        dg, excess = d2.solve_gamma(x0, with_excess=True, nockm=nockm)
        x = design.poly_eval(self._jT, dg, jnp) @ y
        if self._jTd is not None:
            dlnv = d2.dlnv(nockm)
            if dlnv is not None:
                x = x + jnp.tensordot(
                    dlnv, design.poly_eval(self._jTd, dg, jnp) @ y, axes=(0, 0)
                )
        out = jnp.maximum(jnp.abs(dg) - self._dg_max, 0.0) / self._dg_max
        return self.S.chi2(x, dg=dg, xp=jnp) + self._penalty * (out**2 + excess**2)


class fast_likelihood_quarks(_FlavourBlock):
    """smelli ``fast_likelihood_quarks.yaml``: 352 observables, SM chi2 501.468."""

    SUPPORT = "S2"


class likelihood_lfu_fcnc(_FlavourBlock):
    """smelli ``likelihood_lfu_fcnc.yaml`` (q2 <= 6): 17 observables, SM chi2 23.514."""

    SUPPORT = "S2L"


# ---------------------------------------------------------------------------
# the two electroweak blocks: Warsaw(M) -> EW tables, one smelli block each
# ---------------------------------------------------------------------------


class _EWBlock(_ZprimeBlock):
    BLOCK = None

    SUPPORT = "EW1"

    def __init__(self, coefficients, rge_dict=None, model=None, tables=None, **kw):
        super().__init__(coefficients, rge_dict, model, **kw)
        self.E = ew.EWSurrogate(
            _abs(tables, os.path.join(TABLES, self.SUPPORT, "tables.pkl")),
            blocks=(self.BLOCK,),
        )
        if abs(self.E.coords.scale - self.M) > 1e-6 * self.M:
            raise ValueError(
                "the EW tables are built at %g GeV, the model card has "
                "M = %g GeV" % (self.E.coords.scale, self.M)
            )
        self.num_data = len(self.E.obs)
        # EW coordinates are real, flavour-diagonal Warsaw coefficients: pick the
        # real slots of the Warsaw vector and rescale to internal units
        pos = {n: 2 * i for i, n in enumerate(self.warsaw)}
        names = self.E.coords.names
        self._idx = jnp.asarray([pos.get(n, 0) for n in names])
        self._sc = jnp.asarray(
            [(1.0 if n in pos else 0.0) for n in names]
        ) / jnp.asarray(self.E.coords.units)

    def compute_chi2(self, values):
        v = self._warsaw_vector(values)
        return self.E.chi2(v[self._idx] * self._sc, xp=jnp)


class likelihood_ewpt(_EWBlock):
    """smelli ``likelihood_ewpt.yaml``: 28 observables, SM chi2 36.772."""

    BLOCK = "likelihood_ewpt.yaml"


class likelihood_eell(_EWBlock):
    """smelli ``likelihood_eell.yaml`` (LEP2 e+e- -> ll): 148 observables, SM chi2 149.816."""

    BLOCK = "likelihood_eell.yaml"
