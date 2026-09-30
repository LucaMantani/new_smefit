"""Numpy evaluator for the built surrogate.  Imports neither flavio nor smelli."""

import pickle
import warnings

import numpy as np

from . import coords as C
from . import design, df2, ew, families, jms


def _assemble(xp, n, pieces):
    """Scatter per-family blocks into the full prediction vector."""
    if xp is np:
        out = np.empty(n)
        for idx, vals in pieces:
            out[idx] = vals
        return out
    out = xp.zeros(n)
    for idx, vals in pieces:
        out = out.at[np.asarray(idx)].set(vals)
    return out


class Surrogate:

    def __init__(self, path):
        with open(path, "rb") as fh:
            t = pickle.load(fh)
        self.t = t
        self.meta = t["meta"]
        self.coords = C.Coords(self.meta["coords_spec"])
        self.obs = self.meta["observables"]
        self.mu, self.err, self.corr_inv = t.get("mu"), t["err"], t.get("corr_inv")
        # a plain-likelihood block (PLAN 12) carries its constraint terms instead
        # of a Gaussian pseudo-measurement
        self.block = self.meta.get("block", "fast_likelihood_quarks.yaml")
        self.terms = ew.pack(t["terms"]) if t.get("terms") is not None else None
        self.df2 = df2.Df2(t["df2"])
        allf = [
            f
            for f in families.build_families(None, self.coords, self.obs)
            if f.name in t["coef"]
        ]
        # auxiliary families feed the CKM solver instead of a prediction slot
        self.fams = [f for f in allf if not f.aux]
        self.aux_fams = [f for f in allf if f.aux]
        self.degrees = self.meta.get("degrees", {})
        self.dlnv_full = self.meta.get("dlnv_full", {})
        pos = {str(o): i for i, o in enumerate(self.obs)}
        self.slots = {
            f.name: np.array([pos[str(o)] for o in f.obs], dtype=int) for f in self.fams
        }
        # the Delta F=2 observables are not fitted at all -- df2.py computes them
        # in closed form and they are written straight into their slots
        self.df2_slots = [
            (name, pos[name])
            for name in families.df2_observables(self.coords)
            if name in pos
        ]
        self.gamma_range = (
            float(np.min(self.coords.gamma_nodes)),
            float(np.max(self.coords.gamma_nodes)),
        )
        self.jms_names = t.get("jms_names")
        self.jms_T = t.get("jms_T")
        self.jms_T_dlnv = t.get("jms_T_dlnv")
        self.units = self.coords.units
        self.n_gam = len(self.coords.gamma_nodes)
        self._sub_idx = {
            f.name: (
                self.coords.sector_index(*f.sectors)
                if f.sectors
                else np.zeros(0, dtype=int)
            )
            for f in self.fams + self.aux_fams
        }
        self._blocks = self._merge_blocks()

    def _merge_blocks(self):
        """Group families that share a coordinate block into one matrix product.

        Families over the same coordinates and degree (the eight b->s ones, for
        instance) evaluate the *same* feature vector and differ only in their
        coefficients, so their tables concatenate along the output axis and the
        whole group becomes a single (features x outputs) matmul instead of one
        `einsum` each.  Exactly the same arithmetic -- agreement is 3e-16 -- but
        ~16x faster on CPU, because BLAS gets one big product rather than twenty
        small ones with a materialised outer product each.
        """
        by_key = {}
        for f in self.fams + self.aux_fams:
            coef = self.t["coef"][f.name]
            key = (
                tuple(self._sub_idx[f.name].tolist()),
                self.degrees.get(f.name, 2),
                coef.shape[0],
                coef.shape[1],
            )
            by_key.setdefault(key, []).append(f.name)
        blocks = []
        for (idx, deg, n_gam, n_feat), names in by_key.items():
            W = np.concatenate(
                [
                    np.asarray(self.t["coef"][n]).reshape(n_gam * n_feat, -1)
                    for n in names
                ],
                axis=1,
            )
            slices, i = [], 0
            for n in names:
                o = np.asarray(self.t["coef"][n]).shape[2]
                slices.append((n, i, i + o))
                i += o
            blocks.append(
                dict(
                    idx=np.asarray(idx, dtype=int),
                    n=len(idx),
                    deg=deg,
                    n_gam=n_gam,
                    W=W,
                    slices=slices,
                )
            )
        return blocks

    def _all_elem(self, x, dg, xp, dlnv=None):
        """Every family's elementary quantities, block by block."""
        out = {}
        for b in self._blocks:
            q = design.quad_features_1d(x[b["idx"]], b["n"], xp, degree=b["deg"])
            g = design.gamma_features_1d(dg, b["n_gam"] - 1, xp)
            vals = (g[:, None] * q[None, :]).reshape(-1) @ xp.asarray(b["W"])
            for name, lo, hi in b["slices"]:
                out[name] = vals[lo:hi]
        if dlnv is not None:
            for b in self._blocks:
                for name, _, _ in b["slices"]:
                    out[name] = self._apply_dlnv(name, out[name], x, dg, xp, dlnv)
        return out

    def _apply_dlnv(self, name, elem, x, dg, xp, dlnv):
        """The CKM axis beyond gamma, as in `_elem` (see there)."""
        coef = (self.t.get("dlnv") or {}).get(name)
        if coef is None:
            return elem
        if self.dlnv_full.get(name):
            idx = self._sub_idx[name]
            q = design.quad_features_1d(
                x[idx], len(idx), xp, degree=self.degrees.get(name, 2)
            )
            g = design.gamma_features_1d(dg, self.n_gam - 1, xp)
            d = xp.einsum("g,q,gkqo->ko", g, q, xp.asarray(coef))
            return elem + xp.tensordot(dlnv, d, axes=(0, 0))
        return df2.apply_dlnv(elem, xp.asarray(coef), dlnv, dg, xp)

    # ----------------------------------------------------------------
    def delta_gamma(self, x, xp=np):
        x = xp.asarray(x)
        return self._df2(xp).solve_gamma(x, nockm=self.nockm(x, xp))

    # `_elem` is the per-family form: the reference implementation, kept because
    # the build and validation tools call it family by family.  `predict` goes
    # through `_all_elem`, which is the same arithmetic merged into one product
    # per coordinate block.
    def _elem(self, f, x, dg, xp, dlnv=None):
        idx = self._sub_idx[f.name]
        q = design.quad_features_1d(
            x[idx], len(idx), xp, degree=self.degrees.get(f.name, 2)
        )
        g = design.gamma_features_1d(dg, self.n_gam - 1, xp)
        elem = xp.einsum("g,q,gqo->o", g, q, xp.asarray(self.t["coef"][f.name]))
        # the CKM axis beyond gamma: Vus/Vcb/Vub move too once charged-current
        # coordinates are on, and the tables are built at their SM values
        coef = (self.t.get("dlnv") or {}).get(f.name)
        if coef is None or dlnv is None:
            return elem
        if self.dlnv_full.get(f.name):
            # absolute derivative, a polynomial in the coordinates like the
            # quantity itself: (n_gam, 3, n_feat, n_out)
            d = xp.einsum("g,q,gkqo->ko", g, q, xp.asarray(coef))
            return elem + xp.tensordot(dlnv, d, axes=(0, 0))
        return df2.apply_dlnv(elem, xp.asarray(coef), dlnv, dg, xp)

    def nockm(self, x, xp=np):
        """The three CKM-input `nockm` predictions, or None if inert here.

        They are exactly CKM-independent (df2.py), so they are read off at
        delta gamma = 0 and the gamma fixed point does not couple back into them.
        """
        for f in self.aux_fams:
            if f.name == "ckm_inputs":
                return self._elem(f, xp.asarray(x), xp.zeros(()) * 1.0, xp)
        return None

    def _df2(self, xp):
        return self.df2 if xp is np else df2.Df2(self.t["df2"], xp=xp)

    def predict(self, x, dg=None, xp=np):
        """Predictions for the 352 observables.

        `xp` is the array module: numpy by default, `jax.numpy` for the JAX
        evaluator.  The whole path -- the gamma fixed point included -- is
        branch-free and has a fixed trip count, so it traces and jits.
        """
        x = xp.asarray(x)
        d2f = self._df2(xp)
        nockm = self.nockm(x, xp)
        dlnv = d2f.dlnv(nockm)
        if dg is None:
            dg = d2f.solve_gamma(x, nockm=nockm)
        if xp is np and not self.gamma_range[0] <= dg <= self.gamma_range[1]:
            warnings.warn(
                "delta gamma = %.4f outside the tabulated range %s; "
                "the surrogate is extrapolating" % (dg, self.gamma_range)
            )
        ctx = d2f.context(x, dg, dlnv)
        elems = self._all_elem(x, dg, xp, dlnv)
        pieces = []
        for f in self.fams:
            pieces.append((self.slots[f.name], f.rebuild(elems[f.name], ctx, xp)))
        for name, slot in self.df2_slots:
            pieces.append((np.array([slot]), xp.stack([ctx[name]])))
        return _assemble(xp, len(self.obs), pieces)

    def chi2(self, x, dg=None, xp=np):
        return self.chi2_pred(self.predict(x, dg=dg, xp=xp), xp=xp)

    def chi2_pred(self, pred, xp=np):
        """The block's chi2 of a prediction vector, in smelli's convention.

        Quark block: the Gaussian pseudo-measurement, (Delta/sigma)^T corr^-1
        (Delta/sigma).  Plain block: the constraint sum of `ew.chi2`, i.e.
        -2 [ln p(pred) - ln p(central)] per constraint -- smelli's `chi2_dict`.
        """
        if self.terms is not None:
            return ew.chi2(pred, self.terms, xp)
        d = (xp.asarray(pred) - xp.asarray(self.mu)) / xp.asarray(self.err)
        return d @ xp.asarray(self.corr_inv) @ d

    # ----------------------------------------------------------------
    # the external contract: WET coefficients in the JMS basis at coords.scale
    # ----------------------------------------------------------------
    def from_jms(self, values, dg=None):
        """JMS input -> internal coordinate vector (and the delta gamma it implies).

        `values` is a dict of complex JMS coefficients (``Wilson.match_run(...).dict``
        or rgevolve's output), a dict of ``'<name>_R'``/``'_I'`` floats, or a
        vector ordered like ``self.jms_names``.  Returns ``(x, dg)``.
        """
        if self.jms_T is None:
            raise RuntimeError(
                "these tables carry no JMS map; add jms_sectors "
                "to the coordinate file and rebuild"
            )
        if isinstance(values, dict) and any(
            np.iscomplexobj(v) or isinstance(v, complex) for v in values.values()
        ):
            y = jms.jms_dict_to_vector(self.jms_names, values)
        else:
            y = values
        # the Delta F=2 block carries no CKM factor, so x_bsbs does not move with
        # gamma and the fixed point does not couple back into the input map
        x0 = jms.apply_map(self.jms_T, self.jms_names, y, 0.0)
        nockm = self.nockm(x0)
        dlnv = self.df2.dlnv(nockm)
        if dg is None:
            dg = self.df2.solve_gamma(x0, nockm=nockm)
        return (
            jms.apply_map(self.jms_T, self.jms_names, y, dg, self.jms_T_dlnv, dlnv),
            dg,
        )

    # ----------------------------------------------------------------
    # parameter coordinates (PLAN 11)
    # ----------------------------------------------------------------
    @property
    def par_names(self):
        """Names of the parameter coordinates, e.g. ``B+->pi+ deltaC9 a1_R``."""
        return [self.coords.names[i] for i in self.coords.par_index]

    def with_par(self, x, values=None, **kw):
        """Copy of `x` with parameter coordinates set, in **internal units**.

        `values` is a dict keyed either by the coordinate name
        (``'B+->pi+ deltaC9 a1_R'``) or by flavio's own spelling
        (``'B+->pi+ deltaC9 a1 Re'``).  Unknown keys raise, so a typo in a
        parameter name cannot silently do nothing.

        Physical units are `x_i * coords.units[i]`; `coords.x_to_par(x)` gives
        the physical dict back.
        """
        values = dict(values or {}, **kw)
        x = np.array(x, dtype=float, copy=True)
        alias = {}
        for i in self.coords.par_index:
            base, part = self.coords.pars[i - self.coords.n_wc]
            alias[self.coords.names[i]] = i
            alias["{} {}".format(base, "Re" if part == "R" else "Im")] = i
        for k, v in values.items():
            if k not in alias:
                raise KeyError(
                    "%r is not a parameter coordinate of %s; have %s"
                    % (k, self.meta["coords"], self.par_names)
                )
            x[alias[k]] = v
        return x

    # ----------------------------------------------------------------
    def jms_matrix(self, dg=0.0, xp=np):
        """The JMS -> internal-coordinate matrix at a given delta gamma."""
        return design.poly_eval(xp.asarray(self.jms_T), dg, xp)

    def predict_jms(self, values):
        x, dg = self.from_jms(values)
        return self.predict(x, dg=dg)

    def chi2_jms(self, values):
        x, dg = self.from_jms(values)
        return self.chi2(x, dg=dg)
