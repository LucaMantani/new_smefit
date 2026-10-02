"""Map between the SMEFiT operator basis and the Warsaw basis of WCxf.

Only the forward direction is written by hand, in :data:`SMEFIT_TO_WARSAW`::

    O_SMEFiT = sum_w  M[w, op] * O_Warsaw,w

so switching on a SMEFiT coefficient ``c`` switches on the Warsaw coefficients
``C_w = M[w, op] * c``. This direction always exists.

The inverse only exists on the image of ``M``, i.e. on Warsaw points respecting
the flavour symmetry the SMEFiT basis assumes. :class:`WarsawMap` derives it from
``M`` and checks both conditions it rests on: ``M`` must have full column rank
(no two SMEFiT operators are the same Warsaw direction), and a point mapped back
should lie in the image of ``M``.

On the image every left inverse agrees; off it a convention is needed. The
inverse reads, for each group of operators sharing Warsaw coefficients, the first
independent components in the order they are listed in the table. Entries list
generation 1 first, so a point breaking the flavour symmetry is mapped back from
its generation-1 components.

The SMEFiT basis is itself not flavour-consistent in the right-handed down
sector: ``Opdi``, ``O1dt``, ``O8dt``, ``O1qd`` and ``O8qd`` assume d, s and b
universal, while ``Obb``, ``Ol{1,2,3}b``, ``O{e,mu,ta}b`` (b only) and
``Ol{1,2,3}d``, ``O{e,mu,ta}d`` (d and s only) split b off. Running the latter
therefore produces points off the image of ``M``, and :meth:`WarsawMap.to_smefit`
warns about them.
"""

import functools
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from smefit.constants import cw, sw

_logger = logging.getLogger(__name__)

# A coefficient is a number, or a function of the strong coupling g_s.
Coeff = float | Callable[[float], float]

SMEFIT_TO_WARSAW: dict[str, dict[str, Coeff]] = {
    # Bosonic
    "OWWW": {"W": -1.0},
    "OpBox": {"phiBox": 1.0},
    "OpD": {"phiD": 1.0},
    "OpWB": {"phiWB": 1.0},
    "OpG": {"phiG": 1.0},
    "OpW": {"phiW": 1.0},
    "OpB": {"phiB": 1.0},
    "Op": {"phi": 1.0},
    # Dipoles
    "OtG": {"uG_33": lambda gs: -gs},
    "OtW": {"uB_33": -cw / sw, "uW_33": -1.0},
    "OtZ": {"uB_33": 1 / sw},
    # Quark Currents
    "O3pq": {"phiq1_11": 1.0, "phiq1_22": 1.0, "phiq3_11": 1.0, "phiq3_22": 1.0},
    "OpqMi": {"phiq1_11": 1.0, "phiq1_22": 1.0},
    "O3pQ3": {"phiq1_33": 1.0, "phiq3_33": 1.0},
    "OpQM": {"phiq1_33": 1.0},
    "Opt": {"phiu_33": 1.0},
    "Opui": {"phiu_11": 1.0, "phiu_22": 1.0},
    "Opdi": {"phid_11": 1.0, "phid_22": 1.0, "phid_33": 1.0},
    # Lepton Currents
    "Opl1": {"phil1_11": 1.0},
    "Opl2": {"phil1_22": 1.0},
    "Opl3": {"phil1_33": 1.0},
    "O3pl1": {"phil3_11": 1.0},
    "O3pl2": {"phil3_22": 1.0},
    "O3pl3": {"phil3_33": 1.0},
    "Ope": {"phie_11": 1.0},
    "Opmu": {"phie_22": 1.0},
    "Opta": {"phie_33": 1.0},
    # Yukawas
    "Otp": {"uphi_33": 1.0},
    "Ocp": {"uphi_22": 1.0},
    "Obp": {"dphi_33": 1.0},
    "Otap": {"ephi_33": 1.0},
    "Omup": {"ephi_22": 1.0},
    # 2L2H quark operators
    "O81qq": {
        "qq1_1331": 1 / 4,
        "qq1_2332": 1 / 4,
        "qq1_1133": -1 / 6,
        "qq1_2233": -1 / 6,
        "qq3_1331": 1 / 4,
        "qq3_2332": 1 / 4,
    },
    "O11qq": {"qq1_1133": 1.0, "qq1_2233": 1.0},
    "O83qq": {
        "qq1_1331": 3 / 4,
        "qq1_2332": 3 / 4,
        "qq3_1133": -1 / 6,
        "qq3_2233": -1 / 6,
        "qq3_1331": -1 / 4,
        "qq3_2332": -1 / 4,
    },
    "O13qq": {"qq3_1133": 1.0, "qq3_2233": 1.0},
    "O8qt": {"qu8_1133": 1.0, "qu8_2233": 1.0},
    "O1qt": {"qu1_1133": 1.0, "qu1_2233": 1.0},
    "O8ut": {"uu_1133": -1 / 6, "uu_2233": -1 / 6, "uu_1331": 1 / 2, "uu_2332": 1 / 2},
    "O1ut": {"uu_1133": 1.0, "uu_2233": 1.0},
    "O8qu": {"qu8_3311": 1.0, "qu8_3322": 1.0},
    "O1qu": {"qu1_3311": 1.0, "qu1_3322": 1.0},
    "O8dt": {"ud8_3311": 1.0, "ud8_3322": 1.0, "ud8_3333": 1.0},
    "O1dt": {"ud1_3311": 1.0, "ud1_3322": 1.0, "ud1_3333": 1.0},
    "O8qd": {"qd8_3311": 1.0, "qd8_3322": 1.0, "qd8_3333": 1.0},
    "O1qd": {"qd1_3311": 1.0, "qd1_3322": 1.0, "qd1_3333": 1.0},
    # 4H quark operators
    "OQQ1": {"qq1_3333": 1 / 2},
    "OQQ8": {"qq1_3333": 1 / 24, "qq3_3333": 1 / 8},
    "OQt1": {"qu1_3333": 1.0},
    "OQt8": {"qu8_3333": 1.0},
    "Ott1": {"uu_3333": 1.0},
    "Obb": {"dd_3333": 1.0},
    # 4 leptons
    # left-left leptons: notice the factor 2, see table 28 of arXiv:2012.11343
    "Oll1221": {"ll_1221": 2.0},
    "Oll1331": {"ll_1331": 2.0},
    "Oll2332": {"ll_2332": 2.0},
    "Oll1111": {"ll_1111": 1.0},
    "Oll1122": {"ll_1122": 2.0},
    "Oll1133": {"ll_1133": 2.0},
    "Oll2233": {"ll_2233": 2.0},
    "Oll2222": {"ll_2222": 1.0},
    "Oll3333": {"ll_3333": 1.0},
    # left-right leptons
    "Ole1111": {"le_1111": 1.0},
    "Ole2222": {"le_2222": 1.0},
    "Ole3333": {"le_3333": 1.0},
    "Ole1133": {"le_1133": 1.0},
    "Ole1122": {"le_1122": 1.0},
    "Ole2233": {"le_2233": 1.0},
    "Ole3322": {"le_3322": 1.0},
    "Ole3311": {"le_3311": 1.0},
    "Ole2211": {"le_2211": 1.0},
    # right-right leptons: factor 4 to agree with the SMEFTsim general convention
    "Oee1111": {"ee_1111": 1.0},
    "Oee2222": {"ee_2222": 1.0},
    "Oee3333": {"ee_3333": 1.0},
    "Oee1122": {"ee_1122": 4.0},
    "Oee1133": {"ee_1133": 4.0},
    "Oee2233": {"ee_2233": 4.0},
    # 2 quark 2 lepton operators
    "Oeu": {"eu_1111": 1.0, "eu_1122": 1.0},
    "Omuu": {"eu_2211": 1.0, "eu_2222": 1.0},
    "Otau": {"eu_3311": 1.0, "eu_3322": 1.0},
    "Oed": {"ed_1111": 1.0, "ed_1122": 1.0},
    "Omud": {"ed_2211": 1.0, "ed_2222": 1.0},
    "Otad": {"ed_3311": 1.0, "ed_3322": 1.0},
    "Oeb": {"ed_1133": 1.0},
    "Omub": {"ed_2233": 1.0},
    "Otab": {"ed_3333": 1.0},
    "Otl1": {"lu_1133": 1.0},
    "Otl2": {"lu_2233": 1.0},
    "Otl3": {"lu_3333": 1.0},
    "Ote": {"eu_1133": 1.0},
    "Otmu": {"eu_2233": 1.0},
    "Otta": {"eu_3333": 1.0},
    "Oql13": {"lq1_1111": 1.0, "lq1_1122": 1.0, "lq3_1111": 1.0, "lq3_1122": 1.0},
    "Oql23": {"lq1_2211": 1.0, "lq1_2222": 1.0, "lq3_2211": 1.0, "lq3_2222": 1.0},
    "Oql33": {"lq1_3311": 1.0, "lq1_3322": 1.0, "lq3_3311": 1.0, "lq3_3322": 1.0},
    "Oql1M": {"lq1_1111": 1.0, "lq1_1122": 1.0},
    "Oql2M": {"lq1_2211": 1.0, "lq1_2222": 1.0},
    "Oql3M": {"lq1_3311": 1.0, "lq1_3322": 1.0},
    "OQl13": {"lq1_1133": 1.0, "lq3_1133": 1.0},
    "OQl23": {"lq1_2233": 1.0, "lq3_2233": 1.0},
    "OQl33": {"lq1_3333": 1.0, "lq3_3333": 1.0},
    "OQl1M": {"lq1_1133": 1.0},
    "OQl2M": {"lq1_2233": 1.0},
    "OQl3M": {"lq1_3333": 1.0},
    "Ol1u": {"lu_1111": 1.0, "lu_1122": 1.0},
    "Ol2u": {"lu_2211": 1.0, "lu_2222": 1.0},
    "Ol3u": {"lu_3311": 1.0, "lu_3322": 1.0},
    "Ol1d": {"ld_1111": 1.0, "ld_1122": 1.0},
    "Ol2d": {"ld_2211": 1.0, "ld_2222": 1.0},
    "Ol3d": {"ld_3311": 1.0, "ld_3322": 1.0},
    "Ol1b": {"ld_1133": 1.0},
    "Ol2b": {"ld_2233": 1.0},
    "Ol3b": {"ld_3333": 1.0},
    "Oqe": {"qe_1111": 1.0, "qe_2211": 1.0},
    "Oqmu": {"qe_1122": 1.0, "qe_2222": 1.0},
    "Oqta": {"qe_1133": 1.0, "qe_2233": 1.0},
    "OQe": {"qe_3311": 1.0},
    "OQmu": {"qe_3322": 1.0},
    "OQta": {"qe_3333": 1.0},
}


def _blocks(matrix: np.ndarray) -> list[np.ndarray]:
    """Group the columns of *matrix* into independent blocks.

    Two columns are in the same block when they share a non-zero row, directly
    or through other columns. Inverting block by block keeps the inverse exactly
    zero between operators that have no Warsaw coefficient in common.
    """
    nonzero = matrix != 0
    linked = (nonzero.T.astype(int) @ nonzero.astype(int)) > 0
    unassigned = set(range(matrix.shape[1]))
    blocks = []
    while unassigned:
        frontier = [unassigned.pop()]
        block = []
        while frontier:
            col = frontier.pop()
            block.append(col)
            neighbours = unassigned.intersection(np.flatnonzero(linked[col]).tolist())
            unassigned -= neighbours
            frontier.extend(neighbours)
        blocks.append(np.array(sorted(block)))
    return blocks


@functools.cache
def _warn_flavour_breaking(origin: str, targets: tuple[str, ...]) -> None:
    """Log the flavour-breaking warning, once per distinct *origin* and *targets*."""
    _logger.warning(
        f"{origin} breaks the flavour symmetry assumed by {list(targets)}: "
        "only their generation-1 Warsaw components are mapped back."
    )


@dataclass(frozen=True)
class WarsawMap:
    """The SMEFiT → Warsaw matrix at a given g_s, and its derived inverse.

    Attributes
    ----------
    ops : tuple[str, ...]
        SMEFiT operators, the columns of :attr:`matrix`.
    warsaw : tuple[str, ...]
        Warsaw coefficients any SMEFiT operator switches on, the rows of
        :attr:`matrix`.
    matrix : numpy.ndarray
        ``M``, shape ``(len(warsaw), len(ops))``.
    inverse : numpy.ndarray
        Left inverse of ``M``, shape ``(len(ops), len(warsaw))``, reading the
        first independent listed components of each block (see the module
        docstring).
    """

    ops: tuple[str, ...]
    warsaw: tuple[str, ...]
    matrix: np.ndarray
    inverse: np.ndarray

    @classmethod
    def from_table(
        cls, table: Mapping[str, Mapping[str, Coeff]], gs: float
    ) -> "WarsawMap":
        """Build the map from a ``{op: {warsaw: coeff}}`` table.

        Parameters
        ----------
        table : Mapping[str, Mapping[str, Coeff]]
            Forward map, in the format of :data:`SMEFIT_TO_WARSAW`.
        gs : float
            Strong coupling the g_s-dependent coefficients are evaluated at.

        Raises
        ------
        ValueError
            If some SMEFiT operators are linearly dependent in the Warsaw
            basis, so that no inverse exists.
        """
        ops = tuple(table)
        # rows in the order the table first lists them: the inverse reads the
        # first independent ones
        warsaw = tuple(dict.fromkeys(w for entry in table.values() for w in entry))
        row = {w: i for i, w in enumerate(warsaw)}

        matrix = np.zeros((len(warsaw), len(ops)))
        for col, op in enumerate(ops):
            for w, coeff in table[op].items():
                matrix[row[w], col] = coeff(gs) if callable(coeff) else coeff

        inverse = np.zeros((len(ops), len(warsaw)))
        for cols in _blocks(matrix):
            read = []
            for r in np.flatnonzero(matrix[:, cols].any(axis=1)):
                if np.linalg.matrix_rank(matrix[np.ix_([*read, r], cols)]) > len(read):
                    read.append(r)
            if len(read) < len(cols):
                raise ValueError(
                    "SMEFiT operators "
                    f"{[ops[c] for c in cols]} are linearly dependent in the "
                    "Warsaw basis: the map to it cannot be inverted."
                )
            inverse[np.ix_(cols, read)] = np.linalg.inv(matrix[np.ix_(read, cols)])

        return cls(ops, warsaw, matrix, inverse)

    def to_warsaw(self, op: str) -> dict[str, float]:
        """Warsaw coefficients switched on by a unit coefficient of *op*."""
        col = self.matrix[:, self.ops.index(op)]
        return {self.warsaw[i]: float(col[i]) for i in np.flatnonzero(col)}

    def to_smefit(
        self,
        values: Mapping[str, float],
        origin: str = "Warsaw point",
        tol: float = 1e-4,
    ) -> dict[str, float]:
        """Map a point in the Warsaw basis back to the SMEFiT basis.

        Warsaw coefficients no SMEFiT operator switches on are dropped. The rest
        should lie in the image of :attr:`matrix`; the SMEFiT operators whose
        Warsaw components miss it by more than *tol* (relative to the size of
        the point) are named in a warning, logged once per *origin*.

        Parameters
        ----------
        values : Mapping[str, float]
            Warsaw coefficient values, keyed by WCxf name.
        origin : str
            What the point comes from, for the warning.
        tol : float
            Largest flavour-symmetry violation accepted silently, relative to
            the norm of the point.

        Returns
        -------
        dict[str, float]
            The non-zero SMEFiT coefficients.
        """
        point = np.array([values.get(w, 0.0) for w in self.warsaw])
        coeffs = self.inverse @ point

        broken = np.abs(self.matrix @ coeffs - point) > tol * np.linalg.norm(point)
        if broken.any():
            targets = np.flatnonzero(self.matrix[broken].any(axis=0))
            _warn_flavour_breaking(origin, tuple(self.ops[t] for t in targets))

        return {op: float(c) for op, c in zip(self.ops, coeffs) if c != 0.0}


@functools.cache
def warsaw_map(gs: float) -> WarsawMap:
    """The :data:`SMEFIT_TO_WARSAW` map at strong coupling *gs*."""
    return WarsawMap.from_table(SMEFIT_TO_WARSAW, gs)
