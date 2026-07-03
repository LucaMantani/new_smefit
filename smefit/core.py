"""
smefit.core.py

Core module of smefit, containing the main data classes for the framework.
"""

import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, List, Mapping, Optional

import jax.numpy as jnp
import jax.scipy.linalg as la
import numpy as np
import pandas as pd

from smefit.data_utils import covmat_from_systematics

# Namespace available in coefficient expressions.
# All functions map to JAX equivalents so they are fully differentiable.
_EXPR_NAMESPACE: dict = {
    "__builtins__": {},
    "abs": jnp.abs,
    "sqrt": jnp.sqrt,
    "exp": jnp.exp,
    "log": jnp.log,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "pi": float(jnp.pi),
}


@dataclass
class Dataset:
    """Class representing a dataset in smefit.

    Attributes
    ----------
    name : str
        Name of the dataset.
    num_data : int
        Number of data points in the dataset.
    central_values : jnp.ndarray 1D array
        Central values of the data points.
    stat_err : jnp.ndarray 1D array
        Statistical uncertainties of the data points.
    syst_err : jnp.ndarray 2D array
        Systematic uncertainties of the data points, shape (n_sys, num_data).
    sys_names : List[str]
        Names of the systematic uncertainties.
    sys_types : List[str]
        Types of the systematic uncertainties.
    luminosity : jnp.ndarray 1D array
        Luminosity values associated with the data points.
    """

    name: str
    num_data: int
    central_values: jnp.ndarray
    stat_err: jnp.ndarray
    syst_err: jnp.ndarray
    sys_names: List[str]
    sys_types: List[str]
    luminosity: jnp.ndarray

    # compute syst_err as percentage of central values
    @cached_property
    def syst_err_mult(self) -> jnp.ndarray:
        """Systematic uncertainties as percentage of central values."""
        return self.syst_err / self.central_values[None, :]


@dataclass
class Theory:
    """Class representing theory predictions for a dataset in smefit.

    Attributes
    ----------
    name : str
        Name of the dataset.
    order : str
        Perturbative order of the EFT theory prediction (e.g., 'LO', 'NLO', 'NNLO').
    sm_pred: jnp.ndarray
        Standard Model theory predictions for the dataset.
    eft_pred: dict
        EFT contributions to the theory predictions for the dataset.
    sm_covmat: jnp.ndarray
        Theory covariance matrix of the SM for the dataset.
    scales: jnp.ndarray
        Energy scales associated with the EFT predictions.
        This corresponds to the scale at which the Wilson coefficients are defined,
        i.e. the scale at which one has to run the rge to use the predictions.
    operators: list
        List of EFT operators included in the predictions.
    """

    name: str
    order: str
    sm_pred: jnp.ndarray
    eft_pred: jnp.ndarray
    sm_covmat: jnp.ndarray
    scales: jnp.ndarray
    operators: List[str]

    def __post_init__(self):
        # ensure operators are sorted
        self.operators.sort()
        # build linear eft prediction matrix of shape (ndata, n_ops),
        # corresponding to self.operators order. An operator might be missing
        # if linear contribution is zero, but be present in the list if
        # it has a non-zero quadratic contribution.
        self.eft_lin_pred = jnp.vstack(
            [
                (
                    jnp.asarray(self.eft_pred[op])
                    if op in self.eft_pred
                    else jnp.zeros(self.sm_pred.shape[0])
                )
                for op in self.operators
            ]
        ).T
        # build quadratic eft prediction tensors of shape (ndata, n_ops, n_ops)
        self.n_ops = len(self.operators)
        self.n_data = self.sm_pred.shape[0]

        # Build reverse lookup: map operator pairs to their eft_pred values
        # normalise key ordering so each pair is looked up only once
        quad_lookup = {}
        for key, val in self.eft_pred.items():
            if "*" not in key:
                continue
            a, b = key.split("*")
            normalised = (a, b) if a <= b else (b, a)
            quad_lookup[normalised] = val

        eft_quad_pred = np.zeros((self.n_data, self.n_ops, self.n_ops))
        for i, op1 in enumerate(self.operators):
            for j in range(i, self.n_ops):
                pair = (op1, self.operators[j])
                if pair in quad_lookup:
                    eft_quad_pred[:, i, j] = quad_lookup[pair]

        self.eft_quad_pred = jnp.asarray(eft_quad_pred)
        # Define operator index mapping
        self.op_index = {op: i for i, op in enumerate(self.operators)}


@dataclass
class Coefficient:
    """Represent an EFT coefficient.

    Invariants
    ----------
    - If free is True: prior must be provided; value and expr must be None.
    - If free is False: exactly one of (value, expr) must be provided; prior must be None.
    """

    name: str
    free: bool = True
    prior: Optional[Mapping[str, Any]] = None
    value: Optional[float] = None
    expr: Optional[str] = None
    vars: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.free:
            if self.value is not None or self.expr is not None:
                raise ValueError(f"{self.name}: free=True forbids 'value' and 'expr'.")
            if self.vars is not None:
                raise ValueError(f"{self.name}: free=True forbids 'vars'.")
            return

        if self.prior is not None:
            raise ValueError(f"{self.name}: free=False forbids 'prior'.")

        has_value = self.value is not None
        has_expr = self.expr is not None
        if has_value == has_expr:
            raise ValueError(
                f"{self.name}: free=False requires exactly one of 'value' or 'expr'."
            )

        if has_value and self.vars is not None:
            raise ValueError(f"{self.name}: constant coefficient forbids 'vars'.")

        if has_expr:
            if not self.vars:
                raise ValueError(
                    f"{self.name}: expr coefficient requires non-empty 'vars'."
                )
            if len(set(self.vars)) != len(self.vars):
                raise ValueError(
                    f"{self.name}: 'vars' contains duplicates: {self.vars!r}"
                )

    @cached_property
    def _expr_fn(self) -> Optional[Callable[..., float]]:
        """Cached compiled lambda for expr coefficients."""
        if self.free or self.expr is None:
            return None
        assert self.vars is not None
        code = f"lambda {', '.join(self.vars)}: ({self.expr})"
        return eval(code, _EXPR_NAMESPACE)

    def constrain(self, *args: float) -> float:
        """
        Evaluate this coefficient deterministically.

        - If value is set, returns the constant value (args ignored).
        - If expr is set, evaluates expr as a lambda with parameters self.vars.
        """
        if self.free:
            raise TypeError(f"{self.name}: free coefficient cannot be constrained.")

        if self.value is not None:
            return self.value

        fn = self._expr_fn

        return fn(*args)


class DataGroup:
    """Class representing a group of datasets in smefit."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        # order datasets by name for consistency
        self.datasets.sort(key=lambda ds: ds.name)
        # concatenate central values
        self.cv = jnp.concatenate([ds.central_values for ds in self.datasets], axis=0)
        # total number of data points
        self.num_data = sum(ds.num_data for ds in datasets)
        # concatenate luminosities
        self.lumi = jnp.concatenate([ds.luminosity for ds in self.datasets], axis=0)
        # list of dataset names
        self.names = [ds.name for ds in datasets]
        # list of number of data points per dataset
        self.ndata_list = [ds.num_data for ds in datasets]
        # build full exp covariance matrix
        self.exp_covmat = self._build_exp_covmat()

    def _build_exp_covmat(self) -> jnp.ndarray:
        """Build experimental covariance matrix from all datasets in the group.

        This combines statistical and systematic uncertainties from all datasets,
        accounting for correlations both within and across datasets.

        Returns
        -------
        covmat : jnp.ndarray
            Full covariance matrix of shape (num_data, num_data)
        """
        stat_errors = [ds.stat_err for ds in self.datasets]

        # Convert syst_err jnp arrays to pd.DataFrames with sys_names as columns
        # ds.syst_err has shape (n_sys, n_data), we need to transpose it
        sys_errors = [
            pd.DataFrame(ds.syst_err.T, columns=ds.sys_names) for ds in self.datasets
        ]
        return jnp.array(covmat_from_systematics(stat_errors, sys_errors))

    def t0_covmat(self, theory_predictions) -> jnp.ndarray:
        """Build t0 covariance matrix using theory predictions.
        This method constructs the t0 covariance matrix by replacing multiplicative
        systematic uncertainties with values derived from the provided theory predictions.

        Parameters
        ----------
        theory_predictions : jnp.ndarray
            Array of theory predictions corresponding to the concatenated data points.
        Returns
        -------
        covmat : jnp.ndarray
            Full t0 covariance matrix of shape (num_data, num_data)
        """
        stat_errors = [ds.stat_err for ds in self.datasets]
        sys_errors = []

        offset = 0
        for ds in self.datasets:
            n = ds.num_data
            # we take the theory predictions for this dataset
            t_ds = theory_predictions[offset : offset + n]
            offset += n

            sys_types = [t.upper() for t in ds.sys_types]
            is_add = jnp.array([t == "ADD" for t in sys_types])  # shape (n_sys,)

            mult_abs = ds.syst_err_mult * t_ds[None, :]  # shape (n_sys, n)

            syst_errs = jnp.where(is_add[:, None], ds.syst_err, mult_abs)

            sys_errors.append(
                pd.DataFrame(jnp.asarray(syst_errs).T, columns=ds.sys_names)
            )

        return jnp.array(covmat_from_systematics(stat_errors, sys_errors))


class TheoryGroup:
    """Class representing a group of theory predictions in smefit."""

    def __init__(self, theories: List[Theory]):
        self.theories = theories
        # order theories by name for consistency
        self.theories.sort(key=lambda th: th.name)
        # list of dataset names
        self.names = [th.name for th in self.theories]
        # concatenate sm predictions
        self.sm_pred = jnp.concatenate([th.sm_pred for th in self.theories], axis=0)
        # Construct block diagonal theory covariance matrix
        self.sm_covmat = la.block_diag(*[th.sm_covmat for th in self.theories])
        # get list of all unique operators across theories
        all_operators = set()
        for th in self.theories:
            all_operators.update(th.operators)
        self.operators = sorted(list(all_operators))

        self.n_ops = len(self.operators)
        self.n_data = self.sm_pred.shape[0]

        self.eft_lin_pred = self._build_eft_lin_pred()
        self.eft_quad_pred = self._build_eft_quad_pred()
        self.scales = jnp.concatenate([th.scales for th in self.theories], axis=0)

    def _operator_mapping(self, th):
        """Return (global_indices, local_indices) mapping theory operators to group operators."""
        global_idx = [g for g, op in enumerate(self.operators) if op in th.op_index]
        local_idx = [th.op_index[op] for op in self.operators if op in th.op_index]
        return global_idx, local_idx

    def _build_eft_lin_pred(self):
        # build concatenated linear eft prediction matrix of shape (ndata, n_ops)
        # if an operator is not present in a theory, its contribution is zero
        eft_lin_pred = np.zeros((self.n_data, self.n_ops))
        offset = 0
        for th in self.theories:
            n = th.n_data
            global_idx, local_idx = self._operator_mapping(th)
            eft_lin_pred[offset : offset + n][:, global_idx] = np.asarray(
                th.eft_lin_pred[:, local_idx]
            )
            offset += n
        return jnp.asarray(eft_lin_pred)

    def _build_eft_quad_pred(self):
        # build concatenated quadratic eft prediction tensor of shape (ndata, n_ops, n_ops)
        # if an operator is not present in a theory, its contribution is zero
        # upper-triangular convention is preserved because both global and local
        # operator lists are sorted, maintaining relative ordering
        eft_quad_pred = np.zeros((self.n_data, self.n_ops, self.n_ops))
        offset = 0
        for th in self.theories:
            n = th.n_data
            global_idx, local_idx = self._operator_mapping(th)
            rows = np.arange(n)
            eft_quad_pred[offset : offset + n][np.ix_(rows, global_idx, global_idx)] = (
                np.asarray(th.eft_quad_pred)[np.ix_(rows, local_idx, local_idx)]
            )
            offset += n
        return jnp.asarray(eft_quad_pred)


class CoefficientGroup:
    """Class representing a group of EFT coefficients in smefit."""

    def __init__(self, coefficients: List[Coefficient]):
        self.coefficients = coefficients
        # order coefficients by name for consistency
        self.coefficients.sort(key=lambda c: c.name)
        # build coefficient index mapping
        self.coeff_index = {c.name: i for i, c in enumerate(self.coefficients)}
        # names of the coefficients
        self.names = [c.name for c in self.coefficients]
        # precompute free/fixed coefficient lists
        self.free_coeffs = [c for c in self.coefficients if c.free]
        self.free_names = [c.name for c in self.free_coeffs]
        self.fixed_coeffs = [c for c in self.coefficients if not c.free]

        # Whitening matrix (set via whitened())
        self._W = None

        # Precompute resolution mapping
        self._free_indices = [i for i, c in enumerate(self.coefficients) if c.free]
        self._fixed_base = []  # (index, value) for fixed coefficients
        self._expr_specs = (
            []
        )  # (index, constrain_fn, [dep_indices]) for expression coefficients

        # Process coefficients in order, so expression constraints can depend
        # on previously defined coefficients
        for i, c in enumerate(self.coefficients):
            # If coefficient is fixed with a constant value, add to fixed_base
            if not c.free and c.value is not None:
                self._fixed_base.append((i, c.value))
            # If coefficient is fixed with an expression, add to expr_specs
            elif not c.free and c.vars:
                all_names = sorted(self.coeff_index.keys())
                dep_indices = []
                for v in c.vars:
                    if v not in self.coeff_index:
                        raise ValueError(
                            f"Coefficient '{c.name}': var '{v}' in 'vars' is not defined. "
                            f"Available coefficients: {all_names}."
                        )
                    dep = self.coefficients[self.coeff_index[v]]
                    if not dep.free and dep.value is None:
                        raise ValueError(
                            f"Coefficient '{c.name}': var '{v}' in 'vars' must be a free "
                            "coefficient or a fixed coefficient with a constant value."
                        )
                    dep_indices.append(self.coeff_index[v])
                self._expr_specs.append((i, c.constrain, dep_indices))

    def prior_specs(self) -> Dict[str, object]:
        return {c.name: c.prior for c in self.free_coeffs}

    def whitened(self, W: "jnp.ndarray") -> "CoefficientGroup":
        """Return a CoefficientGroup whose resolve un-whitens free coefficients first.

        Parameters
        ----------
        W : jnp.ndarray
            Unwhitening matrix of shape (n_free, n_free). Applied as c = W @ c_w
            before the standard resolve logic.
        """

        new = copy.copy(self)
        new._W = W
        return new

    def resolve(self, free_coeffs: "jnp.ndarray") -> "jnp.ndarray":
        """Map free coefficient values to the full set of coefficient values.

        Parameters
        ----------
        free_coeffs : jnp.ndarray
            Values of the free coefficients, ordered by self.free_coeffs.
            If this CoefficientGroup was created via ``whitened(W)``, the input
            is treated as whitened coordinates and un-whitened first (c = W @ c_w).

        Returns
        -------
        jnp.ndarray
            Values for all coefficients, in self.coefficients order.
        """
        if self._W is not None:
            free_coeffs = self._W @ free_coeffs
        result = jnp.zeros(len(self.coefficients))
        # Place free coefficients
        result = result.at[jnp.array(self._free_indices, dtype=int)].set(free_coeffs)
        # Place fixed values
        for idx, val in self._fixed_base:
            result = result.at[idx].set(val)
        # Evaluate expression constraints
        for idx, fn, deps in self._expr_specs:
            args = tuple(result[d] for d in deps)
            result = result.at[idx].set(fn(*args))
        return result

    def single_free(self, target_name: str) -> "CoefficientGroup":
        """Return a CoefficientGroup where only *target_name* is free.

        Non-target free coefficients are dropped (their zero contribution to
        predictions is omitted), unless they appear as ``vars`` in an
        expression-constrained coefficient that itself depends on the target,
        in which case they are set to ``value=0.0`` so the expression can be
        evaluated. For example, if we want to perform individual fits of c1 and c2
        and we have a constrained c3 = c1 + c2, then we set c2 = 0 when we fit c1
        alone so that c3 = c1.

        Expression-constrained coefficients are kept only if they directly
        depend on the target; those that don't evaluate to constants w.r.t.
        the target are dropped.

        Fixed-constant coefficients (``value`` only) are always kept — they
        have real, non-zero contributions to predictions.
        """

        free_set = set(self.free_names)
        kept_expr_names = {
            c.name for c in self.coefficients if c.vars and target_name in c.vars
        }
        needed_zero_vars = {
            var
            for c in self.coefficients
            if c.name in kept_expr_names
            for var in c.vars
            if var in free_set and var != target_name
        }

        new_coeffs = []
        for c in self.coefficients:
            if c.name == target_name:
                new_coeffs.append(c)
            elif c.free:
                if c.name in needed_zero_vars:
                    new_coeffs.append(Coefficient(name=c.name, free=False, value=0.0))
            elif c.value is not None or c.name in kept_expr_names:
                new_coeffs.append(c)
        return CoefficientGroup(new_coeffs)
