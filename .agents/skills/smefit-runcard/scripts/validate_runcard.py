#!/usr/bin/env python
"""Validate a smefit runcard without importing smefit.

Checks YAML structure, coefficient invariants, prior distributions, settings
blocks (against the auto-generated runcard-keys.json), and that every dataset
has matching commondata/theory files under data_path/theory_path.

Usage:
    python validate_runcard.py <runcard.yaml> [--strict]

Exit codes: 0 = valid, 1 = errors found, 2 = cannot read input.
--strict additionally treats unknown top-level/sub-keys as errors (default: warnings).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required (pip install pyyaml).")
    sys.exit(2)

KEYS_JSON = Path(__file__).resolve().parent.parent / "references" / "runcard-keys.json"

# Actions that require specific runcard content (subset — validated best-effort).
SAMPLER_BLOCKS = {
    "run_ultranest_fit": "ultranest_settings",
    "run_blackjax_fit": "blackjax_settings",
    "run_individual_ultranest_fits": "ultranest_settings",
    "run_individual_blackjax_fits": "blackjax_settings",
}


class Report:
    def __init__(self, strict):
        self.errors = []
        self.warnings = []
        self.strict = strict

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        (self.errors if self.strict else self.warnings).append(msg)


def load_keys():
    if not KEYS_JSON.exists():
        return None
    with open(KEYS_JSON) as f:
        return json.load(f)


def find_paths_config():
    """Locate the machine-specific .config/paths.yaml (created by smefit_setup_local).

    Mirrored verbatim in ../../smefit-datasets/scripts/smefit_db.py — each skill
    directory has to stand alone (see the skills README.md), so keep the two
    copies in sync rather than factoring one out.
    """
    # 1. Walk up from cwd (covers running inside the repo or a work dir below it).
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        candidate = parent / ".config" / "paths.yaml"
        if candidate.is_file():
            return candidate
    # 2. Script-relative repo root (.agents/skills/<skill>/scripts/ -> repo).
    candidate = Path(__file__).resolve().parents[4] / ".config" / "paths.yaml"
    if candidate.is_file():
        return candidate
    # 3. Ask an installed smefit where its config lives (smefit.paths only
    #    imports pathlib+yaml, so this subprocess is cheap).
    try:
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "from smefit.paths import USER_PATHS_CONFIG; print(USER_PATHS_CONFIG)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0:
            candidate = Path(out.stdout.strip())
            if candidate.is_file():
                return candidate
    except (OSError, subprocess.SubprocessError):
        pass
    return None


class PathResolver:
    """Mirror of smefit.paths.resolve_path: expand prefix-relative runcard paths
    (e.g. smefit_database/commondata) using .config/paths.yaml, longest key first.
    Standard prefixes that are not configured are reported as errors."""

    def __init__(self, standard_prefixes):
        self.standard_prefixes = standard_prefixes
        self.config_file = find_paths_config()
        # Set when a path could have been a prefix but no config was available to
        # expand it: worth a hint, unlike a runcard that only uses real paths.
        self.saw_unresolvable_prefix = False
        self.user_paths = {}
        if self.config_file is not None:
            try:
                loaded = yaml.safe_load(self.config_file.read_text())
                if isinstance(loaded, dict):
                    self.user_paths = loaded
            except (yaml.YAMLError, OSError):
                pass

    def has_prefix(self, path_str, prefix):
        return path_str == prefix or path_str.startswith(prefix + "/")

    @staticmethod
    def looks_prefix_relative(path_str):
        """True for a bare 'name/sub/...' path, i.e. one that paths.yaml could
        have expanded. Absolute, '~'- and '.'-rooted paths never are."""
        head = path_str.split("/", 1)[0]
        return "/" in path_str and head != "" and not head.startswith(("~", "."))

    def resolve(self, path_str, rep, label):
        """Return the resolved path string, or None if a prefix cannot be resolved."""
        path_str = str(path_str)
        for key in sorted(self.user_paths, key=len, reverse=True):
            if self.has_prefix(path_str, key):
                return str(self.user_paths[key]).rstrip("/") + path_str[len(key) :]
        for prefix in self.standard_prefixes:
            if self.has_prefix(path_str, prefix):
                where = self.config_file or ".config/paths.yaml (none found)"
                rep.error(
                    f"{label}: '{path_str}' uses the '{prefix}' prefix but it is "
                    f"not configured in {where} — run 'smefit_setup_local'"
                )
                return None
        if not self.user_paths and self.looks_prefix_relative(path_str):
            self.saw_unresolvable_prefix = True
        return path_str


def check_coefficient(
    name, spec, prior_dists, rep, coeff_keys=None, require_prior=True
):
    if not isinstance(spec, dict):
        rep.error(f"coefficients.{name}: must be a mapping, got {type(spec).__name__}")
        return
    if coeff_keys:
        for k in set(spec) - set(coeff_keys):
            rep.warn(
                f"coefficients.{name}: unknown sub-key '{k}' — it is ignored "
                f"(known: {sorted(coeff_keys)})"
            )
    free = spec.get("free", True)
    if free:
        for forbidden in ("value", "expr", "vars"):
            if forbidden in spec:
                rep.error(f"coefficients.{name}: free=True forbids '{forbidden}'")
        prior = spec.get("prior")
        if prior is None:
            if require_prior:
                rep.error(
                    f"coefficients.{name}: free coefficient needs a prior — required "
                    "by a sampler action in 'actions_' (unless the runcard uses "
                    "whitening or bayesian_update_path)"
                )
        elif isinstance(prior, dict):
            dist = prior.get("dist")
            if dist is None:
                rep.error(f"coefficients.{name}: prior needs a 'dist' key")
            elif prior_dists and str(dist).lower() not in prior_dists:
                rep.error(
                    f"coefficients.{name}: unknown prior dist '{dist}' "
                    f"(allowed: {sorted(prior_dists)})"
                )
            elif prior_dists:
                required = set(prior_dists[str(dist).lower()])
                given = set(prior) - {"dist"}
                if given != required:
                    rep.error(
                        f"coefficients.{name}: prior '{dist}' needs exactly "
                        f"{sorted(required)}, got {sorted(given)}"
                    )
        else:
            rep.error(f"coefficients.{name}: prior must be a mapping")
        return

    # fixed / constrained
    if "prior" in spec:
        rep.error(f"coefficients.{name}: free=False forbids 'prior'")
    has_value = "value" in spec
    has_expr = "expr" in spec
    if has_value == has_expr:
        rep.error(
            f"coefficients.{name}: free=False requires exactly one of 'value' or 'expr'"
        )
    if has_value and "vars" in spec:
        rep.error(f"coefficients.{name}: constant coefficient forbids 'vars'")
    if has_expr:
        variables = spec.get("vars")
        if not variables:
            rep.error(
                f"coefficients.{name}: expr coefficient requires non-empty 'vars'"
            )
        elif len(set(variables)) != len(variables):
            rep.error(f"coefficients.{name}: 'vars' contains duplicates")


def check_vars_references(coefficients, rep):
    """Every name in a `vars:` list must be another coefficient in this runcard.

    CoefficientGroup raises "var '<v>' in 'vars' is not defined" at run time;
    catching it here saves a full startup.
    """
    defined = set(coefficients)
    for name, spec in coefficients.items():
        if not isinstance(spec, dict) or spec.get("free", True):
            continue
        for var in spec.get("vars") or []:
            if var not in defined:
                rep.error(
                    f"coefficients.{name}: 'vars' references '{var}', which is not "
                    f"defined in this runcard (defined: {sorted(defined)})"
                )


# Nonlinearity markers in an `expr:` — any of these makes the model nonlinear
# even at use_quad: False, which invalidates run_analytic_fit.
_NONLINEAR_FUNCS = ("sqrt", "exp", "log", "sin", "cos", "tan", "abs")


def expr_is_nonlinear(expr, variables):
    """Best-effort: True when `expr` is certainly nonlinear in `variables`."""
    expr = str(expr)
    if "**" in expr:
        return True
    if any(re.search(rf"\b{fn}\s*\(", expr) for fn in _NONLINEAR_FUNCS):
        return True
    # a product of two coefficient names, e.g. "y*OpWB" (but not "100*y")
    for lhs, rhs in re.findall(r"([A-Za-z_]\w*)\s*\*(?!\*)\s*([A-Za-z_]\w*)", expr):
        if lhs in variables and rhs in variables:
            return True
    return False


def check_coefficients_against_theory(coefficients, operators, rep):
    """Warn about coefficients no selected theory file knows about.

    `operators` is the union of operator keys across the chosen datasets/orders.
    A coefficient absent from all of them is either a typo or simply not probed
    by this data — in both cases its posterior comes back equal to its prior.
    """
    if not operators:
        return
    for name, spec in coefficients.items():
        if not isinstance(spec, dict) or not spec.get("free", True):
            continue  # fixed/constrained entries include auxiliary parameters
        if name not in operators:
            rep.warn(
                f"coefficients.{name}: no selected dataset's theory file contains "
                "this operator — check the spelling (smefit_db.py operators), or "
                "expect an unconstrained, prior-shaped posterior"
            )


def check_datasets(runcard, keys, rep, resolver):
    operators = set()  # union of operator keys across the selected datasets/orders
    datasets = runcard.get("datasets")
    if datasets is None:
        return operators
    if not isinstance(datasets, list):
        rep.error("datasets: must be a list of {name, order, ...} entries")
        return operators

    data_path = runcard.get("data_path")
    theory_path = runcard.get("theory_path")
    if data_path:
        data_path = resolver.resolve(data_path, rep, "data_path")
    if theory_path:
        theory_path = resolver.resolve(theory_path, rep, "theory_path")
    data_dir = Path(data_path) if data_path else None
    theory_dir = Path(theory_path) if theory_path else None
    if data_dir and not data_dir.is_dir():
        rep.error(f"data_path does not exist: {data_dir}")
        data_dir = None
    if theory_dir and not theory_dir.is_dir():
        rep.error(f"theory_path does not exist: {theory_dir}")
        theory_dir = None

    entry_keys = set(keys["dataset_entry_keys"]) if keys else set()

    for i, entry in enumerate(datasets):
        label = f"datasets[{i}]"
        if not isinstance(entry, dict) or "name" not in entry:
            rep.error(f"{label}: each entry needs at least a 'name' key")
            continue
        name = entry["name"]
        label = f"datasets[{i}] ({name})"
        if entry_keys:
            for k in set(entry) - entry_keys:
                rep.warn(f"{label}: unknown key '{k}' (known: {sorted(entry_keys)})")
        if "order" not in entry:
            rep.error(f"{label}: missing 'order' (e.g. LO or NLO_QCD)")
        if data_dir and not (data_dir / f"{name}.yaml").exists():
            rep.error(f"{label}: no commondata file {data_dir / (name + '.yaml')}")
        if theory_dir:
            theory_file = theory_dir / f"{name}.json"
            if not theory_file.exists():
                rep.error(f"{label}: no theory file {theory_file}")
                continue
            # Allowed orders and covariance types are per-dataset: they are
            # literally the keys of the theory JSON (what load_theory reads).
            try:
                theory = json.loads(theory_file.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                rep.error(f"{label}: cannot read theory file {theory_file}: {exc}")
                continue
            orders = sorted(
                k
                for k in theory
                if k not in {"best_sm", "scales"} and not k.startswith("theory_cov")
            )
            if "order" in entry and entry["order"] not in orders:
                rep.error(
                    f"{label}: order '{entry['order']}' not in theory file "
                    f"(available: {orders})"
                )
            elif "order" in entry:
                # Keys of the order dict are "SM", "<Op>" and "<OpA>*<OpB>".
                block = theory[entry["order"]]
                if isinstance(block, dict):
                    operators.update(k for k in block if k != "SM" and "*" not in k)
            cov = entry.get("theory_cov", "current")
            if f"theory_cov_{cov}" not in theory:
                available = sorted(
                    k[len("theory_cov_") :]
                    for k in theory
                    if k.startswith("theory_cov_")
                )
                rep.error(
                    f"{label}: theory_cov '{cov}' not in theory file "
                    f"(available: {available})"
                )

    return operators


def check_settings_blocks(runcard, keys, rep):
    if not keys:
        return
    for block, info in keys["settings_blocks"].items():
        if block not in runcard or not info.get("known_keys"):
            continue
        value = runcard[block]
        if not isinstance(value, dict):
            continue  # path-like parse keys (data_path etc.) are strings
        for k in set(value) - set(info["known_keys"]):
            rep.warn(f"{block}: unknown sub-key '{k}' (known: {info['known_keys']})")


BLACKJAX_ALGORITHMS = ("nested_sampling", "nuts")


def check_blackjax_algorithm(runcard, rep):
    """Validate blackjax_settings.algorithm and its incompatibilities.

    check_settings_blocks only checks that sub-keys are known, not that their
    values are legal, so the enum needs its own check here.
    """
    settings = runcard.get("blackjax_settings")
    if not isinstance(settings, dict):
        return
    algorithm = settings.get("algorithm", "nested_sampling")
    if algorithm not in BLACKJAX_ALGORITHMS:
        rep.error(
            f"blackjax_settings.algorithm: '{algorithm}' is not a known algorithm "
            f"(allowed: {list(BLACKJAX_ALGORITHMS)})"
        )
        return
    if algorithm == "nuts" and "bayesian_update_path" in runcard:
        rep.error(
            "blackjax_settings.algorithm: 'nuts' is incompatible with "
            "'bayesian_update_path' — the exact-posterior prior has no "
            "per-parameter bijectors. Use algorithm: nested_sampling."
        )
    init = settings.get("init", "prior")
    if init not in ("prior", "baseline"):
        rep.error(
            f"blackjax_settings.init: '{init}' is not valid "
            "(allowed: ['prior', 'baseline'])"
        )


def check_rg_matrix(value, label, resolver, rep):
    """Check an rg_matrix path. Missing files under smefit_results/ are only a
    warning: smefit auto-downloads the fit from the server before failing."""
    resolved = resolver.resolve(value, rep, label)
    if resolved is None or Path(resolved).exists():
        return
    if resolver.has_prefix(str(value), "smefit_results"):
        rep.warn(
            f"{label}: {resolved} not found locally — smefit will try to "
            "download the fit from the server at run time"
        )
    else:
        rep.error(f"{label}: file not found: {resolved}")


def check_actions(runcard, rep, nonlinear_exprs=()):
    actions = runcard.get("actions_")
    if not actions:
        rep.error("runcard needs a non-empty 'actions_' list")
        return
    if not isinstance(actions, list):
        rep.error("actions_: must be a list")
        return
    for action in actions:
        block = SAMPLER_BLOCKS.get(action)
        if block and block not in runcard:
            rep.warn(
                f"actions_: '{action}' usually pairs with a '{block}' block "
                "(defaults will be used)"
            )
        if (
            action == "report"
            and "template_text" not in runcard
            and "template" not in runcard
        ):
            rep.error(
                "actions_: 'report' requires a 'template_text' (or 'template') key"
            )
        if action in ("run_analytic_fit", "run_individual_analytic_fits"):
            if runcard.get("use_quad", False):
                rep.error(
                    f"actions_: '{action}' requires a linear model — set use_quad: False"
                )
            if nonlinear_exprs:
                rep.error(
                    f"actions_: '{action}' requires a linear model, but "
                    f"{', '.join(sorted(nonlinear_exprs))} "
                    f"{'has' if len(nonlinear_exprs) == 1 else 'have'} a nonlinear "
                    "'expr' — the posterior is not Gaussian; use a sampler or the "
                    "hessian fit instead"
                )


def check_top_level(runcard, keys, rep):
    if not keys:
        return
    known = set(keys["top_level_keys"]) | {"template"}
    for k in set(runcard) - known:
        rep.warn(f"unknown top-level key '{k}' — smefit will ignore it silently")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runcard", type=Path)
    parser.add_argument(
        "--strict", action="store_true", help="treat unknown keys as errors"
    )
    args = parser.parse_args()

    try:
        runcard = yaml.safe_load(args.runcard.read_text())
    except OSError as exc:
        print(f"ERROR: cannot read {args.runcard}: {exc}")
        return 2
    except yaml.YAMLError as exc:
        print(f"ERROR: invalid YAML in {args.runcard}:\n{exc}")
        return 2
    if not isinstance(runcard, dict):
        print(f"ERROR: {args.runcard} does not contain a YAML mapping")
        return 2

    keys = load_keys()
    rep = Report(strict=args.strict)
    if keys is None:
        rep.warnings.append(
            f"runcard-keys.json not found at {KEYS_JSON} — key checks skipped "
            "(regenerate with scripts/generate_skill_reference.py)"
        )
    standard_prefixes = (keys.get("path_prefixes") if keys else None) or [
        "new_smefit",
        "smefit_database",
        "smefit_results",
    ]
    resolver = PathResolver(standard_prefixes)

    if "datasets" not in runcard and "external_chi2" not in runcard:
        rep.error("runcard needs 'datasets' and/or 'external_chi2' — no data to fit")
    if "datasets" in runcard:
        for key in ("data_path", "theory_path"):
            if key not in runcard:
                rep.error(f"'{key}' is required when 'datasets' is present")

    actions = runcard.get("actions_")
    coefficients = runcard.get("coefficients")
    nonlinear_exprs = []
    if not coefficients:
        rep.error("runcard needs a non-empty 'coefficients' mapping")
    elif isinstance(coefficients, dict):
        prior_dists = keys["prior_dists"] if keys else {}
        coeff_keys = keys.get("coefficient_keys") if keys else None
        # A per-coefficient prior is only actually consumed by reportengine
        # when a sampler action (run_*ultranest*/*blackjax*) needs the
        # `prior`/`individual_prior` node — run_analytic_fit, run_hessian_fit,
        # gradient-descent fits, and their individual variants never resolve
        # it. Even then, `whitening`/`bayesian_update_path` make reportengine
        # synthesize a prior itself, so per-coefficient priors are unneeded.
        needs_sampler_prior = isinstance(actions, list) and any(
            a in SAMPLER_BLOCKS for a in actions
        )
        require_prior = needs_sampler_prior and not (
            "whitening" in runcard or "bayesian_update_path" in runcard
        )
        for name, spec in coefficients.items():
            check_coefficient(
                name, spec, prior_dists, rep, coeff_keys, require_prior=require_prior
            )
            if (
                isinstance(spec, dict)
                and spec.get("expr")
                and expr_is_nonlinear(spec["expr"], spec.get("vars") or [])
            ):
                nonlinear_exprs.append(name)
        check_vars_references(coefficients, rep)
    else:
        rep.error("coefficients: must be a mapping of Name: {…}")
        coefficients = None

    rge = runcard.get("rge")
    if rge is not None:
        if not isinstance(rge, dict) or "init_scale" not in rge:
            rep.error("rge block requires 'init_scale' (GeV)")
        else:
            obs_scale = rge.get("obs_scale", "dynamic")
            if not isinstance(obs_scale, (int, float)) and obs_scale != "dynamic":
                rep.error("rge.obs_scale must be a number or 'dynamic'")
            if "rg_matrix" in rge:
                check_rg_matrix(rge["rg_matrix"], "rge.rg_matrix", resolver, rep)

    external = runcard.get("external_chi2")
    if external is not None:
        if not isinstance(external, dict):
            rep.error("external_chi2: must be a mapping ClassName: {path: ..., ...}")
        else:
            for cname, cfg in external.items():
                if not isinstance(cfg, dict) or "path" not in cfg:
                    rep.error(f"external_chi2.{cname}: needs a 'path' key")
                    continue
                label = f"external_chi2.{cname}"
                resolved = resolver.resolve(cfg["path"], rep, f"{label}.path")
                if resolved is not None and not Path(resolved).exists():
                    rep.error(f"{label}: path not found: {resolved}")
                if cfg.get("rg_matrix"):
                    check_rg_matrix(
                        cfg["rg_matrix"], f"{label}.rg_matrix", resolver, rep
                    )

    operators = check_datasets(runcard, keys, rep, resolver)
    # An `rge` block evolves coefficients from a different scale, and external
    # chi2 modules bring their own parameters: in both cases a coefficient can
    # legitimately be absent from every theory file, so skip the cross-check.
    if isinstance(coefficients, dict) and not ("rge" in runcard or external):
        check_coefficients_against_theory(coefficients, operators, rep)
    check_settings_blocks(runcard, keys, rep)
    check_blackjax_algorithm(runcard, rep)
    check_actions(runcard, rep, nonlinear_exprs)
    check_top_level(runcard, keys, rep)
    # Only relevant once a path in this runcard actually needed the config: a
    # runcard using absolute paths does not care that paths.yaml is missing.
    if resolver.saw_unresolvable_prefix:
        rep.warnings.append(
            "no .config/paths.yaml found — prefix-relative paths (e.g. "
            "smefit_database/commondata) cannot be resolved; run 'smefit_setup_local'"
        )

    for msg in rep.warnings:
        print(f"WARNING: {msg}")
    for msg in rep.errors:
        print(f"ERROR: {msg}")
    if rep.errors:
        print(
            f"\n{args.runcard}: INVALID ({len(rep.errors)} error(s), "
            f"{len(rep.warnings)} warning(s))"
        )
        return 1
    print(f"{args.runcard}: OK ({len(rep.warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
