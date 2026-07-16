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


def check_coefficient(name, spec, prior_dists, rep):
    if not isinstance(spec, dict):
        rep.error(f"coefficients.{name}: must be a mapping, got {type(spec).__name__}")
        return
    free = spec.get("free", True)
    if free:
        for forbidden in ("value", "expr", "vars"):
            if forbidden in spec:
                rep.error(f"coefficients.{name}: free=True forbids '{forbidden}'")
        prior = spec.get("prior")
        if prior is None:
            rep.error(
                f"coefficients.{name}: free coefficient needs a prior "
                "(unless the runcard uses whitening or bayesian_update_path)"
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


def check_datasets(runcard, keys, rep):
    datasets = runcard.get("datasets")
    if datasets is None:
        return
    if not isinstance(datasets, list):
        rep.error("datasets: must be a list of {name, order, ...} entries")
        return

    data_path = runcard.get("data_path")
    theory_path = runcard.get("theory_path")
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


def check_actions(runcard, rep):
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
        if action == "run_analytic_fit" and runcard.get("use_quad", False):
            rep.error(
                "actions_: 'run_analytic_fit' requires a linear model — set use_quad: False"
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

    if "datasets" not in runcard and "external_chi2" not in runcard:
        rep.error("runcard needs 'datasets' and/or 'external_chi2' — no data to fit")
    if "datasets" in runcard:
        for key in ("data_path", "theory_path"):
            if key not in runcard:
                rep.error(f"'{key}' is required when 'datasets' is present")

    coefficients = runcard.get("coefficients")
    if not coefficients:
        rep.error("runcard needs a non-empty 'coefficients' mapping")
    elif isinstance(coefficients, dict):
        prior_dists = keys["prior_dists"] if keys else {}
        skip_prior = "whitening" in runcard or "bayesian_update_path" in runcard
        for name, spec in coefficients.items():
            if skip_prior and isinstance(spec, dict) and spec.get("free", True):
                spec = dict(
                    spec,
                    prior=spec.get("prior", {"dist": "uniform", "low": 0, "high": 1}),
                )
            check_coefficient(name, spec, prior_dists, rep)
    else:
        rep.error("coefficients: must be a mapping of Name: {…}")

    rge = runcard.get("rge")
    if rge is not None:
        if not isinstance(rge, dict) or "init_scale" not in rge:
            rep.error("rge block requires 'init_scale' (GeV)")
        else:
            obs_scale = rge.get("obs_scale", "dynamic")
            if not isinstance(obs_scale, (int, float)) and obs_scale != "dynamic":
                rep.error("rge.obs_scale must be a number or 'dynamic'")

    external = runcard.get("external_chi2")
    if external is not None:
        if not isinstance(external, dict):
            rep.error("external_chi2: must be a mapping ClassName: {path: ..., ...}")
        else:
            for cname, cfg in external.items():
                if not isinstance(cfg, dict) or "path" not in cfg:
                    rep.error(f"external_chi2.{cname}: needs a 'path' key")
                elif not Path(cfg["path"]).exists():
                    rep.error(f"external_chi2.{cname}: path not found: {cfg['path']}")

    check_datasets(runcard, keys, rep)
    check_settings_blocks(runcard, keys, rep)
    check_actions(runcard, rep)
    check_top_level(runcard, keys, rep)

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
