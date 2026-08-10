"""Tests for the standalone scripts bundled with the agent skills.

`validate_runcard.py` and `smefit_db.py` are stdlib-only scripts that the agent
is instructed to trust, so they need the same coverage as the package itself.
They are run as subprocesses (never imported), which is exactly how a skill
invokes them.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS = REPO_ROOT / ".agents" / "skills"
VALIDATOR = SKILLS / "smefit-runcard" / "scripts" / "validate_runcard.py"
SMEFIT_DB = SKILLS / "smefit-datasets" / "scripts" / "smefit_db.py"
FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATES = sorted((REPO_ROOT / "template_runcards").glob("*.yaml"))


def run_script(script, *args, cwd=None):
    """Run a skill script the way a skill would; return (returncode, output)."""
    proc = subprocess.run(
        [sys.executable, str(script), *[str(a) for a in args]],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


def write_runcard(tmp_path, **overrides):
    """A minimal runcard against tests/fixtures, with per-test overrides.

    Keys set to None are removed, so tests can drop a required key.
    """
    runcard = {
        "data_path": str(FIXTURES / "commondata"),
        "theory_path": str(FIXTURES / "theory"),
        "use_quad": False,
        "datasets": [{"name": "TESTDATA", "order": "NLO"}],
        "coefficients": {
            "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1, "high": 1}}
        },
        "actions_": ["run_analytic_fit"],
    }
    runcard.update(overrides)
    runcard = {k: v for k, v in runcard.items() if v is not None}
    path = tmp_path / "runcard.yaml"
    path.write_text(yaml.safe_dump(runcard))
    return path


def validate(tmp_path, **overrides):
    return run_script(VALIDATOR, write_runcard(tmp_path, **overrides))


# ---------------------------------------------------------------------------
# validate_runcard.py — happy path
# ---------------------------------------------------------------------------


def test_minimal_runcard_is_valid(tmp_path):
    code, out = validate(tmp_path)
    assert code == 0, out
    assert "OK (0 warning(s))" in out


def test_absolute_paths_do_not_warn_about_missing_paths_config(tmp_path):
    """Regression guard: the .config/paths.yaml hint is only for runcards whose
    paths actually need it — on a machine without one (e.g. CI) a runcard using
    absolute paths must still be warning-free."""
    code, out = validate(tmp_path)
    assert code == 0, out
    assert "paths.yaml" not in out


def test_baseline_value_is_a_known_coefficient_key(tmp_path):
    """Regression guard: baseline_value comes from the Coefficient dataclass and
    must not be reported as an unknown sub-key."""
    code, out = validate(
        tmp_path,
        coefficients={
            "OpA": {
                "free": True,
                "prior": {"dist": "uniform", "low": -1, "high": 1},
                "baseline_value": 0.5,
            }
        },
    )
    assert code == 0, out
    assert "unknown sub-key" not in out


def test_fixed_and_expression_coefficients_are_valid(tmp_path):
    code, out = validate(
        tmp_path,
        coefficients={
            "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1, "high": 1}},
            "OpB": {"free": False, "value": 0.1},
            "aux": {"free": False, "vars": ["OpA"], "expr": "2*OpA"},
        },
        actions_=["run_ultranest_fit"],
    )
    assert code == 0, out


# ---------------------------------------------------------------------------
# validate_runcard.py — coefficient invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected",
    [
        ({"free": True, "value": 1.0}, "free=True forbids 'value'"),
        (
            {"free": True, "prior": {"dist": "cauchy", "scale": 1.0}},
            "unknown prior dist",
        ),
        (
            {"free": True, "prior": {"dist": "uniform", "low": -1}},
            "needs exactly",
        ),
        (
            {"free": False, "value": 1.0, "prior": {"dist": "uniform"}},
            "forbids 'prior'",
        ),
        ({"free": False}, "exactly one of 'value' or 'expr'"),
        ({"free": False, "expr": "2*OpA"}, "requires non-empty 'vars'"),
        (
            {"free": False, "vars": ["OpA", "OpA"], "expr": "OpA"},
            "'vars' contains duplicates",
        ),
    ],
)
def test_coefficient_invariants_are_rejected(tmp_path, spec, expected):
    code, out = validate(tmp_path, coefficients={"OpA": spec})
    assert code == 1, out
    assert expected in out


def test_free_coefficient_without_prior_is_valid_for_non_sampler_actions(tmp_path):
    """run_analytic_fit (and run_hessian_fit, gradient-descent fits) never
    resolve reportengine's `prior` node, so a free coefficient without one is
    legitimate — regression guard against over-requiring priors."""
    code, out = validate(tmp_path, coefficients={"OpA": {"free": True}})
    assert code == 0, out


def test_free_coefficient_needs_prior_for_sampler_action(tmp_path):
    code, out = validate(
        tmp_path,
        coefficients={"OpA": {"free": True}},
        actions_=["run_ultranest_fit"],
    )
    assert code == 1, out
    assert "needs a prior" in out


def _previous_fit_dir(tmp_path):
    """The minimal layout parse_bayesian_update_path insists on."""
    fit_dir = tmp_path / "previous_fit"
    (fit_dir / "input").mkdir(parents=True)
    (fit_dir / "fit_results.json").write_text("{}")
    (fit_dir / "input" / "runcard.yaml").write_text("")
    return fit_dir


@pytest.mark.parametrize(
    "extra_key, extra_value",
    [
        ("whitening", {"sigma_prior": 5.0}),
        ("bayesian_update_path", None),  # replaced by a real fit dir below
    ],
)
def test_free_coefficient_without_prior_is_valid_for_sampler_with_synthesized_prior(
    tmp_path, extra_key, extra_value
):
    """whitening/bayesian_update_path make reportengine build the `prior`
    node itself, without needing a per-coefficient spec."""
    if extra_key == "bayesian_update_path":
        extra_value = str(_previous_fit_dir(tmp_path))
    code, out = validate(
        tmp_path,
        coefficients={"OpA": {"free": True}},
        actions_=["run_ultranest_fit"],
        **{extra_key: extra_value},
    )
    assert code == 0, out


def test_missing_bayesian_update_path_errors(tmp_path):
    code, out = validate(tmp_path, bayesian_update_path=str(tmp_path / "no_such_fit"))
    assert code == 1, out
    assert "bayesian_update_path" in out


def test_bayesian_update_path_without_fit_results_errors(tmp_path):
    fit_dir = tmp_path / "previous_fit"
    fit_dir.mkdir()
    code, out = validate(tmp_path, bayesian_update_path=str(fit_dir))
    assert code == 1, out
    assert "fit_results.json not found" in out


def test_unknown_coefficient_subkey_warns(tmp_path):
    code, out = validate(
        tmp_path,
        coefficients={
            "OpA": {
                "free": True,
                "prior": {"dist": "uniform", "low": -1, "high": 1},
                "baselinevalue": 0.5,  # typo of baseline_value
            }
        },
    )
    assert code == 0, out
    assert "unknown sub-key 'baselinevalue'" in out


def test_vars_referencing_undefined_coefficient_is_an_error(tmp_path):
    code, out = validate(
        tmp_path,
        coefficients={
            "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1, "high": 1}},
            "OpB": {"free": False, "vars": ["nope"], "expr": "nope**2"},
        },
        actions_=["run_ultranest_fit"],
    )
    assert code == 1, out
    assert "references 'nope'" in out


def test_operator_absent_from_theory_warns(tmp_path):
    code, out = validate(
        tmp_path,
        coefficients={
            "OpNotThere": {
                "free": True,
                "prior": {"dist": "uniform", "low": -1, "high": 1},
            }
        },
    )
    assert code == 0, out
    assert "no selected dataset's theory file contains this operator" in out


def test_operator_cross_check_skipped_with_rge(tmp_path):
    """RGE mixes a high-scale basis into the theory operators, so a coefficient
    absent from every theory file is legitimate there."""
    code, out = validate(
        tmp_path,
        coefficients={
            "OpNotThere": {
                "free": True,
                "prior": {"dist": "uniform", "low": -1, "high": 1},
            }
        },
        rge={"init_scale": 1000.0, "obs_scale": "dynamic"},
        actions_=["run_ultranest_fit"],
    )
    assert code == 0, out
    assert "no selected dataset's theory file" not in out


# ---------------------------------------------------------------------------
# validate_runcard.py — actions, datasets, structure
# ---------------------------------------------------------------------------


def test_analytic_fit_rejects_quadratic_model(tmp_path):
    code, out = validate(tmp_path, use_quad=True)
    assert code == 1, out
    assert "requires a linear model" in out


def test_analytic_fit_rejects_nonlinear_expression(tmp_path):
    code, out = validate(
        tmp_path,
        coefficients={
            "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1, "high": 1}},
            "OpB": {"free": False, "vars": ["OpA"], "expr": "OpA**2"},
        },
    )
    assert code == 1, out
    assert "nonlinear" in out


def test_analytic_fit_accepts_linear_expression(tmp_path):
    code, out = validate(
        tmp_path,
        coefficients={
            "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1, "high": 1}},
            "OpB": {"free": False, "vars": ["OpA"], "expr": "100*OpA"},
        },
    )
    assert code == 0, out


def test_report_action_needs_a_template(tmp_path):
    code, out = validate(tmp_path, actions_=["report"])
    assert code == 1, out
    assert "requires a 'template_text'" in out


def test_unknown_order_is_rejected(tmp_path):
    code, out = validate(tmp_path, datasets=[{"name": "TESTDATA", "order": "N3LO"}])
    assert code == 1, out
    assert "not in theory file" in out


def test_unknown_dataset_is_rejected(tmp_path):
    code, out = validate(tmp_path, datasets=[{"name": "NOPE", "order": "NLO"}])
    assert code == 1, out
    assert "no commondata file" in out


def test_unknown_theory_cov_is_rejected(tmp_path):
    code, out = validate(
        tmp_path,
        datasets=[{"name": "TESTDATA", "order": "NLO", "theory_cov": "aggressive"}],
    )
    assert code == 1, out
    assert "theory_cov 'aggressive' not in theory file" in out


def test_runcard_without_any_data_is_rejected(tmp_path):
    code, out = validate(tmp_path, datasets=None, data_path=None, theory_path=None)
    assert code == 1, out
    assert "no data to fit" in out


def test_missing_actions_is_rejected(tmp_path):
    code, out = validate(tmp_path, actions_=None)
    assert code == 1, out
    assert "non-empty 'actions_' list" in out


def test_unknown_settings_subkey_warns_but_passes(tmp_path):
    code, out = validate(
        tmp_path,
        actions_=["run_ultranest_fit"],
        ultranest_settings={"nonsense_key": 1},
    )
    assert code == 0, out
    assert "unknown sub-key 'nonsense_key'" in out


def test_strict_mode_promotes_warnings_to_errors(tmp_path):
    runcard = write_runcard(tmp_path, unknown_top_level_key=1)
    assert run_script(VALIDATOR, runcard)[0] == 0
    code, out = run_script(VALIDATOR, runcard, "--strict")
    assert code == 1, out


def test_invalid_yaml_exits_two(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("coefficients: {unclosed\n")
    code, out = run_script(VALIDATOR, path)
    assert code == 2, out


def test_missing_file_exits_two(tmp_path):
    code, out = run_script(VALIDATOR, tmp_path / "absent.yaml")
    assert code == 2, out


# ---------------------------------------------------------------------------
# validate_runcard.py — the shipped templates must stay valid
# ---------------------------------------------------------------------------


def database_configured():
    """True when a smefit_database clone is reachable (skip marker for templates)."""
    from smefit.paths import resolve_path

    try:
        return Path(resolve_path("smefit_database/commondata")).is_dir()
    except Exception:
        return False


@pytest.mark.skipif(
    not database_configured(),
    reason="no smefit_database clone configured in .config/paths.yaml",
)
@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_template_runcards_validate(template):
    """Templates must be clean, warnings included.

    Warnings are the drift signal: a key added to the code and used in a
    template, but not yet reflected in the generated runcard-keys.json (or in
    the curated `dataset_entry_keys`), shows up here as "unknown key".
    """
    code, out = run_script(VALIDATOR, template)
    assert code == 0, out
    assert "OK (0 warning(s))" in out, out


def test_templates_are_discovered():
    """Guard against the parametrisation above silently collecting nothing."""
    assert TEMPLATES


def test_paths_helper_is_mirrored_between_scripts():
    """find_paths_config() is duplicated on purpose (self-contained skill dirs);
    this keeps the two copies from drifting apart."""

    def extract(path):
        lines = path.read_text().splitlines()
        start = next(i for i, ln in enumerate(lines) if "def find_paths_config" in ln)
        body = []
        for line in lines[start + 1 :]:
            if line.strip() and not line.startswith((" ", "\t")):
                break  # first top-level statement after the function
            body.append(line)
        # Drop the docstring: it names the *other* file, so it differs by design.
        quotes = [i for i, ln in enumerate(body) if ln.strip().startswith('"""')]
        if len(quotes) >= 2:
            body = body[quotes[1] + 1 :]
        return "\n".join(body).strip()

    assert extract(VALIDATOR) == extract(SMEFIT_DB)


# ---------------------------------------------------------------------------
# smefit_db.py
# ---------------------------------------------------------------------------


def test_smefit_db_requires_a_subcommand():
    code, _ = run_script(SMEFIT_DB)
    assert code == 2


def test_smefit_db_clone_only_prints(tmp_path):
    code, out = run_script(SMEFIT_DB, "clone", tmp_path / "dest", cwd=tmp_path)
    assert code == 0, out
    assert "git clone" in out
    assert not (tmp_path / "dest").exists(), "clone must never execute"


def test_smefit_db_locate_is_json_parseable_or_reports_absence(tmp_path):
    code, out = run_script(SMEFIT_DB, "--json", "--offline", "locate", cwd=tmp_path)
    assert code in (0, 3), out
    if code == 0:
        payload = json.loads(out)
        assert Path(payload["resolved_data_path"]).is_dir()
        assert Path(payload["resolved_theory_path"]).is_dir()


@pytest.mark.skipif(
    not database_configured(),
    reason="no smefit_database clone configured in .config/paths.yaml",
)
@pytest.mark.parametrize(
    "args", [("search", "zzzznomatch"), ("info", "NOPE"), ("operators", "zzzznomatch")]
)
def test_smefit_db_reports_no_match_with_exit_one(args):
    code, _ = run_script(SMEFIT_DB, "--offline", *args)
    assert code == 1
