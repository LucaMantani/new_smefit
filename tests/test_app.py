"""Unit tests for smefit/app.py."""

import importlib
import inspect
import pathlib
from unittest.mock import patch

import pytest
from reportengine.app import App
from reportengine.configparser import ExplicitNode

from smefit.app import smefit_providers, smefitApp
from smefit.config import smefitConfig


@pytest.fixture
def app():
    return smefitApp()


def test_float32_short_flag(app):
    args = app.argparser.parse_args(["-f32", "run.yaml"])
    assert args.float32 is True


def test_float32_long_flag(app):
    args = app.argparser.parse_args(["--float32", "run.yaml"])
    assert args.float32 is True


def test_float32_absent_defaults_false(app):
    args = app.argparser.parse_args(["run.yaml"])
    assert args.float32 is False


def test_output_short_flag(app):
    args = app.argparser.parse_args(["-o", "some/dir", "run.yaml"])
    assert args.output == "some/dir"


def test_output_long_flag(app):
    args = app.argparser.parse_args(["--output", "some/dir", "run.yaml"])
    assert args.output == "some/dir"


def test_output_absent_defaults_none(app):
    args = app.argparser.parse_args(["run.yaml"])
    assert args.output is None


def test_get_commandline_arguments_derives_output_from_stem(app):
    """When output is None, get_commandline_arguments should derive it from config_yml stem."""
    fake_args = {"config_yml": "path/to/my_run.yaml", "output": None}
    with patch.object(App, "get_commandline_arguments", return_value=fake_args):
        result = app.get_commandline_arguments()
    assert result["output"] == "my_run"


def test_get_commandline_arguments_keeps_explicit_output(app):
    """When output is already set, get_commandline_arguments should not override it."""
    fake_args = {"config_yml": "path/to/my_run.yaml", "output": "custom_dir"}
    with patch.object(App, "get_commandline_arguments", return_value=fake_args):
        result = app.get_commandline_arguments()
    assert result["output"] == "custom_dir"


def test_get_commandline_arguments_stem_no_parent(app):
    """Stem extraction works when config_yml has no parent directory."""
    fake_args = {"config_yml": "run.yaml", "output": None}
    with patch.object(App, "get_commandline_arguments", return_value=fake_args):
        result = app.get_commandline_arguments()
    assert result["output"] == "run"


# ---------------------------------------------------------------------------
# Annotations under reportengine's runtime type checks
# ---------------------------------------------------------------------------
# reportengine runs isinstance(value, annotation) on the input of every parse_
# (configparser._parse_func) and, while building the graph, on every provider
# parameter whose value the config has already produced
# (resourcebuilder.check_types). Calling these functions directly in a unit
# test bypasses both, so an annotation isinstance rejects only shows up when a
# runcard is run. These tests scan the annotations statically.


def _provider_functions():
    """Public functions of the smefit provider modules: the possible graph nodes."""
    for module_name in smefit_providers:
        if not module_name.startswith("smefit."):
            continue
        module = importlib.import_module(module_name)
        for name, func in vars(module).items():
            if (
                not name.startswith("_")
                and inspect.isfunction(func)
                and func.__module__ == module_name
            ):
                yield f"{module_name}.{name}", func


def _isinstance_rejects(annotation):
    try:
        isinstance(None, annotation)
    except TypeError:
        return True
    return False


def _explicit_node_resources():
    """Resource names produced by an @explicit_node produce_ on smefitConfig."""
    return {
        name.removeprefix("produce_")
        for name, method in inspect.getmembers(smefitConfig, inspect.isfunction)
        if name.startswith("produce_")
        and method.__code__.co_qualname.startswith("explicit_node.")
    }


def test_provider_annotations_accepted_by_isinstance():
    """No parametrised generics (Mapping[str, Any], list[str], ...) on providers."""
    bad = [
        f"{qualname}({param.name}: {param.annotation})"
        for qualname, func in _provider_functions()
        for param in inspect.signature(func).parameters.values()
        if param.annotation is not param.empty and _isinstance_rejects(param.annotation)
    ]
    assert not bad, f"annotations isinstance cannot take: {bad}"


def test_parse_input_annotations_accepted_by_isinstance():
    """The checked argument of every parse_ is the first one after self."""
    bad = []
    for name, method in inspect.getmembers(smefitConfig, inspect.isfunction):
        if not name.startswith("parse_"):
            continue
        params = list(inspect.signature(method).parameters.values())[1:2]
        bad += [
            f"{name}({p.name}: {p.annotation})"
            for p in params
            if p.annotation is not p.empty and _isinstance_rejects(p.annotation)
        ]
    assert not bad, f"annotations isinstance cannot take: {bad}"


def test_explicit_node_parameters_accept_explicit_node():
    """While the graph is built, an @explicit_node resource is its ExplicitNode.

    So a provider parameter it fills must admit ExplicitNode in its annotation
    (see smefit.whitening.WhitenTransformNode), or carry none.
    """
    resources = _explicit_node_resources()
    assert "whitening_transformation" in resources
    node = ExplicitNode(lambda: None)
    bad = [
        f"{qualname}({param.name}: {param.annotation})"
        for qualname, func in _provider_functions()
        for param in inspect.signature(func).parameters.values()
        if param.name in resources
        and param.annotation is not param.empty
        and not isinstance(node, param.annotation)
    ]
    assert not bad, f"annotations rejecting the ExplicitNode: {bad}"
