"""Unit tests for smefit/app.py."""

import pathlib
from unittest.mock import patch

import pytest
from reportengine.app import App

from smefit.app import smefitApp


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
