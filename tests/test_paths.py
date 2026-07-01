"""Unit tests for smefit/paths.py — resolve_path."""

from unittest.mock import patch

import pytest

from smefit.paths import resolve_path


def test_resolve_smefit_database_prefix():
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"path_to_smefit_database": "/data/smefit_database"},
    ):
        assert (
            resolve_path("smefit_database/commondata")
            == "/data/smefit_database/commondata"
        )


def test_resolve_new_smefit_prefix():
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"path_to_new_smefit": "/home/user/new_smefit"},
    ):
        assert (
            resolve_path("new_smefit/external_chi2/foo.py")
            == "/home/user/new_smefit/external_chi2/foo.py"
        )


def test_resolve_smefit_results_prefix():
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"path_to_smefit_results": "/data/results"},
    ):
        assert resolve_path("smefit_results/myfit") == "/data/results/myfit"


def test_resolve_prefix_only_no_slash():
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"path_to_smefit_database": "/data/smefit_database"},
    ):
        assert resolve_path("smefit_database") == "/data/smefit_database"


def test_resolve_missing_key_raises():
    with patch("smefit.paths.load_user_paths", return_value={}):
        with pytest.raises(ValueError, match="smefit_setup_paths"):
            resolve_path("smefit_database/commondata")


def test_resolve_absolute_path_unchanged():
    assert resolve_path("/absolute/path/to/data") == "/absolute/path/to/data"


def test_resolve_unknown_prefix_unchanged():
    assert resolve_path("some_other_dir/foo") == "some_other_dir/foo"


def test_resolve_trailing_slash_stripped_from_base():
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"path_to_smefit_database": "/data/smefit_database/"},
    ):
        assert (
            resolve_path("smefit_database/commondata")
            == "/data/smefit_database/commondata"
        )
