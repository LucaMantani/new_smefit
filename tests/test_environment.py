"""Unit tests for smefit.environment."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _no_earlier_precision_choice(monkeypatch):
    """The precision a constructor records is process-wide: start each test
    without one, and leave none behind for the rest of the suite."""
    import smefit.environment as environment

    monkeypatch.setattr(environment, "_chosen_float32", None)


def test_float64_mode():
    """float32=False → jax_enable_x64 set to True."""
    mock_backend = MagicMock()
    mock_backend.platform = "cpu"

    with patch("jax.config.update") as mock_update, patch(
        "smefit.environment.jbackend.get_backend", return_value=mock_backend
    ), patch("reportengine.environment.Environment.__init__", return_value=None):
        from smefit.environment import smefitEnvironment

        env = smefitEnvironment(float32=False)

    mock_update.assert_called_with("jax_enable_x64", True)
    assert env.float32 is False


def test_float32_mode():
    """float32=True → jax_enable_x64 set to False."""
    mock_backend = MagicMock()
    mock_backend.platform = "cpu"

    with patch("jax.config.update") as mock_update, patch(
        "smefit.environment.jbackend.get_backend", return_value=mock_backend
    ), patch("reportengine.environment.Environment.__init__", return_value=None):
        from smefit.environment import smefitEnvironment

        env = smefitEnvironment(float32=True)

    mock_update.assert_called_with("jax_enable_x64", False)
    assert env.float32 is True


def test_default_is_float64():
    """Default constructor (no args), with no earlier choice, uses float64."""
    mock_backend = MagicMock()
    mock_backend.platform = "cpu"

    with patch("jax.config.update") as mock_update, patch(
        "smefit.environment.jbackend.get_backend", return_value=mock_backend
    ), patch("reportengine.environment.Environment.__init__", return_value=None):
        from smefit.environment import smefitEnvironment

        env = smefitEnvironment()

    mock_update.assert_called_with("jax_enable_x64", True)
    assert env.float32 is False


def test_unset_precision_keeps_an_earlier_explicit_choice():
    """smefitAPI builds its environment with no argument when smefit.api is
    first imported, which may be inside a ``smefit -f32`` run: it must keep
    float32, not switch the process back to float64."""
    import smefit.environment as environment

    mock_backend = MagicMock()
    mock_backend.platform = "cpu"

    with patch("jax.config.update") as mock_update, patch(
        "smefit.environment.jbackend.get_backend", return_value=mock_backend
    ), patch("reportengine.environment.Environment.__init__", return_value=None):
        environment.smefitEnvironment(float32=True)  # the CLI, with -f32
        env = environment.smefitEnvironment()  # smefitAPI's

    mock_update.assert_called_with("jax_enable_x64", False)
    assert env.float32 is True
