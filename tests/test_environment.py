"""Unit tests for smefit.environment."""

from unittest.mock import MagicMock, patch


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
    """Default constructor (no args) should use float64."""
    mock_backend = MagicMock()
    mock_backend.platform = "cpu"

    with patch("jax.config.update") as mock_update, patch(
        "smefit.environment.jbackend.get_backend", return_value=mock_backend
    ), patch("reportengine.environment.Environment.__init__", return_value=None):
        from smefit.environment import smefitEnvironment

        env = smefitEnvironment()

    mock_update.assert_called_with("jax_enable_x64", True)
    assert env.float32 is False
