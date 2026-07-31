"""Unit tests for smefit.utils_actions."""

from unittest.mock import MagicMock

from smefit.utils_actions import write_rge_matrix


def test_write_rge_matrix_delegates_to_the_matrix(tmp_path):
    """The action stays thin: the RGEMatrix owns its own serialisation."""
    rge_matrix = MagicMock()

    write_rge_matrix(rge_matrix, tmp_path)

    rge_matrix.write.assert_called_once_with(tmp_path)


def test_write_rge_matrix_produces_the_reusable_pickle(tmp_path):
    """End to end with a real RGEMatrix: the file a later runcard points at."""
    import jax.numpy as jnp

    from smefit.rge import RGEMatrix, RGESettings, load_precomputed_rge_matrix

    matrix = RGEMatrix(
        stacked_mats=jnp.array([[[1.0], [2.0]]]),
        obs_operators=["OpBox", "OpD"],
        init_operators=["OpBox"],
        scales=[91.2],
        settings=RGESettings(init_scale=1000.0),
    )

    write_rge_matrix(matrix, tmp_path)

    out_file = tmp_path / "rge_matrix.pkl"
    assert out_file.exists()
    cache = load_precomputed_rge_matrix(out_file, matrix.settings.to_dict())
    assert list(cache) == [91.2]
