"""Unit tests for smefit.utils_actions."""

from pathlib import Path
from unittest.mock import MagicMock

from smefit.pca import PCA
from smefit.utils_actions import run_pca


def test_run_pca_delegates_to_the_pca_object():
    """The action only prints and writes; the analysis itself is the pca node."""
    result = MagicMock(spec=PCA)
    out = Path("/output")

    run_pca(result, out)

    result.print_summary.assert_called_once_with()
    result.write.assert_called_once_with(out)
