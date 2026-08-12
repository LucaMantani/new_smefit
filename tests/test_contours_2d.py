"""Unit tests for smefit.contours_2d — the 2D confidence-contour primitives.

Figures are never drawn or saved here, only their artists inspected, so the
suite never depends on a LaTeX installation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from smefit.contours_2d import (
    KDEGrid,
    confidence_ellipse,
    density_level,
    ellipse_half_axis,
    fit_colors,
    kde_contour,
    kde_grid,
    plot_contours,
    plot_uncorrelated_contours,
    split_solution,
    uncorrelated_ellipse,
)

# The (x, y) sample arrays of the gaussian_samples fixture, and the
# name-to-samples posterior built from them — the FitResult.samples form.
SamplePair = tuple[np.ndarray, np.ndarray]
Posterior = dict[str, jax.Array]


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture
def gaussian_samples() -> SamplePair:
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    values = rng.multivariate_normal([0.3, -0.2], cov, size=4000)
    return values[:, 0], values[:, 1]


# ---------------------------------------------------------------------------
# split_solution
# ---------------------------------------------------------------------------


def test_split_solution_returns_solution_closest_to_zero_first() -> None:
    samples = np.concatenate([np.linspace(-0.1, 0.1, 50), np.linspace(4.9, 5.1, 50)])

    first, second = split_solution(samples)

    assert np.abs(first).min() < np.abs(second).min()
    assert first.max() < second.min()


def test_split_solution_orders_by_distance_to_zero_not_by_value() -> None:
    """A negative solution closer to zero is still returned first."""
    samples = np.concatenate([np.linspace(-5.1, -4.9, 50), np.linspace(-0.1, 0.1, 50)])

    first, second = split_solution(samples)

    assert np.abs(first).min() < np.abs(second).min()
    assert first.min() > second.max()


def test_split_solution_single_mode_returns_full_sample_twice() -> None:
    """A unimodal sample cannot be split: both solutions are the full sample."""
    samples = np.array([1.0, 1.0, 1.0])

    first, second = split_solution(samples)

    assert np.array_equal(first, samples)
    assert np.array_equal(second, samples)


# ---------------------------------------------------------------------------
# confidence_ellipse
# ---------------------------------------------------------------------------


def test_confidence_ellipse_is_centred_on_the_median(
    gaussian_samples: SamplePair,
) -> None:
    x_values, y_values = gaussian_samples
    _, ax = plt.subplots()

    ellipse = confidence_ellipse(x_values, y_values, ax)

    # the affine transform translates the (0, 0)-centred ellipse to the median
    center = ellipse.get_transform().transform((0, 0))
    expected = ax.transData.transform((np.median(x_values), np.median(y_values)))
    np.testing.assert_allclose(center, expected, rtol=1e-6)


def test_confidence_ellipse_grows_with_confidence_level(
    gaussian_samples: SamplePair,
) -> None:
    x_values, y_values = gaussian_samples
    _, ax = plt.subplots()

    small = confidence_ellipse(x_values, y_values, ax, confidence_level=68)
    large = confidence_ellipse(x_values, y_values, ax, confidence_level=95)

    assert large.get_width() > small.get_width()
    assert large.get_height() > small.get_height()


def test_confidence_ellipse_axes_match_the_sample_covariance(
    gaussian_samples: SamplePair,
) -> None:
    """Semi-axes are sqrt(chi2_quantile * eigenvalue) of the sample covariance."""
    import scipy.stats

    x_values, y_values = gaussian_samples
    _, ax = plt.subplots()

    ellipse = confidence_ellipse(x_values, y_values, ax, confidence_level=95)

    eig = np.sort(np.linalg.eigvalsh(np.cov(x_values, y_values)))
    chi2_qnt = scipy.stats.chi2.ppf(0.95, 2)
    assert ellipse.get_width() == pytest.approx(2 * np.sqrt(chi2_qnt * eig[-1]))
    assert ellipse.get_height() == pytest.approx(2 * np.sqrt(chi2_qnt * eig[0]))


def test_confidence_ellipse_rejects_mismatched_sizes() -> None:
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match="same size"):
        confidence_ellipse(np.zeros(10), np.zeros(11), ax)


def test_confidence_ellipse_is_added_to_the_axes(
    gaussian_samples: SamplePair,
) -> None:
    x_values, y_values = gaussian_samples
    _, ax = plt.subplots()

    ellipse = confidence_ellipse(x_values, y_values, ax)

    assert ellipse in ax.patches


# ---------------------------------------------------------------------------
# uncorrelated_ellipse
# ---------------------------------------------------------------------------


def test_uncorrelated_ellipse_axes_are_the_scaled_standard_deviations() -> None:
    """A point known by a central value and a std per axis: the ellipse is
    sqrt(chi2 quantile) standard deviations wide, as the sample one is sqrt of
    the same quantile times a covariance eigenvalue."""
    import scipy.stats

    _, ax = plt.subplots()

    ellipse = uncorrelated_ellipse((1.0, -2.0), (0.5, 0.25), ax, confidence_level=95)

    scale = np.sqrt(scipy.stats.chi2.ppf(0.95, 2))
    assert ellipse.get_width() == pytest.approx(2 * scale * 0.5)
    assert ellipse.get_height() == pytest.approx(2 * scale * 0.25)
    assert ellipse.get_center() == (1.0, -2.0)


def test_uncorrelated_ellipse_is_axis_aligned() -> None:
    """Nothing says how the two coefficients covary, so the ellipse cannot be
    tilted: that is the assumption the name carries."""
    _, ax = plt.subplots()

    ellipse = uncorrelated_ellipse((0.0, 0.0), (1.0, 2.0), ax)

    assert ellipse.get_angle() == pytest.approx(0.0)


def test_uncorrelated_ellipse_grows_with_confidence_level() -> None:
    _, ax = plt.subplots()

    small = uncorrelated_ellipse((0.0, 0.0), (1.0, 1.0), ax, confidence_level=68)
    large = uncorrelated_ellipse((0.0, 0.0), (1.0, 1.0), ax, confidence_level=95)

    assert large.get_width() > small.get_width()


def test_uncorrelated_ellipse_is_added_to_the_axes_unfilled_by_default() -> None:
    """Like confidence_ellipse, the primitive draws an outline and the caller
    layers a fill on top."""
    _, ax = plt.subplots()

    ellipse = uncorrelated_ellipse((0.0, 0.0), (1.0, 1.0), ax, edgecolor="firebrick")

    assert ellipse in ax.patches
    assert ellipse.get_facecolor()[3] == 0.0  # transparent fill


def test_plot_uncorrelated_contours_draws_an_outline_and_a_fill() -> None:
    """The convention plot_contours uses for a linear fit: a solid outline and
    a translucent fill of the same colour, so the two read alike."""
    _, ax = plt.subplots()

    outline, fill = plot_uncorrelated_contours(ax, (0.0, 0.0), (1.0, 0.5), "C0")

    assert len(ax.patches) == 2
    assert fill.get_alpha() == pytest.approx(0.3)
    assert outline.get_alpha() == pytest.approx(1)
    assert outline.get_facecolor()[3] == 0.0
    assert outline.get_width() == pytest.approx(fill.get_width())


def test_plot_uncorrelated_contours_adds_the_dashed_outer_level() -> None:
    """Two confidence levels draw the outer one dashed, as the fits' do."""
    _, ax = plt.subplots()

    _, fill = plot_uncorrelated_contours(
        ax,
        (0.0, 0.0),
        (1.0, 1.0),
        "C0",
        confidence_level=95,
        dashed_confidence_level=68,
    )

    assert len(ax.patches) == 3
    dashed = ax.patches[0]
    assert dashed.get_linestyle() == "dashed"
    assert dashed.get_width() < fill.get_width()  # 68% inside the filled 95%


def test_ellipse_half_axis_scales_the_standard_deviation() -> None:
    assert ellipse_half_axis(2.0, 95) == pytest.approx(2 * ellipse_half_axis(1.0, 95))


# ---------------------------------------------------------------------------
# kde_grid / density_level
# ---------------------------------------------------------------------------


def test_kde_grid_shapes_and_support(gaussian_samples: SamplePair) -> None:
    x_values, y_values = gaussian_samples

    xx, yy, density = kde_grid(x_values, y_values, gridsize=50)

    assert xx.shape == yy.shape == density.shape == (50, 50)
    # the grid is padded beyond the samples so the tails are not truncated
    assert xx.min() < x_values.min() and xx.max() > x_values.max()
    assert yy.min() < y_values.min() and yy.max() > y_values.max()
    assert (density > 0).all()


def test_kde_grid_bandwidth_adjust_widens_the_density(
    gaussian_samples: SamplePair,
) -> None:
    x_values, y_values = gaussian_samples

    _, _, narrow = kde_grid(x_values, y_values, bw_adjust=0.5, gridsize=50)
    _, _, wide = kde_grid(x_values, y_values, bw_adjust=2.0, gridsize=50)

    assert wide.max() < narrow.max()


def test_density_level_encloses_the_requested_mass() -> None:
    """The mass above the returned level matches the requested confidence."""
    rng = np.random.default_rng(1)
    x_values, y_values = rng.multivariate_normal([0, 0], np.eye(2), size=5000).T
    _, _, density = kde_grid(x_values, y_values, gridsize=200)

    level = density_level(density, 95)

    mass = density[density > level].sum() / density.sum()
    assert mass == pytest.approx(0.95, abs=0.01)


def test_density_level_is_monotonic_in_confidence_level() -> None:
    """A tighter contour sits at a higher density level."""
    rng = np.random.default_rng(2)
    x_values, y_values = rng.multivariate_normal([0, 0], np.eye(2), size=2000).T
    _, _, density = kde_grid(x_values, y_values, gridsize=100)

    assert density_level(density, 68) > density_level(density, 95)


def test_kde_contour_draws_a_single_level(gaussian_samples: SamplePair) -> None:
    x_values, y_values = gaussian_samples
    _, ax = plt.subplots()

    contour = kde_contour(x_values, y_values, ax, color="C0", confidence_level=95)

    assert len(contour.levels) == 1
    assert contour.levels[0] == pytest.approx(
        density_level(kde_grid(x_values, y_values)[2], 95)
    )


def test_kde_grid_thins_large_posteriors(gaussian_samples: SamplePair) -> None:
    """Above max_samples the posterior is strided down before estimating."""
    x_values, y_values = gaussian_samples

    capped = kde_grid(x_values, y_values, gridsize=40, max_samples=200)
    thinned = kde_grid(x_values[::20], y_values[::20], gridsize=40, max_samples=None)

    np.testing.assert_allclose(capped[2], thinned[2])


def test_kde_grid_rejects_mismatched_sizes() -> None:
    """Two samples of different length are not the coordinates of one set of
    draws. Thinning is what makes this worth checking rather than leaving to
    numpy: a common stride erases a difference of one (2001 and 2000 samples
    both thin to 1000), and the estimate would silently pair up samples from
    different draws."""
    rng = np.random.default_rng(4)

    with pytest.raises(ValueError, match="same size"):
        kde_grid(rng.normal(size=2001), rng.normal(size=2000), max_samples=1000)


def test_kde_grid_leaves_small_posteriors_untouched() -> None:
    rng = np.random.default_rng(3)
    x_values, y_values = rng.multivariate_normal([0, 0], np.eye(2), size=100).T

    capped = kde_grid(x_values, y_values, gridsize=40, max_samples=5000)
    full = kde_grid(x_values, y_values, gridsize=40, max_samples=None)

    np.testing.assert_allclose(capped[2], full[2])


def test_kde_contour_accepts_a_precomputed_grid(gaussian_samples: SamplePair) -> None:
    """Passing `grid` skips the KDE and draws the same contour."""
    x_values, y_values = gaussian_samples
    _, ax = plt.subplots()
    grid = kde_grid(x_values, y_values)

    reused = kde_contour(x_values, y_values, ax, color="C0", grid=grid)
    recomputed = kde_contour(x_values, y_values, ax, color="C0")

    assert reused.levels == pytest.approx(recomputed.levels)


def test_kde_contour_fill_spans_level_to_peak(gaussian_samples: SamplePair) -> None:
    x_values, y_values = gaussian_samples
    _, ax = plt.subplots()

    contour = kde_contour(x_values, y_values, ax, color="C0", fill=True)

    assert len(contour.levels) == 2
    assert contour.levels[0] < contour.levels[1]


# ---------------------------------------------------------------------------
# plot_contours
# ---------------------------------------------------------------------------


@pytest.fixture
def posterior_samples(gaussian_samples: SamplePair) -> Posterior:
    """Posterior in the form plot_contours consumes: FitResult.samples."""
    x_values, y_values = gaussian_samples
    return {"OpA": jnp.array(x_values), "OpB": jnp.array(y_values)}


def test_plot_contours_ellipse_mode_adds_two_patches(
    posterior_samples: Posterior,
) -> None:
    _, ax = plt.subplots()

    hndls = plot_contours(ax, posterior_samples, "OpA", "OpB", kde=False, color="C0")

    assert len(hndls) == 2
    assert len(ax.patches) == 2
    # no best-fit marker unless asked for
    assert len(ax.collections) == 0


def test_plot_contours_accepts_a_dataframe(posterior_samples: Posterior) -> None:
    """A DataFrame maps a column name to its values like the dict does."""
    frame = pd.DataFrame({name: np.asarray(v) for name, v in posterior_samples.items()})
    _, ax = plt.subplots()

    hndls = plot_contours(ax, frame, "OpA", "OpB", kde=False, color="C0")

    assert len(hndls) == 2
    assert len(ax.patches) == 2


def test_plot_contours_show_best_fit_adds_the_marker(
    posterior_samples: Posterior,
) -> None:
    _, ax = plt.subplots()

    plot_contours(
        ax, posterior_samples, "OpA", "OpB", kde=False, color="C0", show_best_fit=True
    )

    assert len(ax.collections) == 1


def test_plot_contours_best_fit_overrides_the_posterior_mean(
    posterior_samples: Posterior,
) -> None:
    _, ax = plt.subplots()

    plot_contours(
        ax,
        posterior_samples,
        "OpA",
        "OpB",
        kde=False,
        color="C0",
        show_best_fit=True,
        best_fit=(1.5, -2.5),
    )

    assert tuple(ax.collections[0].get_offsets()[0]) == (1.5, -2.5)


def test_plot_contours_kde_mode_marks_one_point_per_solution(
    posterior_samples: Posterior,
) -> None:
    _, ax = plt.subplots()

    plot_contours(
        ax,
        posterior_samples,
        "OpA",
        "OpB",
        kde=True,
        color="C0",
        double_solution=["OpA"],
        show_best_fit=True,
    )

    scatters = [c for c in ax.collections if c.get_offsets().shape[-1] == 2]
    marker_x = sorted(float(s.get_offsets()[0][0]) for s in scatters[-2:])
    solution1, solution2 = split_solution(posterior_samples["OpA"])
    assert marker_x == pytest.approx(
        sorted([np.mean(solution1), np.mean(solution2)]), rel=1e-6
    )


def test_plot_contours_kde_mode_splits_the_second_coefficient(
    posterior_samples: Posterior,
) -> None:
    """A double solution on coeff2 splits the y values, coeff1 stays whole."""
    _, ax = plt.subplots()

    plot_contours(
        ax,
        posterior_samples,
        "OpA",
        "OpB",
        kde=True,
        color="C0",
        double_solution=["OpB"],
        show_best_fit=True,
    )

    scatters = [c for c in ax.collections if c.get_offsets().shape[-1] == 2]
    marker_y = sorted(float(s.get_offsets()[0][1]) for s in scatters[-2:])
    solution1, solution2 = split_solution(posterior_samples["OpB"])
    assert marker_y == pytest.approx(
        sorted([np.mean(solution1), np.mean(solution2)]), rel=1e-6
    )
    # coeff1 is not split, so both markers share the same x
    marker_x = [float(s.get_offsets()[0][0]) for s in scatters[-2:]]
    assert marker_x[0] == pytest.approx(marker_x[1])


def test_plot_contours_kde_mode_best_fit_overrides_the_posterior_mean(
    posterior_samples: Posterior,
) -> None:
    _, ax = plt.subplots()

    plot_contours(
        ax,
        posterior_samples,
        "OpA",
        "OpB",
        kde=True,
        color="C0",
        show_best_fit=True,
        best_fit=(1.5, -2.5),
    )

    scatters = [c for c in ax.collections if c.get_offsets().shape[-1] == 2]
    assert tuple(scatters[-1].get_offsets()[0]) == (1.5, -2.5)


def test_plot_contours_kde_mode_double_solution_ignores_the_best_fit(
    posterior_samples: Posterior,
) -> None:
    """A single stored best-fit cannot represent two modes: mark both means."""
    _, ax = plt.subplots()

    plot_contours(
        ax,
        posterior_samples,
        "OpA",
        "OpB",
        kde=True,
        color="C0",
        double_solution=["OpA"],
        show_best_fit=True,
        best_fit=(1.5, -2.5),
    )

    scatters = [c for c in ax.collections if c.get_offsets().shape[-1] == 2]
    marker_x = sorted(float(s.get_offsets()[0][0]) for s in scatters[-2:])
    solution1, solution2 = split_solution(posterior_samples["OpA"])
    assert marker_x == pytest.approx(
        sorted([np.mean(solution1), np.mean(solution2)]), rel=1e-6
    )


def test_plot_contours_kde_mode_without_double_solution_marks_the_mean(
    posterior_samples: Posterior,
) -> None:
    _, ax = plt.subplots()

    plot_contours(
        ax, posterior_samples, "OpA", "OpB", kde=True, color="C0", show_best_fit=True
    )

    scatters = [c for c in ax.collections if c.get_offsets().shape[-1] == 2]
    both = [s.get_offsets()[0] for s in scatters[-2:]]
    np.testing.assert_allclose(both[0], both[1])
    assert both[0][0] == pytest.approx(np.mean(posterior_samples["OpA"]))


def test_plot_contours_dashed_level_adds_one_ellipse(
    posterior_samples: Posterior,
) -> None:
    _, ax = plt.subplots()

    plot_contours(
        ax,
        posterior_samples,
        "OpA",
        "OpB",
        kde=False,
        color="C0",
        confidence_level=95,
        dashed_confidence_level=68,
    )

    assert len(ax.patches) == 3  # dashed 68% outline + filled/outlined 95%
    assert ax.patches[0].get_linestyle() == "dashed"


def test_plot_contours_dashed_level_adds_one_kde_contour(
    posterior_samples: Posterior,
) -> None:
    _, ax = plt.subplots()

    with_dashed = plt.subplots()[1]
    plot_contours(ax, posterior_samples, "OpA", "OpB", kde=True, color="C0")
    plot_contours(
        with_dashed,
        posterior_samples,
        "OpA",
        "OpB",
        kde=True,
        color="C0",
        dashed_confidence_level=68,
    )

    assert len(with_dashed.collections) == len(ax.collections) + 1


def test_plot_contours_kde_evaluates_the_density_once(
    posterior_samples: Posterior, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All contours of a panel share a single KDE evaluation."""
    import smefit.contours_2d as contours_mod

    calls: list[int] = []
    real_kde_grid = contours_mod.kde_grid

    def counting_kde_grid(*args: Any, **kwargs: Any) -> KDEGrid:
        calls.append(1)
        return real_kde_grid(*args, **kwargs)

    monkeypatch.setattr(contours_mod, "kde_grid", counting_kde_grid)
    _, ax = plt.subplots()

    contours_mod.plot_contours(
        ax,
        posterior_samples,
        "OpA",
        "OpB",
        kde=True,
        color="C0",
        dashed_confidence_level=68,
    )

    assert len(calls) == 1


def test_plot_contours_uses_x_for_coeff1_and_y_for_coeff2(
    posterior_samples: Posterior,
) -> None:
    """coeff1 is the x-axis, coeff2 the y-axis."""
    _, ax = plt.subplots()

    plot_contours(
        ax, posterior_samples, "OpA", "OpB", kde=False, color="C0", show_best_fit=True
    )

    marker = ax.collections[0].get_offsets()[0]
    assert marker[0] == pytest.approx(np.mean(posterior_samples["OpA"]))
    assert marker[1] == pytest.approx(np.mean(posterior_samples["OpB"]))


# ---------------------------------------------------------------------------
# fit_colors
# ---------------------------------------------------------------------------


def test_fit_colors_returns_one_color_per_fit() -> None:
    assert len(fit_colors(3)) == 3


def test_fit_colors_cycles_when_more_fits_than_colors() -> None:
    n_cycle = len(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    colors = fit_colors(n_cycle + 2)

    assert colors[n_cycle] == colors[0]
    assert colors[n_cycle + 1] == colors[1]
