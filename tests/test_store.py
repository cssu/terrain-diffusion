"""Checks the terrain store: the region grids and the tile cache.

The two are tested apart because they are meant to stay apart. The region grids are scratch space
for one region being generated, and the tile cache holds finished tiles for the life of the
program. Nothing here should need both at once.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from terrain_diffusion.store import RegionGrids, TileCache

# ---------------------------------------------------------------------------
# region grids: starting empty
# ---------------------------------------------------------------------------


def test_new_grids_are_the_size_asked_for() -> None:
    grids = RegionGrids(4, 6)

    assert grids.sums.shape == (4, 6)
    assert grids.weights.shape == (4, 6)


def test_new_grids_start_at_zero() -> None:
    grids = RegionGrids(4, 6)

    assert np.all(grids.sums == 0)
    assert np.all(grids.weights == 0)


def test_an_empty_region_is_refused() -> None:
    with pytest.raises(ValueError):
        RegionGrids(0, 5)


# ---------------------------------------------------------------------------
# region grids: adding windows
# ---------------------------------------------------------------------------


def test_a_window_lands_where_it_was_put_and_nowhere_else() -> None:
    grids = RegionGrids(4, 4)

    grids.add_window(np.ones((2, 2)), np.ones((2, 2)), row=1, col=1)

    assert np.all(grids.sums[1:3, 1:3] == 1)
    assert np.all(grids.weights[1:3, 1:3] == 1)

    # everything outside the window is untouched
    assert grids.sums.sum() == 4
    assert grids.weights.sum() == 4


def test_adding_the_same_window_twice_doubles_it() -> None:
    """Contributions accumulate. The second window must not replace the first."""
    grids = RegionGrids(4, 4)
    values = np.full((2, 2), 3.0)
    weights = np.full((2, 2), 0.5)

    grids.add_window(values, weights, row=0, col=0)
    grids.add_window(values, weights, row=0, col=0)

    assert_allclose(grids.sums[0:2, 0:2], 3.0)
    assert_allclose(grids.weights[0:2, 0:2], 1.0)


def test_overlapping_windows_add_together_in_the_overlap() -> None:
    grids = RegionGrids(1, 6)
    values = np.full((1, 4), 1.0)
    weights = np.full((1, 4), 1.0)

    grids.add_window(values, weights, row=0, col=0)
    grids.add_window(values, weights, row=0, col=2)

    # cells 2 and 3 were reached by both windows, the rest by one
    assert_allclose(grids.weights[0], [1, 1, 2, 2, 1, 1])


@pytest.mark.parametrize(
    "row, col",
    [
        (3, 0),
        (0, 3),
        (-1, 0),
        (0, -1),
    ],
)
def test_a_window_that_hangs_off_the_edge_is_refused(row: int, col: int) -> None:
    """numpy slices clip rather than complain, so this has to be caught by hand."""
    grids = RegionGrids(4, 4)

    with pytest.raises(ValueError):
        grids.add_window(np.ones((2, 2)), np.ones((2, 2)), row=row, col=col)


def test_a_window_and_its_weights_must_be_the_same_shape() -> None:
    grids = RegionGrids(4, 4)

    with pytest.raises(ValueError):
        grids.add_window(np.ones((2, 2)), np.ones((2, 3)), row=0, col=0)


# ---------------------------------------------------------------------------
# region grids: reading heights back
# ---------------------------------------------------------------------------


def test_heights_do_not_depend_on_how_large_the_weights_were() -> None:
    """A weighted average of one value is that value, whatever the weight."""
    grids = RegionGrids(2, 2)

    grids.add_window(np.full((2, 2), 5.0), np.full((2, 2), 0.25), row=0, col=0)

    assert_allclose(grids.heights(), 5.0)


def test_the_overlap_of_two_equal_windows_does_not_read_double() -> None:
    """If the overlap reads as ten then the division by the weights is missing."""
    grids = RegionGrids(1, 6)
    values = np.full((1, 4), 5.0)
    weights = np.array([[0.25, 0.75, 0.75, 0.25]])

    grids.add_window(values, weights, row=0, col=0)
    grids.add_window(values, weights, row=0, col=2)

    assert_allclose(grids.heights(), 5.0)


def test_the_overlap_of_two_different_windows_mixes_them() -> None:
    """The worked example from the ticket: two windows of 10 and 20 over a 1x6 region."""
    grids = RegionGrids(1, 6)
    weights = np.array([[0.25, 0.75, 0.75, 0.25]])

    grids.add_window(np.full((1, 4), 10.0), weights, row=0, col=0)
    grids.add_window(np.full((1, 4), 20.0), weights, row=0, col=2)

    heights = grids.heights()

    assert_allclose(heights[0], [10, 10, 12.5, 17.5, 20, 20])
    # the overlap is a blend of the two, not one or the other
    assert 10 < heights[0][2] < 20
    assert 10 < heights[0][3] < 20


def test_the_finished_grid_is_the_size_of_the_region() -> None:
    grids = RegionGrids(3, 5)

    grids.add_window(np.ones((3, 5)), np.ones((3, 5)), row=0, col=0)

    assert grids.heights().shape == (3, 5)


# ---------------------------------------------------------------------------
# region grids: unfilled cells
# ---------------------------------------------------------------------------


def test_a_region_nothing_was_written_to_is_not_complete() -> None:
    assert RegionGrids(4, 4).is_complete() is False


def test_a_fully_covered_region_is_complete() -> None:
    grids = RegionGrids(4, 4)

    grids.add_window(np.ones((4, 4)), np.ones((4, 4)), row=0, col=0)

    assert grids.is_complete() is True


def test_a_region_with_a_gap_is_not_complete_and_cannot_be_read() -> None:
    """A weight of zero divides into nonsense rather than an error, so reading has to refuse."""
    grids = RegionGrids(1, 5)
    window = np.ones((1, 2))

    grids.add_window(window, window, row=0, col=0)
    grids.add_window(window, window, row=0, col=3)

    assert grids.is_complete() is False
    assert grids.unfilled_count() == 1
    with pytest.raises(ValueError):
        grids.heights()


# ---------------------------------------------------------------------------
# tile cache: holding tiles
# ---------------------------------------------------------------------------


def test_a_tile_comes_back_out_unchanged() -> None:
    cache = TileCache(capacity=4)
    tile = np.arange(9.0).reshape(3, 3)

    cache.put(1234, tile)

    assert np.array_equal(cache.get(1234), tile)


def test_a_tile_that_was_never_generated_is_missing() -> None:
    cache = TileCache(capacity=4)

    assert cache.get(1234) is None


def test_two_seeds_do_not_overwrite_each_other() -> None:
    cache = TileCache(capacity=4)

    cache.put(1, np.zeros((2, 2)))
    cache.put(2, np.ones((2, 2)))

    assert_allclose(cache.get(1), 0.0)
    assert_allclose(cache.get(2), 1.0)


def test_a_stored_tile_is_not_changed_by_its_generator_afterwards() -> None:
    cache = TileCache(capacity=4)
    tile = np.zeros((2, 2))

    cache.put(1, tile)
    tile[0, 0] = 99

    assert_allclose(cache.get(1), 0.0)


# ---------------------------------------------------------------------------
# tile cache: dropping tiles
# ---------------------------------------------------------------------------


def test_the_cache_never_holds_more_than_its_limit() -> None:
    cache = TileCache(capacity=3)

    for seed in range(10):
        cache.put(seed, np.zeros((2, 2)))

    assert len(cache) == 3


def test_the_tile_dropped_is_the_least_recently_used_not_the_oldest() -> None:
    """Reading tile 1 keeps it, so tile 2 goes instead. First in first out would drop 1."""
    cache = TileCache(capacity=2)
    cache.put(1, np.zeros((2, 2)))
    cache.put(2, np.zeros((2, 2)))

    cache.get(1)
    cache.put(3, np.zeros((2, 2)))

    assert 1 in cache
    assert 2 not in cache
    assert 3 in cache


def test_a_tile_read_constantly_is_never_dropped() -> None:
    cache = TileCache(capacity=2)
    cache.put(0, np.zeros((2, 2)))

    for seed in range(1, 6):
        cache.get(0)
        cache.put(seed, np.zeros((2, 2)))

    assert cache.get(0) is not None


def test_a_dropped_tile_can_be_generated_again_and_put_back() -> None:
    cache = TileCache(capacity=1)
    cache.put(1, np.zeros((2, 2)))
    cache.put(2, np.zeros((2, 2)))

    assert cache.get(1) is None

    cache.put(1, np.full((2, 2), 7.0))

    assert_allclose(cache.get(1), 7.0)


def test_a_cache_that_holds_nothing_is_refused() -> None:
    with pytest.raises(ValueError):
        TileCache(capacity=0)
