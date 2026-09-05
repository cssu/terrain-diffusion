"""Checks the terrain store: the sum and weight grids held per tile.

The store is one object, not two. It holds a running sum grid and a running weight grid for every
tile it is keeping, and works heights out by dividing one by the other on the way out. Tiles are
named by the global seed together with their coordinate, and the least recently used one is
dropped when the store is full.

A tile is a fixed block of the world that does not overlap its neighbours, so while a request is
one tile these tests use one tile as the whole of what is being generated.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from terrain_diffusion.store import TerrainStore

SEED = 1234
OTHER_SEED = 5678


def a_store(tile_height: int = 4, tile_width: int = 4, capacity: int = 8) -> TerrainStore:
    """A store holding tiles of a given size, with room to spare unless a test says otherwise."""
    return TerrainStore(tile_height, tile_width, capacity)


# ---------------------------------------------------------------------------
# starting empty
# ---------------------------------------------------------------------------


def test_a_new_tile_has_the_size_asked_for() -> None:
    sums, weights = a_store(4, 6).grids(SEED, 0, 0)

    assert sums.shape == (4, 6)
    assert weights.shape == (4, 6)


def test_a_new_tile_starts_at_zero() -> None:
    sums, weights = a_store(4, 6).grids(SEED, 0, 0)

    assert np.all(sums == 0)
    assert np.all(weights == 0)


def test_a_store_of_empty_tiles_is_refused() -> None:
    with pytest.raises(ValueError):
        TerrainStore(0, 5, capacity=4)


def test_a_store_that_holds_nothing_is_refused() -> None:
    with pytest.raises(ValueError):
        TerrainStore(4, 4, capacity=0)


# ---------------------------------------------------------------------------
# adding windows
# ---------------------------------------------------------------------------


def test_a_window_lands_where_it_was_put_and_nowhere_else() -> None:
    store = a_store()

    store.add_window(SEED, 0, 0, np.ones((2, 2)), np.ones((2, 2)), row=1, col=1)

    sums, weights = store.grids(SEED, 0, 0)
    assert np.all(sums[1:3, 1:3] == 1)
    assert np.all(weights[1:3, 1:3] == 1)
    # everything outside the window is untouched
    assert sums.sum() == 4
    assert weights.sum() == 4


def test_adding_the_same_window_twice_doubles_it() -> None:
    """Contributions accumulate. The second window must not replace the first."""
    store = a_store()
    values = np.full((2, 2), 3.0)
    weights = np.full((2, 2), 0.5)

    store.add_window(SEED, 0, 0, values, weights, row=0, col=0)
    store.add_window(SEED, 0, 0, values, weights, row=0, col=0)

    sums, tile_weights = store.grids(SEED, 0, 0)
    assert_allclose(sums[0:2, 0:2], 3.0)
    assert_allclose(tile_weights[0:2, 0:2], 1.0)


def test_overlapping_windows_add_together_in_the_overlap() -> None:
    store = a_store(1, 6)
    values = np.full((1, 4), 1.0)
    weights = np.full((1, 4), 1.0)

    store.add_window(SEED, 0, 0, values, weights, row=0, col=0)
    store.add_window(SEED, 0, 0, values, weights, row=0, col=2)

    # cells 2 and 3 were reached by both windows, the rest by one
    assert_allclose(store.grids(SEED, 0, 0)[1][0], [1, 1, 2, 2, 1, 1])


@pytest.mark.parametrize(
    "row, col",
    [
        (3, 0),
        (0, 3),
        (-1, 0),
        (0, -1),
    ],
)
def test_a_window_that_crosses_the_edge_of_a_tile_is_refused(row: int, col: int) -> None:
    """Splitting a window across tiles is not built yet, and numpy slices clip rather than complain."""
    store = a_store()

    with pytest.raises(ValueError):
        store.add_window(SEED, 0, 0, np.ones((2, 2)), np.ones((2, 2)), row=row, col=col)


def test_a_window_and_its_weights_must_be_the_same_shape() -> None:
    store = a_store()

    with pytest.raises(ValueError):
        store.add_window(SEED, 0, 0, np.ones((2, 2)), np.ones((2, 3)), row=0, col=0)


# ---------------------------------------------------------------------------
# reading heights back
# ---------------------------------------------------------------------------


def test_heights_do_not_depend_on_how_large_the_weights_were() -> None:
    """A weighted average of one value is that value, whatever the weight."""
    store = a_store(2, 2)

    store.add_window(SEED, 0, 0, np.full((2, 2), 5.0), np.full((2, 2), 0.25), row=0, col=0)

    assert_allclose(store.heights(SEED, 0, 0), 5.0)


def test_the_overlap_of_two_equal_windows_does_not_read_double() -> None:
    """If the overlap reads as ten then the division by the weights is missing."""
    store = a_store(1, 6)
    values = np.full((1, 4), 5.0)
    weights = np.array([[0.25, 0.75, 0.75, 0.25]])

    store.add_window(SEED, 0, 0, values, weights, row=0, col=0)
    store.add_window(SEED, 0, 0, values, weights, row=0, col=2)

    assert_allclose(store.heights(SEED, 0, 0), 5.0)


def test_the_overlap_of_two_different_windows_mixes_them() -> None:
    """The worked example from the ticket: two windows of 10 and 20 over a 1x6 tile."""
    store = a_store(1, 6)
    weights = np.array([[0.25, 0.75, 0.75, 0.25]])

    store.add_window(SEED, 0, 0, np.full((1, 4), 10.0), weights, row=0, col=0)
    store.add_window(SEED, 0, 0, np.full((1, 4), 20.0), weights, row=0, col=2)

    heights = store.heights(SEED, 0, 0)

    assert_allclose(heights[0], [10, 10, 12.5, 17.5, 20, 20])
    # the overlap is a blend of the two, not one or the other
    assert 10 < heights[0][2] < 20
    assert 10 < heights[0][3] < 20


def test_the_finished_grid_is_the_size_of_the_tile() -> None:
    store = a_store(3, 5)

    store.add_window(SEED, 0, 0, np.ones((3, 5)), np.ones((3, 5)), row=0, col=0)

    assert store.heights(SEED, 0, 0).shape == (3, 5)


def test_reading_a_tile_twice_does_not_blend_it_twice() -> None:
    """What is kept is the sums and the weights, so the division does not accumulate."""
    store = a_store(2, 2)

    store.add_window(SEED, 0, 0, np.full((2, 2), 5.0), np.full((2, 2), 0.5), row=0, col=0)

    assert_allclose(store.heights(SEED, 0, 0), store.heights(SEED, 0, 0))


# ---------------------------------------------------------------------------
# unfilled cells
# ---------------------------------------------------------------------------


def test_a_tile_nothing_was_written_to_is_not_complete() -> None:
    assert a_store().is_complete(SEED, 0, 0) is False


def test_a_fully_covered_tile_is_complete() -> None:
    store = a_store()

    store.add_window(SEED, 0, 0, np.ones((4, 4)), np.ones((4, 4)), row=0, col=0)

    assert store.is_complete(SEED, 0, 0) is True


def test_a_tile_with_a_gap_is_not_complete_and_cannot_be_read() -> None:
    """A weight of zero divides into nonsense rather than an error, so reading has to refuse."""
    store = a_store(1, 5)
    window = np.ones((1, 2))

    store.add_window(SEED, 0, 0, window, window, row=0, col=0)
    store.add_window(SEED, 0, 0, window, window, row=0, col=3)

    assert store.is_complete(SEED, 0, 0) is False
    assert store.unfilled_count(SEED, 0, 0) == 1
    with pytest.raises(ValueError):
        store.heights(SEED, 0, 0)


# ---------------------------------------------------------------------------
# naming a tile by its seed and coordinate
# ---------------------------------------------------------------------------


def fill(store: TerrainStore, seed: int, x: int, y: int, value: float) -> None:
    """Cover a whole tile with one value, so it can be read back."""
    ones = np.ones((store.tile_height, store.tile_width))
    store.add_window(seed, x, y, np.full_like(ones, value), ones, row=0, col=0)


def test_a_tile_comes_back_out_under_the_same_seed_and_coordinate() -> None:
    store = a_store(2, 2)

    fill(store, SEED, 3, 7, 5.0)

    assert_allclose(store.heights(SEED, 3, 7), 5.0)


def test_a_tile_that_was_never_generated_is_missing() -> None:
    assert a_store().heights(SEED, 0, 0) is None


def test_two_coordinates_under_one_seed_are_two_tiles() -> None:
    store = a_store(2, 2)

    fill(store, SEED, 0, 0, 1.0)
    fill(store, SEED, 0, 1, 2.0)

    assert_allclose(store.heights(SEED, 0, 0), 1.0)
    assert_allclose(store.heights(SEED, 0, 1), 2.0)


def test_the_same_coordinate_under_two_seeds_is_two_tiles() -> None:
    """Two seeds are two worlds and must never share a tile."""
    store = a_store(2, 2)

    fill(store, SEED, 4, 4, 1.0)
    fill(store, OTHER_SEED, 4, 4, 2.0)

    assert_allclose(store.heights(SEED, 4, 4), 1.0)
    assert_allclose(store.heights(OTHER_SEED, 4, 4), 2.0)


def test_tiles_from_two_worlds_are_held_at_the_same_time() -> None:
    store = a_store(2, 2)

    fill(store, SEED, 0, 0, 1.0)
    fill(store, OTHER_SEED, 0, 0, 2.0)

    assert len(store) == 2
    assert (SEED, 0, 0) in store
    assert (OTHER_SEED, 0, 0) in store


def test_the_store_can_say_where_the_tiles_it_holds_sit() -> None:
    store = a_store(2, 2)

    fill(store, SEED, 0, 0, 1.0)
    fill(store, SEED, 9, 9, 1.0)

    assert set(store.held_tiles()) == {(SEED, 0, 0), (SEED, 9, 9)}


# ---------------------------------------------------------------------------
# dropping tiles
# ---------------------------------------------------------------------------


def test_the_store_never_holds_more_tiles_than_its_limit() -> None:
    store = a_store(2, 2, capacity=3)

    for x in range(10):
        fill(store, SEED, x, 0, 1.0)

    assert len(store) == 3


def test_the_tile_dropped_is_the_least_recently_used_not_the_oldest() -> None:
    """Reading tile 0 keeps it, so tile 1 goes instead. First in first out would drop 0."""
    store = a_store(2, 2, capacity=2)
    fill(store, SEED, 0, 0, 1.0)
    fill(store, SEED, 1, 0, 1.0)

    store.heights(SEED, 0, 0)
    fill(store, SEED, 2, 0, 1.0)

    assert (SEED, 0, 0) in store
    assert (SEED, 1, 0) not in store
    assert (SEED, 2, 0) in store


def test_a_tile_read_constantly_is_never_dropped() -> None:
    store = a_store(2, 2, capacity=2)
    fill(store, SEED, 0, 0, 1.0)

    for x in range(1, 6):
        store.heights(SEED, 0, 0)
        fill(store, SEED, x, 0, 1.0)

    assert store.heights(SEED, 0, 0) is not None


def test_a_dropped_tile_can_be_generated_again_and_put_back() -> None:
    store = a_store(2, 2, capacity=1)
    fill(store, SEED, 0, 0, 1.0)
    fill(store, SEED, 1, 0, 1.0)

    assert store.heights(SEED, 0, 0) is None

    fill(store, SEED, 0, 0, 7.0)

    assert_allclose(store.heights(SEED, 0, 0), 7.0)


def test_a_dropped_tile_starts_from_zero_when_it_comes_back() -> None:
    """Nothing of the old tile survives, so its windows are not counted a second time."""
    store = a_store(2, 2, capacity=1)
    fill(store, SEED, 0, 0, 5.0)
    fill(store, SEED, 1, 0, 5.0)

    sums, weights = store.grids(SEED, 0, 0)

    assert np.all(sums == 0)
    assert np.all(weights == 0)


def test_a_window_that_is_not_a_2d_grid_is_refused() -> None:
    store = a_store()

    with pytest.raises(ValueError):
        store.add_window(SEED, 0, 0, np.ones(4), np.ones(4), row=0, col=0)


def test_a_tile_that_is_not_held_counts_as_entirely_unfilled() -> None:
    assert a_store(2, 3).unfilled_count(SEED, 0, 0) == 6
