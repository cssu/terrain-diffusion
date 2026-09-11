import numpy as np
import pytest
from numpy.testing import assert_allclose

from terrain_diffusion.store import TerrainStore

SEED = 1234
OTHER_SEED = 5678


@pytest.fixture
def store() -> TerrainStore:
    return TerrainStore(tile_height=4, tile_width=4, capacity=8)


@pytest.fixture
def make_store():
    def build(tile_height: int = 4, tile_width: int = 4, capacity: int = 8) -> TerrainStore:
        return TerrainStore(tile_height, tile_width, capacity)

    return build


@pytest.fixture
def fill_tile():
    def fill(store: TerrainStore, seed: int, x: int, y: int, value: float) -> None:
        weights = np.ones((store.tile_height, store.tile_width))
        store.add_window(
            seed,
            np.full_like(weights, value),
            weights,
            row=y * store.tile_height,
            col=x * store.tile_width,
        )

    return fill


def test_a_new_tile_has_the_size_asked_for(make_store) -> None:
    sums, weights = make_store(4, 6).grids(SEED, 0, 0)

    assert sums.shape == (4, 6)
    assert weights.shape == (4, 6)


def test_a_new_tile_starts_at_zero(store) -> None:
    sums, weights = store.grids(SEED, 0, 0)

    assert np.all(sums == 0)
    assert np.all(weights == 0)


def test_a_store_of_empty_tiles_is_refused() -> None:
    with pytest.raises(ValueError):
        TerrainStore(0, 5, capacity=4)


def test_a_store_that_holds_nothing_is_refused() -> None:
    with pytest.raises(ValueError):
        TerrainStore(4, 4, capacity=0)


def test_a_window_lands_where_it_was_put_and_nowhere_else(store) -> None:
    store.add_window(SEED, np.ones((2, 2)), np.ones((2, 2)), row=1, col=1)

    sums, weights = store.grids(SEED, 0, 0)
    assert np.all(sums[1:3, 1:3] == 1)
    assert np.all(weights[1:3, 1:3] == 1)
    assert sums.sum() == 4
    assert weights.sum() == 4


def test_overlapping_windows_add_together_in_the_overlap(make_store) -> None:
    store = make_store(1, 6)
    values = np.full((1, 4), 1.0)
    weights = np.full((1, 4), 1.0)

    store.add_window(SEED, values, weights, row=0, col=0)
    store.add_window(SEED, values, weights, row=0, col=2)

    assert_allclose(store.grids(SEED, 0, 0)[1][0], [1, 1, 2, 2, 1, 1])


def test_a_window_and_its_weights_must_be_the_same_shape(store) -> None:
    with pytest.raises(ValueError):
        store.add_window(SEED, np.ones((2, 2)), np.ones((2, 3)), row=0, col=0)


def test_a_window_that_is_not_a_2d_grid_is_refused(store) -> None:
    with pytest.raises(ValueError):
        store.add_window(SEED, np.ones(4), np.ones(4), row=0, col=0)


def test_a_window_crossing_a_side_is_split_between_the_two_tiles(make_store) -> None:
    store = make_store(2, 2)

    store.add_window(SEED, np.full((2, 2), 5.0), np.ones((2, 2)), row=0, col=1)

    assert_allclose(store.grids(SEED, 0, 0)[1], [[0, 1], [0, 1]])
    assert_allclose(store.grids(SEED, 1, 0)[1], [[1, 0], [1, 0]])
    assert_allclose(store.grids(SEED, 0, 0)[0], [[0, 5], [0, 5]])
    assert_allclose(store.grids(SEED, 1, 0)[0], [[5, 0], [5, 0]])


def test_a_window_crossing_a_corner_is_split_between_four_tiles(make_store) -> None:
    store = make_store(2, 2)

    store.add_window(SEED, np.full((2, 2), 5.0), np.ones((2, 2)), row=1, col=1)

    assert_allclose(store.grids(SEED, 0, 0)[1], [[0, 0], [0, 1]])
    assert_allclose(store.grids(SEED, 1, 0)[1], [[0, 0], [1, 0]])
    assert_allclose(store.grids(SEED, 0, 1)[1], [[0, 1], [0, 0]])
    assert_allclose(store.grids(SEED, 1, 1)[1], [[1, 0], [0, 0]])


def test_a_window_at_a_negative_position_lands_in_the_tiles_before_the_origin(make_store) -> None:
    store = make_store(2, 2)

    store.add_window(SEED, np.full((2, 2), 5.0), np.ones((2, 2)), row=-1, col=-1)

    assert_allclose(store.grids(SEED, -1, -1)[1], [[0, 0], [0, 1]])
    assert_allclose(store.grids(SEED, 0, 0)[1], [[1, 0], [0, 0]])


def test_a_split_window_blends_with_its_neighbours_the_same_as_an_unsplit_one(make_store) -> None:
    store = make_store(2, 4)
    values = np.full((2, 4), 5.0)
    weights = np.array([[0.25, 0.75, 0.75, 0.25], [0.25, 0.75, 0.75, 0.25]])

    store.add_window(SEED, values, weights, row=0, col=0)
    store.add_window(SEED, values, weights, row=0, col=2)
    store.add_window(SEED, values, weights, row=0, col=4)

    assert_allclose(store.heights(SEED, 0, 0), 5.0)


def test_a_window_covering_more_tiles_than_the_store_holds_is_refused(make_store) -> None:
    store = make_store(2, 2, capacity=2)

    with pytest.raises(ValueError):
        store.add_window(SEED, np.ones((4, 4)), np.ones((4, 4)), row=0, col=0)


def test_heights_do_not_depend_on_how_large_the_weights_were(make_store) -> None:
    store = make_store(2, 2)

    store.add_window(SEED, np.full((2, 2), 5.0), np.full((2, 2), 0.25), row=0, col=0)

    assert_allclose(store.heights(SEED, 0, 0), 5.0)


def test_the_overlap_of_two_equal_windows_does_not_read_double(make_store) -> None:
    store = make_store(1, 6)
    values = np.full((1, 4), 5.0)
    weights = np.array([[0.25, 0.75, 0.75, 0.25]])

    store.add_window(SEED, values, weights, row=0, col=0)
    store.add_window(SEED, values, weights, row=0, col=2)

    assert_allclose(store.heights(SEED, 0, 0), 5.0)


def test_the_overlap_of_two_different_windows_mixes_them(make_store) -> None:
    store = make_store(1, 6)
    weights = np.array([[0.25, 0.75, 0.75, 0.25]])

    store.add_window(SEED, np.full((1, 4), 10.0), weights, row=0, col=0)
    store.add_window(SEED, np.full((1, 4), 20.0), weights, row=0, col=2)

    heights = store.heights(SEED, 0, 0)

    assert_allclose(heights[0], [10, 10, 12.5, 17.5, 20, 20])
    assert 10 < heights[0][2] < 20
    assert 10 < heights[0][3] < 20


def test_the_finished_grid_is_the_size_of_the_tile(make_store) -> None:
    store = make_store(3, 5)

    store.add_window(SEED, np.ones((3, 5)), np.ones((3, 5)), row=0, col=0)

    assert store.heights(SEED, 0, 0).shape == (3, 5)


def test_reading_a_tile_twice_does_not_blend_it_twice(make_store) -> None:
    store = make_store(2, 2)

    store.add_window(SEED, np.full((2, 2), 5.0), np.full((2, 2), 0.5), row=0, col=0)

    assert_allclose(store.heights(SEED, 0, 0), store.heights(SEED, 0, 0))


def test_a_tile_nothing_was_written_to_is_not_complete(store) -> None:
    assert store.is_complete(SEED, 0, 0) is False


def test_a_fully_covered_tile_is_complete(store) -> None:
    store.add_window(SEED, np.ones((4, 4)), np.ones((4, 4)), row=0, col=0)

    assert store.is_complete(SEED, 0, 0) is True


def test_a_tile_with_a_gap_is_not_complete_and_cannot_be_read(make_store) -> None:
    store = make_store(1, 5)
    window = np.ones((1, 2))

    store.add_window(SEED, window, window, row=0, col=0)
    store.add_window(SEED, window, window, row=0, col=3)

    assert store.is_complete(SEED, 0, 0) is False
    assert store.unfilled_count(SEED, 0, 0) == 1
    with pytest.raises(ValueError):
        store.heights(SEED, 0, 0)


def test_a_tile_that_is_not_held_counts_as_entirely_unfilled(make_store) -> None:
    assert make_store(2, 3).unfilled_count(SEED, 0, 0) == 6


def test_a_tile_comes_back_out_under_the_same_seed_and_coordinate(make_store, fill_tile) -> None:
    store = make_store(2, 2)

    fill_tile(store, SEED, 3, 7, 5.0)

    assert_allclose(store.heights(SEED, 3, 7), 5.0)


def test_a_tile_that_was_never_generated_is_missing(store) -> None:
    assert store.heights(SEED, 0, 0) is None


def test_two_coordinates_under_one_seed_are_two_tiles(make_store, fill_tile) -> None:
    store = make_store(2, 2)

    fill_tile(store, SEED, 0, 0, 1.0)
    fill_tile(store, SEED, 0, 1, 2.0)

    assert_allclose(store.heights(SEED, 0, 0), 1.0)
    assert_allclose(store.heights(SEED, 0, 1), 2.0)


def test_the_same_coordinate_under_two_seeds_is_two_tiles(make_store, fill_tile) -> None:
    store = make_store(2, 2)

    fill_tile(store, SEED, 4, 4, 1.0)
    fill_tile(store, OTHER_SEED, 4, 4, 2.0)

    assert_allclose(store.heights(SEED, 4, 4), 1.0)
    assert_allclose(store.heights(OTHER_SEED, 4, 4), 2.0)


def test_tiles_from_two_worlds_are_held_at_the_same_time(make_store, fill_tile) -> None:
    store = make_store(2, 2)

    fill_tile(store, SEED, 0, 0, 1.0)
    fill_tile(store, OTHER_SEED, 0, 0, 2.0)

    assert len(store) == 2
    assert (SEED, 0, 0) in store
    assert (OTHER_SEED, 0, 0) in store


def test_the_store_can_say_where_the_tiles_it_holds_sit(make_store, fill_tile) -> None:
    store = make_store(2, 2)

    fill_tile(store, SEED, 0, 0, 1.0)
    fill_tile(store, SEED, 9, 9, 1.0)

    assert set(store.held_tiles()) == {(SEED, 0, 0), (SEED, 9, 9)}


def test_the_store_never_holds_more_tiles_than_its_limit(make_store, fill_tile) -> None:
    store = make_store(2, 2, capacity=3)

    for x in range(10):
        fill_tile(store, SEED, x, 0, 1.0)

    assert len(store) == 3


def test_the_tile_dropped_is_the_least_recently_used_not_the_oldest(make_store, fill_tile) -> None:
    store = make_store(2, 2, capacity=2)
    fill_tile(store, SEED, 0, 0, 1.0)
    fill_tile(store, SEED, 1, 0, 1.0)

    store.heights(SEED, 0, 0)
    fill_tile(store, SEED, 2, 0, 1.0)

    assert (SEED, 0, 0) in store
    assert (SEED, 1, 0) not in store
    assert (SEED, 2, 0) in store


def test_a_tile_read_constantly_is_never_dropped(make_store, fill_tile) -> None:
    store = make_store(2, 2, capacity=2)
    fill_tile(store, SEED, 0, 0, 1.0)

    for x in range(1, 6):
        store.heights(SEED, 0, 0)
        fill_tile(store, SEED, x, 0, 1.0)

    assert store.heights(SEED, 0, 0) is not None


def test_a_dropped_tile_can_be_generated_again_and_put_back(make_store, fill_tile) -> None:
    store = make_store(2, 2, capacity=1)
    fill_tile(store, SEED, 0, 0, 1.0)
    fill_tile(store, SEED, 1, 0, 1.0)

    assert store.heights(SEED, 0, 0) is None

    fill_tile(store, SEED, 0, 0, 7.0)

    assert_allclose(store.heights(SEED, 0, 0), 7.0)


def test_a_dropped_tile_starts_from_zero_when_it_comes_back(make_store, fill_tile) -> None:
    store = make_store(2, 2, capacity=1)
    fill_tile(store, SEED, 0, 0, 5.0)
    fill_tile(store, SEED, 1, 0, 5.0)

    sums, weights = store.grids(SEED, 0, 0)

    assert np.all(sums == 0)
    assert np.all(weights == 0)
