"""
Testing for window blending sampler.
"""

import numpy as np
import pytest

from terrain_diffusion.sampler import (
    generate_noise_from_seed,
    produce_region,
    weight_grid,
    window_positions,
)


class TestWindowPositions:
    def test_all_covered(self):
        "Assert every cell in the region is covered by at least one window"
        height = 4
        width = 4
        window = 2
        step = 2
        positions = window_positions(height, width, window, step)
        assert all(
            any(
                window_r <= row < window_r + window and window_c <= column < window_c + window
                for window_r, window_c in positions
            )
            for row in range(height)
            for column in range(width)
        )

    def test_exceed_region(self):
        "Assert no window exceeds past the region"
        height = 4
        width = 4
        size = 2
        step = 1
        positions = window_positions(height, width, size, step)
        assert all(x[0] + size <= height and x[1] + size <= width for x in positions)

    def test_one_window(self):
        "Assert a region exactly one window in size returns one position"
        WindowRegionSize = 4
        positions = window_positions(
            WindowRegionSize, WindowRegionSize, WindowRegionSize, WindowRegionSize
        )
        assert len(positions) == 1

    # Had to modify test because added assertion to original function
    def test_region_not_divisible(self):
        with pytest.raises(AssertionError):
            window_positions(10, 8, 4, 3)


class TestWeights:
    def test_grid_equal_patch(self):
        "Assert the grid is the size of a patch"
        edge = 10
        weights = weight_grid(edge)
        assert weights.shape == (edge, edge)

    def test_palidrome(self):
        "Assert it reads the same forwards and backwards in both directions"
        edge = 10
        weights = weight_grid(edge)
        assert np.array_equal(weights, weights[::-1, :])  # vertical
        assert np.array_equal(weights, weights[:, ::-1])  # horizontal

    def test_large_middle(self):
        "Assert the largest value is in the middle"
        edge = 11
        weights = weight_grid(edge)

        # middle
        center = edge // 2

        assert weights[center, center] == weights.max()

    def test_edges_smaller(self):
        "Assert values at the edges are smaller than values in the middle"
        edge = 11
        weights = weight_grid(edge)

        # middle
        center = edge // 2
        middle_val = weights[center, center]

        # R/L edges
        assert all(weights[x, 0] < middle_val for x in range(edge))
        assert all(weights[x, edge - 1] < middle_val for x in range(edge))

        # T/B edges
        assert all(weights[0, y] < middle_val for y in range(edge))
        assert all(weights[edge - 1, y] < middle_val for y in range(edge))

    def test_greater_zero(self):
        "Assert every value is greater than zero"
        weights = weight_grid(10)
        assert np.all(weights > 0)


class TestSeed:
    def test_same_seed(self):
        """Assert the same seed twice gives identical grids."""
        seed = 123
        height = 10
        width = 20
        noise1 = generate_noise_from_seed(seed, height, width)
        noise2 = generate_noise_from_seed(seed, height, width)
        assert np.array_equal(noise1, noise2)

    def test_diff_seed(self):
        """Assert two different seeds give different grids."""
        seed1 = 123
        seed2 = 456
        height = 10
        width = 20
        noise1 = generate_noise_from_seed(seed1, height, width)
        noise2 = generate_noise_from_seed(seed2, height, width)
        assert not np.array_equal(noise1, noise2)

    def test_right_size(self):
        """Assert the grid is the size asked for"""
        seed = 123
        height = 10
        width = 20
        noise = generate_noise_from_seed(seed, height, width)
        assert noise.shape == (height, width)



class TestRegionProduction:
    @pytest.fixture
    def clean_patch(self, mocker):
        clean = mocker.Mock()
        clean.side_effect = lambda patch: np.full(patch.shape, 5.0)
        return clean

    @pytest.fixture
    def store(self):
        from terrain_diffusion.store import TerrainStore 
        return TerrainStore(tile_height=8, tile_width=8, capacity=4)

    def test_writes_windows_to_store(self, clean_patch, store):
        seed = 123
        height = 8
        width = 8
        window_size = 4
        step = 2

        positions = window_positions(height, width, window_size, step)

        produce_region(
            seed, height, width, window_size, step,
            clean_patch, store, 0, 0,
        )

        assert clean_patch.call_count == len(positions)
        assert store.is_complete(seed, 0, 0)

        result = store.heights(seed, 0, 0)
        assert np.allclose(result, 5.0)

    def test_writes_windows_at_global_offset(self, clean_patch, store):
        produce_region(
            123, 4, 4, 2, 2,
            clean_patch, store, 2, 2,
        )

        assert store.is_complete(123, 0, 0)

        result = store.heights(123, 0, 0)
        assert np.allclose(result[2:6, 2:6], 5.0)

    def test_same_seed_same_region(self, clean_patch):
        from terrain_diffusion.store import TerrainStore  # adjust import if needed

        store1 = TerrainStore(tile_height=8, tile_width=8, capacity=4)
        store2 = TerrainStore(tile_height=8, tile_width=8, capacity=4)

        produce_region(123, 8, 8, 4, 2, clean_patch, store1, 0, 0)
        produce_region(123, 8, 8, 4, 2, clean_patch, store2, 0, 0)

        result1 = store1.heights(123, 0, 0)
        result2 = store2.heights(123, 0, 0)

        assert np.array_equal(result1, result2)

    def test_deterministic_region(self, mocker):
        from terrain_diffusion.store import TerrainStore  # adjust to your actual import path

        clean_patch = mocker.Mock()
        clean_patch.side_effect = lambda patch: np.ones(patch.shape)

        store = TerrainStore(tile_height=8, tile_width=8, capacity=4)

        produce_region(
            seed=123,
            height=8,
            width=8,
            window_size=4,
            step=2,
            clean_patch=clean_patch,
            store=store,
            global_row=0,
            global_col=0,
        )

        result = store.heights(123, 0, 0)
        assert np.allclose(result, 1.0)

