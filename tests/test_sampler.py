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
        step = 1
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
    def pipeline(self, mocker):
        pipeline = mocker.Mock()  # make it a Mock object, this way can count calls.
        pipeline.generate.side_effect = lambda patch: np.full(
            patch.shape, 5
        )  # added side_effect to keep it a Mock object
        return pipeline

    def test_all_fives(self, pipeline):
        """Assert the finished grid is all fives everywhere, including the overlaps and the corners.
        If the overlaps read higher then the weights are not being divided out"""

        seed = 123
        height = 8
        width = 8
        window_size = 4
        step = 2

        weighted_sum, weight_sum = produce_region(seed, height, width, window_size, step, pipeline)
        result = weighted_sum / weight_sum  # doing job of store
        assert np.allclose(
            result, 5
        )  # All close because was getting float error as some are 4.9999 due to the store

    def test_full_size(self, pipeline):
        """Assert the finished grid is the region's full resolution size"""

        seed = 123
        height = 8
        width = 8
        window_size = 4
        step = 2

        weighted_sum, weight_sum = produce_region(seed, height, width, window_size, step, pipeline)
        result = weighted_sum / weight_sum  # doing job of store

        assert result.shape == (height, width)

    def test_once_per_window(self, pipeline):
        """Assert the fake pipeline was called once per window position and no more"""

        seed = 123
        height = 8
        width = 8
        window_size = 4
        step = 2

        positions = window_positions(height, width, window_size, step)

        produce_region(seed, height, width, window_size, step, pipeline)

        assert pipeline.generate.call_count == len(positions)

    def test_same_seed_grid(self, pipeline):
        """Assert the same seed and region run twice give identical grids"""
        seed = 123
        height = 8
        width = 8
        window_size = 4
        step = 2

        weighted_sum1, weight_sum1 = produce_region(
            seed, height, width, window_size, step, pipeline
        )
        weighted_sum2, weight_sum2 = produce_region(
            seed, height, width, window_size, step, pipeline
        )
        result1 = weighted_sum1 / weight_sum1
        result2 = weighted_sum2 / weight_sum2

        assert np.array_equal(result1, result2)
