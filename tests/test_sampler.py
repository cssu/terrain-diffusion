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


# the requested test for determinitic grid has been added at line 205
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

    def test_deterministic_region(self, pipeline):
        """Assert the output matches a deterministic 9x9 grid."""
        seed = 123
        height = 9
        width = 9
        window_size = 3
        step = 2

        pipeline.generate.side_effect = lambda patch: patch

        weighted_sum, weight_sum = produce_region(seed, height, width, window_size, step, pipeline)

        result = weighted_sum / weight_sum

        expected = np.array(
            [
                [
                    0.68235186,
                    0.05382102,
                    0.22035987,
                    0.18437181,
                    0.1759059,
                    0.81209451,
                    0.923345,
                    0.2765744,
                    0.81975456,
                ],
                [
                    0.88989269,
                    0.51297046,
                    0.2449646,
                    0.8242416,
                    0.21376296,
                    0.74146705,
                    0.6299402,
                    0.92740726,
                    0.23190819,
                ],
                [
                    0.79912513,
                    0.51816504,
                    0.23155562,
                    0.16590399,
                    0.49778897,
                    0.58272464,
                    0.18433799,
                    0.01489492,
                    0.47113323,
                ],
                [
                    0.72824333,
                    0.91860049,
                    0.62553401,
                    0.91712257,
                    0.86469025,
                    0.21814287,
                    0.86612743,
                    0.73075194,
                    0.27786529,
                ],
                [
                    0.79704355,
                    0.86522171,
                    0.2994379,
                    0.52704208,
                    0.07148681,
                    0.58323841,
                    0.2379064,
                    0.76496365,
                    0.17363164,
                ],
                [
                    0.31274226,
                    0.01447448,
                    0.03255192,
                    0.49670184,
                    0.46831253,
                    0.12769032,
                    0.2575625,
                    0.00318111,
                    0.38106775,
                ],
                [
                    0.57587308,
                    0.42729877,
                    0.83510235,
                    0.61649125,
                    0.26608391,
                    0.81102211,
                    0.49948675,
                    0.75881032,
                    0.56608909,
                ],
                [
                    0.43744036,
                    0.39615444,
                    0.02223529,
                    0.46935079,
                    0.6235584,
                    0.94611342,
                    0.43532608,
                    0.4856414,
                    0.51911514,
                ],
                [
                    0.40859098,
                    0.57879572,
                    0.07035067,
                    0.48838383,
                    0.61014483,
                    0.74387911,
                    0.42983032,
                    0.30280213,
                    0.00589003,
                ],
            ]
        )

        assert np.allclose(result, expected)
