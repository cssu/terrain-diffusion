"""
Testing for window blending sampler. 
"""

from terrain_diffusion.sampler import window_positions, weight_grid
import numpy as np


class TestWindowPositions:

    def test_all_covered(self):
        "Assert every cell in the region is covered by at least one window"
        height = 4
        width = 4
        window = 2
        step = 1
        positions = window_positions(height, width, window, step)
        assert all(any(window_r <= row < window_r + window and window_c <= column < window_c + window 
                       for window_r, window_c in positions) 
                       for row in range(height) for column in range(width))

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
        positions = window_positions(WindowRegionSize, WindowRegionSize, WindowRegionSize, WindowRegionSize)
        assert len(positions) == 1

    def test_not_dividing(self):
        "Assert a region whose size does not divide evenly by the step still covers the far edge"
        positions = window_positions(8, 8, 4, 3)
        assert positions == [
            (0, 0), (0, 3), (0, 4),
            (3, 0), (3, 3), (3, 4),
            (4, 0), (4, 3), (4, 4)]



class TestWeights:

    def test_grid_equal_patch(self):
        "Assert the grid is the size of a patch"
        height = 10
        width = 20
        weights = weight_grid(height, width)
        assert weights.shape == (height, width)

    def test_palidrome(self):
        "Assert it reads the same forwards and backwards in both directions"
        height = 10
        width = 20
        weights = weight_grid(height, width)
        assert np.array_equal(weights, weights[::-1])
        # assert vertically

    def test_large_middle(self):
        "Assert the largest value is in the middle"
        height = 10
        width = 20
        weights = weight_grid(height, width)

        # middle
        center_row = height // 2
        center_column = width // 2

        assert weights[center_row, center_column] == weights.max()

    def test_edges_smaller(self):
        "Assert values at the edges are smaller than values in the middle"

    def test_greater_zero(self):
        "Assert every value is greater than zero"