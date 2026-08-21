"""
Testing for window blending sampler. 
"""

from terrain_diffusion.sampler import window_positions, weight_grid, starting_noise, produce_region
import numpy as np
import pytest


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
        assert np.array_equal(weights, weights[::-1, :]) #vertical
        assert np.array_equal(weights, weights[:, ::-1]) #horizontal

    def test_large_middle(self):
        "Assert the largest value is in the middle"
        height = 11
        width = 21
        weights = weight_grid(height, width)

        # middle
        center_row = height // 2
        center_column = width // 2

        assert weights[center_row, center_column] == weights.max()

    def test_edges_smaller(self):
        "Assert values at the edges are smaller than values in the middle"
        height = 11
        width = 21
        weights = weight_grid(height, width)

        # middle
        center_row = height // 2
        center_column = width // 2
        middle_val = weights[center_row, center_column]

        # R/L edges
        assert all(weights[x,0] < middle_val for x in range(height))
        assert all(weights[x,width-1] < middle_val for x in range(height))

        # T/B edges
        assert all(weights[0, y] < middle_val for y in range(width))
        assert all(weights[height - 1, y] < middle_val for y in range(width))

    def test_greater_zero(self):
        "Assert every value is greater than zero"
        weights = weight_grid(10, 20)
        assert np.all(weights > 0)
        


class TestSeed:

    def test_same_seed(self):
        """Assert the same seed twice gives identical grids."""
        seed = 123
        height = 10
        width = 20
        noise1 = starting_noise(seed, height, width)
        noise2 = starting_noise(seed, height, width)
        assert np.array_equal(noise1, noise2)
    
    def test_diff_seed(self):
        """Assert two different seeds give different grids."""
        seed1 = 123
        seed2 = 456
        height = 10
        width = 20
        noise1 = starting_noise(seed1, height, width)
        noise2 = starting_noise(seed2, height, width)
        assert not np.array_equal(noise1, noise2)

    
    def test_right_size(self):
        """Assert the grid is the size asked for"""
        seed = 123
        height = 10
        width = 20
        noise = starting_noise(seed, height, width)
        assert noise.shape == (height, width)


class FakePipeline:
    def __init__(self):
        """For test_once_per_window"""
        self.call_count = 0

    def generate(self, patch):
        """Igrones input and returns a patch of all fives."""
        self.call_count += 1
        return np.full(patch.shape, 5)


class FakeStore:
    #used ai help for this because store is not part of my ticket
    def __init__(self, height, width):
        self.sum_grid = np.zeros((height, width))
        self.weight_grid = np.zeros((height, width))

    def add(self, patch, row, column, weights):
        self.sum_grid[
            row:row + patch.shape[0],
            column:column + patch.shape[1]
        ] += patch * weights

        self.weight_grid[
            row:row + patch.shape[0],
            column:column + patch.shape[1]
        ] += weights

    def finish(self):
        return self.sum_grid / self.weight_grid

    
class TestRegionProduction:

    @pytest.fixture
    def pipeline(self):
        return FakePipeline()

    def test_all_fives(self, pipeline):
        """Assert the finished grid is all fives everywhere, including the overlaps and the corners. 
        If the overlaps read higher then the weights are not being divided out"""

        seed = 123
        height = 8
        width = 8
        window_size = 4
        step = 3

        store = FakeStore(height, width)
        result = produce_region(seed, height, width, window_size, step, pipeline, store)
        assert np.allclose(result, 5)  #All close because was getting float error as some are 4.9999 due to the store 


    def test_full_size(self, pipeline):
        """Assert the finished grid is the region's full resolution size"""

        seed = 123
        height = 8
        width = 8
        window_size = 4
        step = 3

        store = FakeStore(height, width)
        result = produce_region(seed, height, width, window_size, step, pipeline, store)
        assert result.shape == (height, width)


    def test_once_per_window(self, pipeline):
        """Assert the fake pipeline was called once per window position and no more"""
        
        seed = 123
        height = 8
        width = 8
        window_size = 4
        step = 3

        positions = window_positions(height, width, window_size, step)

        store = FakeStore(height, width)
        produce_region(seed, height, width, window_size, step, pipeline, store)

        assert pipeline.call_count == len(positions)


    def test_same_seed_grid(self):
        """Assert the same seed and region run twice give identical grids"""
        height = 8
        width = 8
        window_size = 4
        step = 3

        store1 = FakeStore(height, width)
        store2 = FakeStore(height, width)

        #because using same pipeline might affect results?
        pipeline1 = FakePipeline()
        pipeline2 = FakePipeline()

        result1 = produce_region(123, height, width, window_size, step, pipeline1, store1)
        result2 = produce_region(123, height, width, window_size, step, pipeline2, store2)

        assert np.array_equal(result1, result2)