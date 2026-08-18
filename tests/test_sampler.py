"""
Testing for window blending sampler. 
"""

from terrain_diffusion.sampler import window_positions


class TestSampler:

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