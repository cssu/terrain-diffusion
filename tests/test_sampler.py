"""
Testing for window blending sampler. 
"""

import pytest

from terrain_diffusion.sampler import window_positions


class TestSampler:
    @pytest.fixture

    # def ideal_case_test(self):
    #     "Assert every cell in the region is covered by at least one window"
    #     assert

    # def edge_test(self):
    #     "Assert no window exceeds past the region"
    #     assert

    def one_size_test(self):
        "Assert a region exactly one window in size returns one position"
        positions = window_positions(4, 4, 4, 3)
        assert positions == [(0, 0)]

    def not_even_case(self):
        "Assert a region whose size does not divide evenly by the step still covers the far edge"
        positions = window_positions(8, 8, 4, 3)
        assert positions == [
            (0, 0), (0, 3), (0, 4),
            (3, 0), (3, 3), (3, 4),
            (4, 0), (4, 3), (4, 4)]