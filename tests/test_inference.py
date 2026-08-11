"""
Tests inference interface by instantiating mock models
and implementing the `generate` function
"""

import numpy as np
import pytest

from terrain_diffusion.inference import TerrainModel


class MockTerrainModel(TerrainModel):
    """
    Mock model implementing the TerrainModel abstract class
    """

    def generate(self, patch: np.ndarray) -> np.ndarray:
        """
        Generate an output patch (C, H, W) from the input patch
        by multiplying by 2
        """
        return patch * 2

    @classmethod
    def load_model(cls, model_path):
        print(f"loaded model: {model_path}")
        return MockTerrainModel()

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        return self.generate(patch)


class TestTerrainModel:
    @pytest.fixture
    def initial_model(self) -> MockTerrainModel:
        return MockTerrainModel()

    def test_load(self):
        assert isinstance(MockTerrainModel.load_model("my_model"), TerrainModel)

    def test_generate(self, initial_model):
        input_patch = np.ones((3, 5, 10))  # 10x5 image with 3 channels

        actual = initial_model(input_patch)
        expected = 2 * input_patch

        assert np.array_equal(actual, expected), "patches are not equal"
