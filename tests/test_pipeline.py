"""
Tests ModelPipeline generation functions
"""

import numpy as np
import pytest
from terrain_diffusion.pipeline import *
from terrain_diffusion.inference import PATCH_SIZE

class TestModelPipeline:
    @pytest.fixture
    def initial_pipeline(self) -> ModelPipeline:
        return ModelPipeline()

    @pytest.fixture
    def initial_patch(self) -> np.ndarray:
        return np.ones(PATCH_SIZE)

    def test_generate_size(self, initial_pipeline, initial_patch):
        output = initial_pipeline.generate(initial_patch)
        assert output.shape == initial_patch.shape, "input and output sizes do not match"

    def test_generate_deterministic(self, initial_pipeline, initial_patch):
        output_1 = initial_pipeline.generate(initial_patch)
        output_2 = initial_pipeline.generate(initial_patch)
        assert output_1 == output_2, "outputs are not the same" #TODO: implement __eq__ for elevation encoding output or smth