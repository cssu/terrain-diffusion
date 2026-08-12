"""
Tests inference interface by instantiating mock models
and implementing the `generate` function
"""

import numpy as np
import pytest

from terrain_diffusion.inference import (
    LATENT_MAP_SIZE,
    PATCH_SIZE,
    MockCoreModel,
    MockCoreModelInput,
    MockCoreModelOutput,
    MockDecoderModel,
    MockDecoderModelInput,
    MockDecoderModelOutput,
    load_model,
)


class TestTerrainModel:
    @pytest.fixture
    def initial_decoder(self) -> MockDecoderModel:
        return MockDecoderModel()

    @pytest.fixture
    def initial_core(self) -> MockCoreModel:
        return MockCoreModel()

    def test_load_model(self):
        assert isinstance(load_model("decoder"), MockDecoderModel), (
            "model does not load correct decoder"
        )
        assert isinstance(load_model("core"), MockCoreModel), "model does not load correct core"

    def test_predict_core(self, initial_core):
        input = MockCoreModelInput(np.ones(PATCH_SIZE))

        actual = initial_core(input)
        actual_2 = initial_core(input)

        double = 2 * input.patch

        expected_low_res = np.resize(double, (PATCH_SIZE[0] // 8, PATCH_SIZE[1] // 8))
        expected_latent = np.resize(double, LATENT_MAP_SIZE)
        expected = MockCoreModelOutput(expected_low_res, expected_latent)

        assert actual == expected, "predictions are not equal"
        assert actual == actual_2, "model return different predictions on same input"
        assert actual.low_res_grid.shape == (PATCH_SIZE[0] // 8, PATCH_SIZE[1] // 8), (
            "low resolution map shape is not patch size // 8"
        )
        assert actual.latent_map.shape == LATENT_MAP_SIZE, "latent map size is not correct"

    def test_predict_decoder(self, initial_decoder):
        input = MockDecoderModelInput(np.ones(LATENT_MAP_SIZE))

        actual = initial_decoder(input)
        actual_2 = initial_decoder(input)

        double = 2 * input.latent_map

        expected_full_res = np.resize(double, PATCH_SIZE)
        expected = MockDecoderModelOutput(expected_full_res)

        assert actual == expected, "predictions are not equal"
        assert actual == actual_2, "model return different predictions on same input"
        assert actual.full_res_grid.shape == PATCH_SIZE, (
            "full resolution map does not match patch size"
        )
