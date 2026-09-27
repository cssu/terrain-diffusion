"""
Tests inference interface by instantiating mock models
and implementing the `generate` function
"""

import numpy as np
import pytest

from terrain_diffusion.inference import (
    LATENT_MAP_SIZE,
    LATENT_SIZE,
    PATCH_SIZE,
    CoreModelInput,
    CoreModelOutput,
    DecoderModelInput,
    DecoderModelOutput,
    MockCoreModel,
    MockDecoderModel,
)


class TestTerrainModel:
    @pytest.fixture
    def initial_decoder(self) -> MockDecoderModel:
        return MockDecoderModel()

    @pytest.fixture
    def initial_core(self) -> MockCoreModel:
        return MockCoreModel()

    def test_core_input_generation(self):
        with pytest.raises(AssertionError):
            CoreModelInput(np.ones((1, 1)))

    def test_core_output_generation(self):
        with pytest.raises(AssertionError):
            CoreModelOutput(np.ones((1, 1)), np.ones(LATENT_MAP_SIZE))
        with pytest.raises(AssertionError):
            CoreModelOutput(np.ones((PATCH_SIZE[0] // 8, PATCH_SIZE[1] // 8)), np.ones((1, 1)))

    def test_decoder_input_generation(self):
        with pytest.raises(AssertionError):
            DecoderModelInput(np.ones((1, 1)))

    def test_decoder_output_generation(self):
        with pytest.raises(AssertionError):
            DecoderModelOutput(np.ones((1, 1)))

    def test_predict_core(self, initial_core):
        input = CoreModelInput(np.ones(CoreModelInput.patch_shape))

        actual = initial_core.predict(input)
        actual_2 = initial_core.predict(input)

        expected_low_res = np.ndarray((LATENT_SIZE, LATENT_SIZE))
        expected_latent = np.ndarray(LATENT_MAP_SIZE)

        expected_low_res.fill(2)
        expected_latent.fill(2)

        expected = CoreModelOutput(expected_low_res, expected_latent)

        assert actual == expected, "predictions are not equal"
        assert actual == actual_2, "model return different predictions on same input"
        assert actual.low_res_grid.shape == (LATENT_SIZE, LATENT_SIZE), (
            "low resolution map shape is not the latent size"
        )
        assert actual.latent_map.shape == LATENT_MAP_SIZE, "latent map size is not correct"

    def test_predict_decoder(self, initial_decoder):
        input = DecoderModelInput(np.ones(LATENT_MAP_SIZE))

        actual = initial_decoder.predict(input)
        actual_2 = initial_decoder.predict(input)

        expected_full_res = np.ndarray(DecoderModelOutput.full_res_grid_shape)
        expected_full_res.fill(2)

        expected = DecoderModelOutput(expected_full_res)

        assert actual == expected, "predictions are not equal"
        assert actual == actual_2, "model return different predictions on same input"
        assert actual.full_res_grid.shape == DecoderModelOutput.full_res_grid_shape, (
            "decoder output shape does not match its declared shape"
        )
