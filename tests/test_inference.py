"""
Tests inference interface by instantiating mock models
and implementing the `generate` function
"""

import numpy as np
import pytest

from terrain_diffusion.inference import (
    LATENT_MAP_SIZE,
    PATCH_SIZE,
    CoreModelInput,
    CoreModelOutput,
    CoreModel,
    DecoderModel,
    DecoderModelInput,
    DecoderModelOutput,
    MockCoreModel,
    MockDecoderModel,
    load_model,
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

    def test_load_model(self):
        assert isinstance(load_model("decoder"), MockDecoderModel), (
            "model does not load correct decoder"
        )
        assert isinstance(load_model("core"), MockCoreModel), "model does not load correct core"

    def test_predict_core(self, initial_core):
        input = CoreModelInput(np.ones(PATCH_SIZE))

        actual = initial_core.predict(input)
        actual_2 = initial_core.predict(input)

        expected_low_res = np.ndarray((PATCH_SIZE[0] // 8, PATCH_SIZE[1] // 8))
        expected_latent = np.ndarray(LATENT_MAP_SIZE)

        expected_low_res.fill(2)
        expected_latent.fill(2)

        expected = CoreModelOutput(expected_low_res, expected_latent)

        assert actual == expected, "predictions are not equal"
        assert actual == actual_2, "model return different predictions on same input"
        assert actual.low_res_grid.shape == (PATCH_SIZE[0] // 8, PATCH_SIZE[1] // 8), (
            "low resolution map shape is not patch size // 8"
        )
        assert actual.latent_map.shape == LATENT_MAP_SIZE, "latent map size is not correct"

    def test_predict_decoder(self, initial_decoder):
        input = DecoderModelInput(np.ones(LATENT_MAP_SIZE))

        actual = initial_decoder.predict(input)
        actual_2 = initial_decoder.predict(input)

        expected_full_res = np.ndarray(PATCH_SIZE)
        expected_full_res.fill(2)

        expected = DecoderModelOutput(expected_full_res)

        assert actual == expected, "predictions are not equal"
        assert actual == actual_2, "model return different predictions on same input"
        assert actual.full_res_grid.shape == PATCH_SIZE, (
            "full resolution map does not match patch size"
        )
