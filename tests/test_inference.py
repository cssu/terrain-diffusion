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
    load_models
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
            MockCoreModelInput(np.ones((1, 1)))

    def test_core_output_generation(self):
        with pytest.raises(AssertionError):
            MockCoreModelOutput(np.ones((1, 1)), np.ones(LATENT_MAP_SIZE))
        with pytest.raises(AssertionError):
            MockCoreModelOutput(np.ones((PATCH_SIZE[0] // 8, PATCH_SIZE[1] // 8)), np.ones((1, 1)))

    def test_decoder_input_generation(self):
        with pytest.raises(AssertionError):
            MockDecoderModelInput(np.ones((1, 1)))

    def test_decoder_output_generation(self):
        with pytest.raises(AssertionError):
            MockDecoderModelOutput(np.ones((1, 1)))

    def test_load_model(self):
        assert isinstance(load_model("decoder"), MockDecoderModel), (
            "expected decoder model, got core model instead"
        )
        assert isinstance(load_model("core"), MockCoreModel), "expected core model, got decoder model instead"

    def test_load_models(self):
        models = list(load_models("core", "decoder"))
        assert len(models) == 2, "invalid number of models returned"

        assert isinstance(models[0], MockCoreModel), "expected core model, got decoder model instead"
        assert isinstance(models[1], MockDecoderModel), "expected decoder model, got core model instead"

        models_2 = list(load_models("decoder", "core"))
        assert isinstance(models_2[0], MockDecoderModel), "expected decoder model, got core model instead"
        assert isinstance(models_2[1], MockCoreModel), "expected core model, got decoder model instead"

    def test_predict_core(self, initial_core):
        input = MockCoreModelInput(np.ones(PATCH_SIZE))

        actual = initial_core.predict(input)
        actual_2 = initial_core.predict(input)

        expected_low_res = np.ndarray((PATCH_SIZE[0] // 8, PATCH_SIZE[1] // 8))
        expected_latent = np.ndarray(LATENT_MAP_SIZE)

        expected_low_res.fill(2)
        expected_latent.fill(2)

        expected = MockCoreModelOutput(expected_low_res, expected_latent)

        assert actual == expected, "predictions are not equal"
        assert actual == actual_2, "model return different predictions on same input"
        assert actual.low_res_grid.shape == (PATCH_SIZE[0] // 8, PATCH_SIZE[1] // 8), (
            "low resolution map shape is not patch size // 8"
        )
        assert actual.latent_map.shape == LATENT_MAP_SIZE, "latent map size is not correct"

    def test_predict_decoder(self, initial_decoder):
        input = MockDecoderModelInput(np.ones(LATENT_MAP_SIZE))

        actual = initial_decoder.predict(input)
        actual_2 = initial_decoder.predict(input)

        expected_full_res = np.ndarray(PATCH_SIZE)
        expected_full_res.fill(2)

        expected = MockDecoderModelOutput(expected_full_res)

        assert actual == expected, "predictions are not equal"
        assert actual == actual_2, "model return different predictions on same input"
        assert actual.full_res_grid.shape == PATCH_SIZE, (
            "full resolution map does not match patch size"
        )
