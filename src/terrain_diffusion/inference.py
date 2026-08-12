"""Model Inference

Overview

Runs a single neural model on a patch. It is the only component that loads model weights
and uses the GPU.
Given a patch, a choice of which model to run, and any conditioning, it runs that one model
and returns its raw output.
Different models return different things. The core model returns a low resolution elevation
summary and a latent map. The decoder returns a full resolution grid. Composing these into a
finished patch is the Model Pipeline's job, not this component's.

Neighbours and communication

- The Model Pipeline sends a patch and a choice of model and receives that model's raw output.
- It loads weights from the external model weights download.
- It runs on the GPU compute node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

PATCH_SIZE = (512, 512)
LATENT_MAP_SIZE = (3, 50, 100)  # placeholder


@dataclass
class ModelOutput(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError


@dataclass
class ModelInput(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError


class TerrainModel[InputT: ModelInput, OutputT: ModelOutput](ABC):
    @abstractmethod
    def predict(self, patch: InputT) -> OutputT:
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model_path: str):
        """
        Load a model stored in model_path
        """
        raise NotImplementedError

    def __call__(self, patch: InputT) -> OutputT:
        return self.predict(patch)


@dataclass
class MockCoreModelInput(ModelInput):
    patch: np.ndarray

    def __eq__(self, other: MockCoreModelInput):
        return np.array_equal(self.patch, other.patch)


@dataclass
class MockCoreModelOutput(ModelOutput):
    low_res_grid: np.ndarray
    latent_map: np.ndarray

    def __eq__(self, other: MockCoreModelInput):
        return np.array_equal(self.low_res_grid, other.low_res_grid) and np.array_equal(
            self.latent_map, other.latent_map
        )


@dataclass
class MockDecoderModelInput(ModelInput):
    latent_map: np.ndarray

    def __eq__(self, other: MockDecoderModelInput):
        return np.array_equal(self.latent_map, other.latent_map)


@dataclass
class MockDecoderModelOutput(ModelOutput):
    full_res_grid: np.ndarray

    def __eq__(self, other: MockCoreModelOutput):
        return np.array_equal(self.full_res_grid, other.full_res_grid)


class MockCoreModel(TerrainModel[MockCoreModelInput, MockCoreModelOutput]):
    weights: np.ndarray

    def predict(self, input: MockCoreModelInput) -> MockCoreModelOutput:

        double = input.patch * 2
        low_res_grid = np.resize(double, (PATCH_SIZE[0] // 8, PATCH_SIZE[1] // 8))
        latent_map = np.resize(double, LATENT_MAP_SIZE)
        output = MockCoreModelOutput(low_res_grid, latent_map)

        return output

    def load_weights(self, model_path: str):
        self.weights = np.ones((3, 4, 5))

    def __call__(self, input: MockCoreModelInput) -> MockCoreModelOutput:
        return self.predict(input)


class MockDecoderModel(TerrainModel[MockDecoderModelInput, MockDecoderModelOutput]):
    weights: np.ndarray

    def predict(self, input: MockDecoderModelInput) -> MockDecoderModelOutput:

        double = input.latent_map * 2
        full_res_grid = np.resize(double, PATCH_SIZE)
        output = MockDecoderModelOutput(full_res_grid)

        return output

    def load_weights(self, model_path: str):
        self.weights = np.ones((3, 4, 5))

    def __call__(self, input: MockDecoderModel) -> MockDecoderModel:
        return self.predict(input)


def load_model(model_name: str) -> TerrainModel:
    if model_name == "decoder":
        return MockDecoderModel()
    elif model_name == "core":
        return MockCoreModel()
    else:
        raise ValueError("Invalid model")
