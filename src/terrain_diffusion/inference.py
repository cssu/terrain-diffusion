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

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from terrain_diffusion.models.edm_unet import EDMUnet2D

PATCH_SIZE = (512, 512)

LATENT_SIZE = 64
LATENT_MAP_SIZE = (4, LATENT_SIZE, LATENT_SIZE)
CORE_INPUT_SIZE = (5, LATENT_SIZE, LATENT_SIZE)


LATENT_COMPRESSION = 8
DECODER_INPUT_SIZE = (5, PATCH_SIZE[0], PATCH_SIZE[1])

CORE_COND_VECTOR_DIM = 58

SIGMA_DATA = 0.5
SIGMA_0 = 80.0
CORE_INIT_T = float(np.arctan(SIGMA_0 / SIGMA_DATA))


CORE_INTERMEDIATE_SIGMA = 0.35
CORE_INTERMEDIATE_T = float(np.arctan(CORE_INTERMEDIATE_SIGMA / SIGMA_DATA))


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


@dataclass
class CoreModelInput(ModelInput):
    patch: np.ndarray
    conditioning: np.ndarray | None = None
    noise_level: float = CORE_INIT_T
    patch_shape: ClassVar[tuple] = CORE_INPUT_SIZE
    conditioning_shape: ClassVar[tuple] = (CORE_COND_VECTOR_DIM,)

    def __post_init__(self):
        assert self.patch.shape == self.patch_shape, "invalid input patch shape"
        if self.conditioning is not None:
            assert self.conditioning.shape == self.conditioning_shape, "invalid conditioning shape"

    def __eq__(self, other: CoreModelInput):
        return np.array_equal(self.patch, other.patch)


@dataclass
class CoreModelOutput(ModelOutput):
    low_res_grid: np.ndarray
    latent_map: np.ndarray
    low_res_grid_shape: ClassVar[tuple] = (LATENT_SIZE, LATENT_SIZE)
    latent_map_shape: ClassVar[tuple] = LATENT_MAP_SIZE

    def __post_init__(self):
        assert self.latent_map.shape == self.latent_map_shape, "invalid latent map size"
        assert self.low_res_grid.shape == self.low_res_grid_shape, (
            "invalid low resolution grid shape"
        )

    def __eq__(self, other: CoreModelInput):
        return np.array_equal(self.low_res_grid, other.low_res_grid) and np.array_equal(
            self.latent_map, other.latent_map
        )


@dataclass
class DecoderModelInput(ModelInput):
    latent_map: np.ndarray
    noise_level: float = CORE_INIT_T
    latent_map_shape: ClassVar[tuple] = LATENT_MAP_SIZE

    def __post_init__(self):
        assert self.latent_map.shape == self.latent_map_shape, "invalid latent map size"

    def __eq__(self, other: DecoderModelInput):
        return np.array_equal(self.latent_map, other.latent_map)


@dataclass
class DecoderModelOutput(ModelOutput):
    """The detail layer the decoder predicts, at full patch resolution.

    The decoder takes the core model's latents upsampled to the patch size, so it
    runs at 512x512 and predicts the residual at that resolution. Adding the
    upsampled low frequency grid back in is the elevation encoder's job.
    """

    full_res_grid: np.ndarray
    full_res_grid_shape: ClassVar[tuple] = PATCH_SIZE

    def __post_init__(self):
        assert self.full_res_grid.shape == self.full_res_grid_shape, (
            "invalid full resolution grid shape"
        )

    def __eq__(self, other: CoreModelOutput):
        return np.array_equal(self.full_res_grid, other.full_res_grid)


class MockCoreModel(TerrainModel[CoreModelInput, CoreModelOutput]):
    weights: np.ndarray

    def predict(self, input: CoreModelInput) -> CoreModelOutput:

        double = input.patch * 2
        low_res_grid = np.resize(double, CoreModelOutput.low_res_grid_shape)
        latent_map = np.resize(double, CoreModelOutput.latent_map_shape)
        output = CoreModelOutput(low_res_grid, latent_map)

        return output


class MockDecoderModel(TerrainModel[DecoderModelInput, DecoderModelOutput]):
    weights: np.ndarray

    def predict(self, input: DecoderModelInput) -> DecoderModelOutput:

        double = input.latent_map * 2
        full_res_grid = np.resize(double, DecoderModelOutput.full_res_grid_shape)
        output = DecoderModelOutput(full_res_grid)

        return output


class CoreModel(TerrainModel[CoreModelInput, CoreModelOutput]):
    model: EDMUnet2D

    def __init__(self, model_path, subfolder_name=""):
        self.model = EDMUnet2D.from_pretrained(model_path, subfolder=subfolder_name)
        self.model.eval()

    def predict(self, input: CoreModelInput) -> CoreModelOutput:

        conditioning = input.conditioning
        if conditioning is None:
            conditioning = np.zeros(
                CORE_COND_VECTOR_DIM, dtype=np.float32
            )  # zero for now since its produced by the coarse model
        conditional_inputs = [torch.from_numpy(conditioning)[None].float()]


        x = torch.from_numpy(input.patch).float().unsqueeze(0)
        t = input.noise_level

        with torch.no_grad():
            x_t = x * SIGMA_DATA
            pred = self.model(
                x,
                noise_labels=torch.tensor([t], dtype=torch.float32),
                conditional_inputs=conditional_inputs,
            )
            sample = math.cos(t) * x_t + math.sin(t) * SIGMA_DATA * pred


            t = CORE_INTERMEDIATE_T
            z = torch.randn_like(sample) * SIGMA_DATA
            x_t = math.cos(t) * sample + math.sin(t) * z
            pred = self.model(
                x_t / SIGMA_DATA,
                noise_labels=torch.tensor([t], dtype=torch.float32),
                conditional_inputs=conditional_inputs,
            )
            sample = math.cos(t) * x_t + math.sin(t) * SIGMA_DATA * pred

        sample = sample[0].numpy()
        return CoreModelOutput(low_res_grid=sample[4], latent_map=sample[:4])


class DecoderModel(TerrainModel[DecoderModelInput, DecoderModelOutput]):
    model: EDMUnet2D

    def __init__(self, model_path, subfolder_name=""):
        self.model = EDMUnet2D.from_pretrained(model_path, subfolder=subfolder_name)
        self.model.eval()

    def predict(self, input: DecoderModelInput) -> DecoderModelOutput:
        latents = torch.from_numpy(input.latent_map).float()[None]

        latents = torch.nn.functional.interpolate(
            latents, size=DECODER_INPUT_SIZE[1:], mode="nearest"
        )
        rng = np.random.default_rng()
        noise = rng.standard_normal(DECODER_INPUT_SIZE[1:]).astype(np.float32)
        x = torch.cat([torch.from_numpy(noise)[None, None], latents], dim=1)

        with torch.no_grad():
            sample = self.model(
                x,
                noise_labels=torch.tensor([input.noise_level], dtype=torch.float32),
                conditional_inputs=[],
            )

        return DecoderModelOutput(full_res_grid=sample[0, 0].numpy())


MODELS = {"decoder": DecoderModel, "core": CoreModel}


def load_model(model_name: str) -> TerrainModel:
    if model_name not in MODELS:
        raise ValueError("invalid model name")
    subfolder = "base_model" if model_name == "core" else "decoder_model"
    return MODELS[model_name]("xandergos/terrain-diffusion-30m", subfolder_name=subfolder)
