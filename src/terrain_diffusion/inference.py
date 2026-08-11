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

from abc import ABC, abstractmethod

import numpy as np


class TerrainModel(ABC):
    @abstractmethod
    def generate(self, patch: np.ndarray) -> np.ndarray:
        """
        Generate an output patch (C, H, W) from an input patch (C, H, W) using the model
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_model(cls, model_path: str) -> TerrainModel:
        """
        Load a model stored in model_path
        """
        raise NotImplementedError

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        return self.generate(patch)
