"""The Terrain Diffusion model architecture.

This package is the custom EDM2-style U-Net architecture from the reference
implementation of the InfiniteDiffusion paper, copied from
`xandergos/terrain-diffusion` (MIT License, Copyright (c) 2025 Alexander Goslin)
so the released pretrained weights can be loaded. See LICENSE in this package
for the full attribution and license text.

    - mp_layers          magnitude-preserving layers and helpers
    - unet_block         U-Net blocks built from those layers
    - edm_unet           EDMUnet2D, the architecture behind the core and decoder
    - edm_autoencoder    EDM autoencoder (encoder + decoder stack)
    - perceptron         MLP used for small learned heads

Paper: Goslin, Alexander. "InfiniteDiffusion: Bridging Learned Fidelity and
Procedural Utility for Open-World Terrain Generation." SIGGRAPH Conference
Papers 2026. DOI: 10.1145/3799902.3811080.
"""

from terrain_diffusion.models.edm_unet import EDMUnet2D

__all__ = ["EDMUnet2D"]
