"""Elevation Encoding

Overview

A set of reversible transforms between real elevations and the form the models work in.
Real elevations span a very large range, from deep ocean trenches to high mountains, and the
models were trained on compressed and split elevations rather than raw ones.
The forward direction compresses the range and splits a height grid into a base layer and a
detail layer. The reverse direction reassembles the base and detail and undoes the compression
to recover real elevations.
Generation uses the reverse direction: the Model Pipeline turns the models' base and detail
outputs back into real elevations. The forward direction is its exact inverse, used to define
the transforms and to confirm they round trip.

Neighbours and communication

- The Model Pipeline calls the reverse direction to turn the models' base and detail outputs
  into real elevations.
- The transforms are reversible, so applying the forward direction and then the reverse returns
  the original within a small tolerance.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF

DIMENSIONS_OF_GRID = 2  # Dimension of the grid that contains the map; the map is, naturally, 3D.


class Entrypoint:
    """
    A module entrypoint that takes in two grids of different sizes (low res and detail grid).
    Representation invariants:
     - len(low_res.shape) == DIMENSIONS_OF_GRID
     - len(residual.shape) == DIMENSIONS_OF_GRID
    """
    low_res: np.ndarray
    residual: np.ndarray

    def __init__(self, low_res: np.ndarray, residual: np.ndarray):
        assert len(low_res.shape) == DIMENSIONS_OF_GRID and \
               len(residual.shape) == DIMENSIONS_OF_GRID
        self.low_res = low_res
        self.residual = residual


def pad_linear_extrapolation(x):
    """

    :param x: terrain heightmap
    :return: padded out terrain heightmap, using linear extrapolation
    """
    # x: (..., H, W)
    h, w = x.shape[-2:]

    # Pad H
    if h > 1:
        top = x[..., 0:1, :]
        second = x[..., 1:2, :]
        top_pad = 2 * top - second

        bot = x[..., -1:, :]
        second_last = x[..., -2:-1, :]
        bot_pad = 2 * bot - second_last
    else:
        top_pad = x[..., 0:1, :]
        bot_pad = x[..., -1:, :]

    x = torch.cat([top_pad, x, bot_pad], dim=-2)

    # Pad W
    if w > 1:
        left = x[..., :, 0:1]
        second_w = x[..., :, 1:2]
        left_pad = 2 * left - second_w

        right = x[..., :, -1:]
        second_last_w = x[..., :, -2:-1]
        right_pad = 2 * right - second_last_w
    else:
        left_pad = x[..., :, 0:1]
        right_pad = x[..., :, -1:]

    x = torch.cat([left_pad, x, right_pad], dim=-1)
    return x


def resize_extrapolated(x,
                        size: [tuple | list],
                        interpolation=TF.InterpolationMode.BILINEAR,
                        **kwargs):
    """

    :param x: terrain heightmap
    :param size: the target size wanted
    :param interpolation: the interpolation type used
    :return: x, resized to the desired scale
    """
    if not isinstance(size, (tuple, list)):
        return TF.resize(x, size, interpolation=interpolation, **kwargs)

    target_h, target_w = size
    h, w = x.shape[-2:]

    scale_h = target_h / h
    scale_w = target_w / w

    x_padded = pad_linear_extrapolation(x)

    new_h = int(round(target_h + 2 * scale_h))
    new_w = int(round(target_w + 2 * scale_w))

    out = TF.resize(x_padded, [new_h, new_w], interpolation=interpolation, **kwargs)

    pad_h = int(round(scale_h))
    pad_w = int(round(scale_w))

    return out[..., pad_h:pad_h + target_h, pad_w:pad_w + target_w]


def laplacian_encode(x: np.ndarray | torch.Tensor,
                     downsample_size,
                     sigma,
                     interp_mode=TF.InterpolationMode.BILINEAR,
                     extrapolate=False):
    """

    :param x: terrain heightmap
    :param downsample_size: the target size we want to downsize to
    :param sigma: sigma value used for gaussian blurring
    :param interp_mode: the desired interpolation mode. Bilinear is set as default, since that's what was used in
    the original paper
    :param extrapolate: if extrapolation is desired before encoding. Personally, I don't like merging these two into
    a single step, but that's what the paper does.
    :return:
    """
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(x)

    # Unsqueeze to 4 dimensions if needed
    squeeze_count = 0
    while x.ndim < 4:
        x = x.unsqueeze(0)
        squeeze_count += 1

    lowres = TF.resize(x, downsample_size, interpolation=interp_mode)
    lowres = TF.gaussian_blur(lowres, kernel_size=[(sigma * 2) // 2 * 2 + 1, ], sigma=sigma)
    if not extrapolate:
        lowres_up = TF.resize(lowres, list(x.shape[-2:]), interpolation=interp_mode)
    else:
        lowres_up = resize_extrapolated(lowres, x.shape[-2:], interpolation=interp_mode)
    residual = x - lowres_up

    # Squeeze back to original dimensions
    while squeeze_count > 0:
        residual = residual.squeeze(0)
        lowres = lowres.squeeze(0)
        squeeze_count -= 1

    if is_numpy:
        residual = residual.numpy()
        lowres = lowres.numpy()
    return residual, lowres


def laplacian_decode(residual: np.ndarray,
                     lowres: np.ndarray,
                     interp_mode=TF.InterpolationMode.BILINEAR,
                     extrapolate=False,
                     pre_padded=False) -> tuple[np.ndarray, np.ndarray]:
    """

    :param residual: residual of the heightmap
    :param lowres: low-res version of heightmap
    :param interp_mode: the desired interpolation mode. Bilinear is set as default
    :param extrapolate: if extrapolation is desired.
    :param pre_padded: if the heightmap is padded. I'd prefer if you don't use this option.
    :return: A 2-tuple containing the residual and the upscaled low-res heightmap
    """
    assert (isinstance(residual, np.ndarray) == isinstance(lowres, np.ndarray))
    is_numpy = isinstance(residual, np.ndarray)

    # Convert to torch first if numpy (Should be the case in our case)
    if is_numpy:
        residual = torch.from_numpy(residual)
        lowres = torch.from_numpy(lowres)

    # Unsqueeze to 4 dimensions if needed
    squeeze_count = 0
    while residual.ndim < 4:
        residual = residual.unsqueeze(0)
        lowres = lowres.unsqueeze(0)
        squeeze_count += 1

    resize_shape = residual.shape[-2:]
    if pre_padded:
        pad_pixels = residual.shape[-1] // (lowres.shape[-1] - 2)
        resize_shape = (resize_shape[-2] + 2 * pad_pixels, resize_shape[-1] + 2 * pad_pixels)
    else:
        resize_shape = residual.shape[-2:]
    if not extrapolate:
        lowres_up = TF.resize(lowres, resize_shape, interpolation=interp_mode)
    else:
        lowres_up = resize_extrapolated(lowres, resize_shape, interpolation=interp_mode)

    if pre_padded:
        lowres_up = lowres_up[..., pad_pixels:-pad_pixels, pad_pixels:-pad_pixels]

    # Squeeze back to original dimensions
    while squeeze_count > 0:
        residual = residual.squeeze(0)
        lowres = lowres.squeeze(0)
        lowres_up = lowres_up.squeeze(0)
        squeeze_count -= 1

    if is_numpy:  # Should always run, this is here just bc the original paper supported
        residual = residual.numpy()
        lowres_up = lowres_up.numpy()
    return residual, lowres_up


def laplacian_denoise(residual,
                      lowres,
                      sigma,
                      interp_mode=TF.InterpolationMode.BILINEAR) -> tuple:
    """
    :param residual: residual of the heightmap
    :param lowres: low-res version of heightmap
    :param sigma: sigma value used for gaussian blurring
    :param interp_mode: the desired interpolation mode. Bilinear is set as default
    :return: the residual and the new lowres, decoded and re-encoded.
    """
    decoded_residual, decoded_lowres_up = laplacian_decode(residual, lowres, interp_mode, extrapolate=True)
    _, new_lowres = laplacian_encode(decoded_residual + decoded_lowres_up, lowres.shape[-1], sigma, interp_mode)
    return residual, new_lowres


def re_extraction(x: np.ndarray | torch.Tensor,
                  downsample_size):
    """
    This re-extraction is not perfect, but according to the paper should be robust, precise, and accurate
    :param x:
    :param downsample_size:
    :return:
    """
    _, decoded_lowres_up = laplacian_decode(laplacian_encode(x, downsample_size))
    return x - decoded_lowres_up, decoded_lowres_up


def scale_heights(
        height_grid: np.ndarray,
        src_range: tuple[float, float],
        tgt_range: tuple[float, float],
) -> np.ndarray:
    """
    Preconditions:
     - np.all((height_grid >= src_min) & (height_grid <= src_max)):
     - src_range[0] <= src_range[1]
     - tgt_range[0] <= tgt_range[1]
    :param height_grid:
    :param src_range: a tuple representing the compact interval that the height grid's domain currently lives in
    :param tgt_range: a tuple representing the compact interval that the height grid's domain will be transformed into
    :return: a new height grid linearly transformed with tgt_range as codomain.
    """

    src_min, src_max = src_range
    tgt_min, tgt_max = tgt_range
    if not np.all((height_grid >= src_min) & (height_grid <= src_max)):
        raise ValueError("heights contains values outside src_range")
    assert src_min <= src_max, "Invalid source range"
    assert tgt_min <= tgt_max, "Invalid destination range"

    return tgt_min + (height_grid - src_min) * (tgt_max - tgt_min) / (src_max - src_min)

# def resize(self, input_grid: np.ndarray, desired_scale: ClassVar[tuple]) -> np.ndarray:
#     """
#     A function that takes in a small grid and scale it to a desired size and do the reverse.
#     Utilizes linear interpolation to "fill"
#     :param input_grid: the np grid to be resized
#     :param desired_scale: the target size of the rescaled array
#     :return output_grid: input grid, rescaled to the correct scale
#     """
#     assert len(input_grid.shape) == 2
#     zoom_factors = np.array(desired_scale) / np.array(input_grid.shape)
#     output_grid = zoom(input_grid, zoom_factors, order=1)
#     return output_grid
#
#
# def combine_lowres_and_detail(low_res_grid: np.ndarray, detail_grid: np.ndarray) -> np.ndarray:
#     """
#     Apply the scaling to the low res grid, then add the detail and scaled low res grid together
#     :param low_res_grid: the low res grid to be rescaled to the size of the detail array
#     :param detail_grid: the high res grid that the low-res grid will be applied on top of
#     :return: combined_grid: the sum of these two grids
#     """
#     low_res_grid_expanded = resize(low_res_grid, detail_grid.shape)
#     combined_grid = detail_grid + low_res_grid_expanded
#     return combined_grid
#
#
# def split_map(compiled_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Split a full resolution map into a low res grid and detailed grid
#     :param compiled_grid: the grid that has its low-res and high-res grids compiled
#     :return: (low_res_grid, detail_grid): the grid, decompiled from its combination.
#     """
