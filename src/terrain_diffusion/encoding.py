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
from scipy.ndimage import zoom
from typing import ClassVar


class Entrypoint:
    """
  A module entrypoint that takes in two grids of different sizes (low res and detail grid).
  """
    low_res_grid: np.ndarray
    detail_grid: np.ndarray

    def __init__(self, low_res_grid: np.ndarray, detail_grid):
        self.low_res_grid = low_res_grid
        self.detail_grid = detail_grid



class Encoding:
    """
    Class containing methods to determistically convert two height maps, one low resolution grid, and one high
    resolution detail grid, into one full resolution heightmap.
    """
    def resize(self, input_grid: np.ndarray, desired_scale: ClassVar[tuple]) -> np.ndarray:
        """
        A function that takes in a small grid and scale it to a desired size and do the reverse.
        Utilizes linear interpolation to "fill"
        :param input_grid: the np grid to be resized
        :param desired_scale: the target size of the rescaled array
        :return output_grid: input grid, rescaled to the correct scale
        """
        assert len(input_grid.shape) == 2
        zoom_factors = np.array(desired_scale) / np.array(input_grid.shape)
        output_grid = zoom(input_grid, zoom_factors, order=1)
        return output_grid

    def combine_lowres_and_detail(self, low_res_grid: np.ndarray, detail_grid: np.ndarray) -> np.ndarray:
        """
        Apply the scaling to the low res grid, then add the detail and scaled low res grid together
        :param low_res_grid: the low res grid to be rescaled to the size of the detail array
        :param detail_grid: the high res grid that the low-res grid will be applied on top of
        :return: combined_grid: the sum of these two grids
        """
        low_res_grid_expanded = self.resize(low_res_grid, detail_grid.shape)
        combined_grid = detail_grid + low_res_grid_expanded
        return combined_grid

    def split_map(self, compiled_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split a full resolution map into a low res grid and detailed grid
        :param compiled_grid:
        :return:
        """


    def scale_heights(
            self,
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
        :return: a new height grid linearlly transformed with tgt_range as codomain
        """

        src_min, src_max = src_range
        tgt_min, tgt_max = tgt_range
        if not np.all((height_grid >= src_min) & (height_grid <= src_max)):
            raise ValueError("heights contains values outside src_range")
        assert src_min <= src_max, "Invalid source range"
        assert tgt_min <= tgt_max, "Invalid destination range"


        return tgt_min + (height_grid - src_min) * (tgt_max - tgt_min) / (src_max - src_min)
