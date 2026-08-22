"""Windowed Blending Sampler

Overview

The algorithm that turns a fixed size model into a generator of any size with no seams.

Neighbours and communication

- Receives a request for a region from Generation Orchestration.
- Asks the Model Pipeline to clean patches.
- Writes results into the Terrain Store and reads them back.
"""

# What value does the window sampler take from generation orchestration?
# Seed, Region Size, and Region coordinates

# What does it expect from the model inference?
# It gives a noisy patch to the model and gets a processed full resolution one back

# What does the sampler generate?
# The sampler generates noise, overlapping windows over region, processes each one through model pipleine, then stores in terrain cache. Terrain Store makes and and returns heightgrid (2D numpy array).

# What do we store in the terrain cache?
# It holds weight grids for tiles being generated and finished (Sum grid: running total of value × weight and Weight grid: running total of weight)

import numpy as np


def window_positions(
    region_height: int, region_width: int, window_size: int, step: int
) -> list[tuple[int, int]]:
    """Takes a region height and width, a window size, and a step size, and returns the list of top left positions to place windows at.
    The step is smaller than the window, which is what makes them overlap.
    If step does not divide evenly, throws an assertion error."""

    assert region_height % window_size == 0
    assert region_width % window_size == 0

    # Find row positions
    row_positions = []
    row = 0

    while row + window_size <= region_height:
        row_positions.append(row)
        row += step

    # Find column positions
    column_positions = []
    column = 0

    while column + window_size <= region_width:
        column_positions.append(column)
        column += step

    # Combine every row position with every column position
    positions = []
    for row in row_positions:
        for column in column_positions:
            positions.append((row, column))

    return positions


def weight_grid(edge_len: int) -> np.ndarray:
    """Create a function that returns a grid of weights the size of a patch, since the weights get applied to what is written into the store.
    Weights should be largest in the middle and get smaller toward the edges. Every weight must be greater than zero.
    A weight of exactly zero means a cell in the corner of a region, covered by only one window, can never be filled in.
    The same grid is used for every window so it only needs to be worked out once."""

    # NOTES:
    # Distance-Based Weighting For Vignettes or Radial Masks - linear distance decay function: each (row, column) = 1 - distance to center/maximum patch radius
    # numpy array: [[row 1 contents], [row 2 contents]]
    # indexing in 2D Array: array[row, column]

    assert edge_len > 1

    # create 1D arrays
    positions = np.arange(edge_len)

    # find center (-1 because we start from 0)
    center = (edge_len - 1) / 2

    # distance from center
    distance = np.abs(positions - center)

    # weight: apply formula. multiplied 0.9 so values stay above 0
    weight = 1 - 0.9 * distance / center
    # combine
    weights = np.outer(weight, weight)

    return weights


def generate_noise_from_seed(seed: int, height: int, width: int) -> np.ndarray:
    "Takes a seed and a canvas size and returns a grid of random numbers that size"
    generator = np.random.default_rng(seed)
    return generator.random((height, width))


def produce_region(
    seed: int,
    height: int,
    width: int,
    window_size: int,
    step: int,
    pipeline,
) -> tuple[np.ndarray]:
    """Make noise canvas of given dimensions. Make noise and weight grid.
    For each window position, cut the window out of the noise canvas, send it to pipeline, add the processed output and its weight to Terrain Store (at that position).
    Read the finished height grid from store and return it."""

    noise = generate_noise_from_seed(seed, height, width)

    positions = window_positions(height, width, window_size, step)
    weights = weight_grid(window_size)  # weight grid made on window_size

    weighted_sum = np.zeros((height, width))
    weight_sum = np.zeros((height, width))

    for row, column in positions:
        window = noise[row : row + window_size, column : column + window_size]

        processed_patch = pipeline.generate(window)

        # From FakeStore()
        weighted_sum[row : row + window_size, column : column + window_size] += (
            processed_patch * weights
        )
        weight_sum[row : row + window_size, column : column + window_size] += weights

    return weighted_sum, weight_sum
