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

def window_positions(region_height: int, region_width: int, window_size: int, step: int) -> list[tuple[int, int]]:
    """ Takes a region height and width, a window size, and a step size, and returns the list of top left positions to place windows at.
    The step is smaller than the window, which is what makes them overlap.
    If step does not divide evenly (extends), push last window to row/column of region_cl/row - window_size, so that no window hangs over the edge."""

    # Find row positions
    row_positions = []
    row = 0

    while row + window_size <= region_height:
        row_positions.append(row)

        if row + step + window_size > region_height:
            final = region_height - window_size
            if row_positions[-1] != final:
                row_positions.append(final)
            break

        row += step

    # Find column positions
    column_positions = []
    column = 0

    while column + window_size <= region_width:
        column_positions.append(column)

        if column + step + window_size > region_width:
            final = region_width - window_size
            if column_positions[-1] != final:
                column_positions.append(final)
            break

        column += step

    # Combine every row position with every column position
    positions = []
    for row in row_positions:
        for column in column_positions:
            positions.append((row, column))

    return positions



def weight_grid(height: int, width: int) -> np.ndarray:
    """Create a function that returns a grid of weights the size of a patch, since the weights get applied to what is written into the store.
    Weights should be largest in the middle and get smaller toward the edges. Every weight must be greater than zero. 
    A weight of exactly zero means a cell in the corner of a region, covered by only one window, can never be filled in.
    The same grid is used for every window so it only needs to be worked out once."""

    #NOTES:
    # Distance-Based Weighting For Vignettes or Radial Masks - linear distance decay function: each (row, column) = 1 - distance to center/maximum patch radius 
    # numpy array: [[row 1 contents], [row 2 contents]]
    # indexing in 2D Array: array[row, column]

    # create 1D arrays
    rows = np.arange(height)
    columns = np.arange(width)

    # find center
    center_row = (height - 1) / 2   # -1 because we start from 0
    center_column = (width - 1) / 2

    # distance from center
    row_distance = np.abs(rows - center_row)
    column_distance = np.abs(columns - center_column)

    # weight: apply formula. multiplied 0.9 so values stay above 0
    row_weight = 1 - 0.9 * row_distance / (height / 2)
    column_weight = 1 - 0.9 * column_distance / (width / 2)

    # combine 
    weights = np.outer(row_weight, column_weight)

    return weights