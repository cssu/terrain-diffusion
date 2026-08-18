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


# import numpy as np

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