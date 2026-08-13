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
# It gives a noisy patch to the model and gets a processed full resolution onne back

# What does the sampler generate? 
# The sampler generates noise, overlapping windows over region, processes each one through model pipleine, then stores in terrain cache. Terrain Store makes and and returns heightgrid (2D numpy array).

# What do we store in the terrain cache? 
# It holds weight grids for tiles being generated and finished (Sum grid: running total of value × weight and Weight grid: running total of weight)


# import numpy as np

def window_positions(region_height, region_width, window_size, step) -> list[tuple]:
    """ Takes a region height and width, a window size, and a step size, and returns the list of top left positions to place windows at.
    The step is smaller than the window, which is what makes them overlap.
    If step does not divide evenly, push last window to row/column pf region_cl/row - window_size, so that no window hangs over the edge."""

    #rows:

    #columns:

    raise NotImplementedError