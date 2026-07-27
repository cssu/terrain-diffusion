"""Generation Orchestration

Overview

Produces a finished region for a given seed and coordinate by driving the other core components.
In the build-out it also coordinates several models at different zoom levels.

Neighbours and communication

- Receives requests from Output and CLI and from the 3D Visualizer.
- Drives the Windowed Blending Sampler.
- Reads finished height grids from the Terrain Store.
- In the build-out it also runs the coarse model to produce conditioning for a region.
"""
