"""Terrain Store

Overview

Holds the height grids for tiles being generated and finished.
Its behaviour is what takes the project from a bounded picture to an infinite world in constant
memory.
For a bounded region the sum and weight buffers cover the whole region. For infinite exploration
they are held per tile, so each tile can be finished, cached, evicted, and recomputed on its own.

Neighbours and communication

- The Windowed Blending Sampler writes window contributions into it and reads them back.
- Generation Orchestration reads finished height grids from it.
- It can persist tiles to local disk.
"""
