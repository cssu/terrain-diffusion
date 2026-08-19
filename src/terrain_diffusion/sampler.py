"""Windowed Blending Sampler

Overview

The algorithm that turns a fixed size model into a generator of any size with no seams.

Neighbours and communication

- Receives a request for a region from Generation Orchestration.
- Asks the Model Pipeline to clean patches.
- Writes results into the Terrain Store and reads them back.
"""

def f():
    print("this is a test")

def g():
    print("this is also a test")