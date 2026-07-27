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
