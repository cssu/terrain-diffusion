"""Generation Service

Overview

Puts Generation Orchestration behind HTTP so the 3D Visualizer can reach it.

The visualizer runs in a browser and cannot import Python, and Orchestration
runs on the machine holding the GPU. This component is the boundary between
them. It accepts a seed and a tile coordinate, asks Orchestration for the
region, and returns the height grid.

Output and CLI does not go through here. It calls Orchestration in process.

Neighbours and communication

- Receives requests over HTTP from the 3D Visualizer.
- Asks Generation Orchestration for a region and returns the height grid.

Notes for whoever builds this

- The project machine sits behind carrier-grade NAT and reaches the outside
  through an outbound tunnel, so the client address arrives in the
  `Cf-Connecting-Ip` and `X-Forwarded-For` headers. The socket always reports
  `127.0.0.1`. Anything that logs or limits by address has to read the header.
- Clients may be IPv6. The service itself can stay IPv4 only.
- There is one GPU with about 3 GB spare, so concurrent requests can exhaust
  it. Serialising generation is the simplest answer.
"""
