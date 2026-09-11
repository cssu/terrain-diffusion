"""Terrain Store

Overview

Holds the sum and weight grids that terrain is blended into, per tile.
Its behaviour is what takes the project from a bounded picture to an infinite world in constant
memory.
Tiles partition the world with no overlap, and a window that crosses a tile boundary is split
between the tiles it covers. Only a limited number of tiles are held at once, and the least
recently used one is dropped to make room.

Neighbours and communication

- The Windowed Blending Sampler writes window contributions into it and reads them back.
- Generation Orchestration reads finished height grids from it.
- It can persist tiles to local disk.
"""

from collections import OrderedDict

import numpy as np

TileKey = tuple[int, int, int]


class TerrainStore:
    """Per tile sum and weight grids, read back as height = sum(value * weight) / sum(weight)."""

    def __init__(self, tile_height: int, tile_width: int, capacity: int) -> None:
        if tile_height < 1 or tile_width < 1:
            raise ValueError(f"tiles must be at least 1x1, asked for {tile_height}x{tile_width}")
        if capacity < 1:
            raise ValueError(f"store must hold at least one tile, asked for {capacity}")

        self.tile_height = tile_height
        self.tile_width = tile_width
        self.capacity = capacity
        self._tiles: OrderedDict[TileKey, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._tiles)

    def __contains__(self, key: TileKey) -> bool:
        return key in self._tiles

    def held_tiles(self) -> list[TileKey]:
        """Least recently used first."""
        return list(self._tiles)

    def grids(self, seed: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
        key = (seed, x, y)
        held = self._tiles.get(key)

        if held is None:
            held = (
                np.zeros((self.tile_height, self.tile_width)),
                np.zeros((self.tile_height, self.tile_width)),
            )
            self._tiles[key] = held
            self._drop_least_recently_used_if_over_capacity()
        else:
            self._tiles.move_to_end(key)

        return held

    def add_window(
        self,
        seed: int,
        values: np.ndarray,
        weights: np.ndarray,
        row: int,
        col: int,
    ) -> None:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if values.ndim != 2:
            raise ValueError(f"window must be a 2d grid, got {values.ndim} dimensions")
        if values.shape != weights.shape:
            raise ValueError(
                f"window and weights must be the same shape, got {values.shape} and {weights.shape}"
            )

        window_height, window_width = values.shape
        rows = range(row // self.tile_height, (row + window_height - 1) // self.tile_height + 1)
        cols = range(col // self.tile_width, (col + window_width - 1) // self.tile_width + 1)

        # Parts are written one tile at a time, so a window covering more tiles than the store
        # holds would drop its own earlier parts before the rest were written.
        covered = len(rows) * len(cols)
        if covered > self.capacity:
            raise ValueError(
                f"a {window_height}x{window_width} window at ({row}, {col}) covers {covered} "
                f"tiles, more than the {self.capacity} this store holds"
            )

        for y in rows:
            for x in cols:
                self._add_window_part(seed, x, y, values, weights, row, col)

    def _add_window_part(
        self,
        seed: int,
        x: int,
        y: int,
        values: np.ndarray,
        weights: np.ndarray,
        row: int,
        col: int,
    ) -> None:
        tile_top = y * self.tile_height
        tile_left = x * self.tile_width
        window_height, window_width = values.shape

        top = max(row, tile_top)
        left = max(col, tile_left)
        bottom = min(row + window_height, tile_top + self.tile_height)
        right = min(col + window_width, tile_left + self.tile_width)

        inside_window = (slice(top - row, bottom - row), slice(left - col, right - col))
        inside_tile = (
            slice(top - tile_top, bottom - tile_top),
            slice(left - tile_left, right - tile_left),
        )

        sums, tile_weights = self.grids(seed, x, y)
        part_values = values[inside_window]
        part_weights = weights[inside_window]
        sums[inside_tile] += part_values * part_weights
        tile_weights[inside_tile] += part_weights

    def is_complete(self, seed: int, x: int, y: int) -> bool:
        held = self._tiles.get((seed, x, y))
        if held is None:
            return False

        return bool(np.all(held[1] > 0))

    def unfilled_count(self, seed: int, x: int, y: int) -> int:
        held = self._tiles.get((seed, x, y))
        if held is None:
            return self.tile_height * self.tile_width

        return int(np.count_nonzero(held[1] == 0))

    def heights(self, seed: int, x: int, y: int) -> np.ndarray | None:
        key = (seed, x, y)
        held = self._tiles.get(key)
        if held is None:
            return None

        self._tiles.move_to_end(key)
        sums, weights = held

        # A zero weight would silently divide into nan, so unfilled cells have to be refused here.
        unfilled = int(np.count_nonzero(weights == 0))
        if unfilled:
            raise ValueError(
                f"{unfilled} of {self.tile_height * self.tile_width} cells of tile {key} have not "
                "been written to yet, so it cannot be read"
            )

        return sums / weights

    def _drop_least_recently_used_if_over_capacity(self) -> None:
        if len(self._tiles) > self.capacity:
            self._tiles.popitem(last=False)
