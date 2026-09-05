"""Terrain Store

Overview

Holds the sum and weight grids that terrain is blended into.
Its behaviour is what takes the project from a bounded picture to an infinite world in constant
memory.

The grids are the store. They do not stop at the edge of a request and are not thrown away when
one finishes. They are held per tile, and a tile is a fixed block of the world that does not
overlap its neighbours, so the tiles together partition the world. A tile can be finished, kept,
dropped, and generated again on its own, which is what lets an endless world stay explorable in a
fixed amount of memory.

What is kept per tile is the running sum grid and the running weight grid, not a finished height
grid. Heights are worked out by dividing one by the other on the way out, so a tile that is read
twice is not blended twice.

Neighbours and communication

- The Windowed Blending Sampler writes window contributions into it and reads them back.
- Generation Orchestration reads finished height grids from it.
- It can persist tiles to local disk.

Not built yet

A window that crosses a tile boundary is split between the two tiles it covers, and a record is
kept of which windows have already been added so a later request skips them. Neither is needed
while a request is one tile of a known size, since every window is added exactly once, so
`add_window` refuses a window that crosses the edge rather than splitting it. See #56.
"""

from collections import OrderedDict

import numpy as np

# A tile is named by the global seed together with its coordinate, three numbers kept separate all
# the way to the lookup. Working them into one number would let two different tiles land on the
# same value and silently become one tile, handing back terrain from the wrong world or the wrong
# place. A tuple is compared for equality after it is hashed, so two different tiles stay two.
TileKey = tuple[int, int, int]


class TerrainStore:
    """The sum and weight grids of every tile currently being held.

    Windows of terrain overlap, so a cell is usually written by more than one of them. Rather than
    letting the last window win, each contribution is accumulated as a weighted average:

        height[cell] = sum(value * weight) / sum(weight)

    The two halves of that fraction are kept as two grids of their own, per tile. Every window adds
    into both, and `heights` divides one by the other on the way out.

    Only a limited number of tiles are held. Adding one past that limit drops the least recently
    used tile, where reading counts as use just as writing does.
    """

    def __init__(self, tile_height: int, tile_width: int, capacity: int) -> None:
        """Create an empty store of tiles of the given size, holding at most `capacity` of them."""
        if tile_height < 1 or tile_width < 1:
            raise ValueError(f"tiles must be at least 1x1, asked for {tile_height}x{tile_width}")
        if capacity < 1:
            raise ValueError(f"store must hold at least one tile, asked for {capacity}")

        self.tile_height = tile_height
        self.tile_width = tile_width
        self.capacity = capacity
        # Ordered least recently used first, so the tile to drop is the one at the front.
        self._tiles: OrderedDict[TileKey, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        """How many tiles are being held."""
        return len(self._tiles)

    def __contains__(self, key: TileKey) -> bool:
        """Whether a `(seed, x, y)` tile is held, without counting as a use of it."""
        return key in self._tiles

    def held_tiles(self) -> list[TileKey]:
        """The `(seed, x, y)` of every tile being held, least recently used first.

        The coordinate is kept rather than folded away, so the store can say where in the world the
        tiles it holds sit. Anything that later decides what to keep by distance needs that.
        """
        return list(self._tiles)

    def grids(self, seed: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
        """The sum and weight grids of a tile, created zeroed if it is not held yet.

        This is the write side of the store, so a tile that is not held is started rather than
        reported missing. Counts as a use of the tile.
        """
        key = (seed, x, y)
        held = self._tiles.get(key)

        if held is None:
            # float64, so that the division in `heights` is not silently truncated to whole numbers.
            held = (
                np.zeros((self.tile_height, self.tile_width)),
                np.zeros((self.tile_height, self.tile_width)),
            )
            self._tiles[key] = held
            self._drop_least_recently_used()
        else:
            self._tiles.move_to_end(key)

        return held

    def add_window(
        self,
        seed: int,
        x: int,
        y: int,
        values: np.ndarray,
        weights: np.ndarray,
        row: int,
        col: int,
    ) -> None:
        """Add one window's contribution to a tile, at the position its top left corner sits at.

        `x` and `y` say which tile in the world is being written to. `row` and `col` are inside
        that tile.

        Values are added multiplied by their weights, and the weights are added on their own.
        Adding, not replacing: a second window covering the same cells builds on the first.
        """
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if values.ndim != 2:
            raise ValueError(f"window must be a 2d grid, got {values.ndim} dimensions")
        if values.shape != weights.shape:
            raise ValueError(
                f"window and weights must be the same shape, got {values.shape} and {weights.shape}"
            )

        window_height, window_width = values.shape

        # numpy slices clip instead of complaining, and a negative index wraps round to the far
        # side of the grid. Both would write a window somewhere other than where it was asked for,
        # so the position is checked here rather than left to the slice.
        if row < 0 or col < 0:
            raise ValueError(f"window position ({row}, {col}) is outside the tile")
        if row + window_height > self.tile_height or col + window_width > self.tile_width:
            raise ValueError(
                f"a {window_height}x{window_width} window at ({row}, {col}) crosses the edge of a "
                f"{self.tile_height}x{self.tile_width} tile. Splitting a window across tiles is "
                "not built yet, see #56"
            )

        sums, tile_weights = self.grids(seed, x, y)
        rows = slice(row, row + window_height)
        cols = slice(col, col + window_width)
        sums[rows, cols] += values * weights
        tile_weights[rows, cols] += weights

    def is_complete(self, seed: int, x: int, y: int) -> bool:
        """Whether every cell of a tile has been written to by at least one window.

        A cell with a weight of zero was never written to. Dividing by it gives nonsense rather
        than an error, so nothing else catches it. A tile that is not held at all is not complete.
        """
        held = self._tiles.get((seed, x, y))
        if held is None:
            return False

        return bool(np.all(held[1] > 0))

    def unfilled_count(self, seed: int, x: int, y: int) -> int:
        """How many cells of a tile no window has reached yet."""
        held = self._tiles.get((seed, x, y))
        if held is None:
            return self.tile_height * self.tile_width

        return int(np.count_nonzero(held[1] == 0))

    def heights(self, seed: int, x: int, y: int) -> np.ndarray | None:
        """The finished heights of a tile, or None if that tile has not been generated yet.

        Each cell is its sum divided by its weight. Reading counts as a use of the tile, so a tile
        that is read constantly is never the one dropped.

        Raises if the tile is held but some of it is still unfilled, rather than handing back the
        nan a zero weight would produce.
        """
        key = (seed, x, y)
        held = self._tiles.get(key)
        if held is None:
            return None

        self._tiles.move_to_end(key)
        sums, weights = held

        unfilled = int(np.count_nonzero(weights == 0))
        if unfilled:
            raise ValueError(
                f"{unfilled} of {self.tile_height * self.tile_width} cells of tile {key} have not "
                "been written to yet, so it cannot be read"
            )

        return sums / weights

    def _drop_least_recently_used(self) -> None:
        """Make room by dropping the tile used longest ago, if the store is over its limit.

        A tile is only safe to drop once every window covering it has been added. Dropping one part
        way through loses the windows already added into it. While a request is a single tile that
        cannot happen, because only one tile is being written at a time. It has to be handled once
        windows are split across tile boundaries, see #56.
        """
        if len(self._tiles) > self.capacity:
            self._tiles.popitem(last=False)
