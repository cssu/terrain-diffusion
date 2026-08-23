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

The two jobs are two separate objects on purpose. `RegionGrids` is scratch space that lives only
while one region is being generated, and the sampler is the only thing that writes into it.
`TileCache` holds finished tiles for as long as the program runs, and only Generation
Orchestration touches it. Handing the sampler one object that also held every finished tile would
give it reach over things it has no use for.
"""

from collections import OrderedDict

import numpy as np


class RegionGrids:
    """The running sum and weight grids for one region being generated.

    Windows of terrain overlap, so a cell is usually written by more than one of them. Rather than
    letting the last window win, each contribution is accumulated as a weighted average:

        height[cell] = sum(value * weight) / sum(weight)

    The two halves of that fraction are kept as two grids of their own. Every window adds into
    both, and `heights` divides one by the other once the region is covered.
    """

    def __init__(self, height: int, width: int) -> None:
        """Create zeroed sum and weight grids of the given size."""
        if height < 1 or width < 1:
            raise ValueError(f"region size must be at least 1x1, asked for {height}x{width}")

        self.height = height
        self.width = width
        # float64, so that the division in `heights` is not silently truncated to whole numbers.
        self.sums = np.zeros((height, width))
        self.weights = np.zeros((height, width))

    def add_window(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        row: int,
        col: int,
    ) -> None:
        """Add one window's contribution at the position its top left corner sits at.

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
            raise ValueError(f"window position ({row}, {col}) is outside the region")
        if row + window_height > self.height or col + window_width > self.width:
            raise ValueError(
                f"a {window_height}x{window_width} window at ({row}, {col}) hangs off the edge of "
                f"a {self.height}x{self.width} region"
            )

        rows = slice(row, row + window_height)
        cols = slice(col, col + window_width)
        self.sums[rows, cols] += values * weights
        self.weights[rows, cols] += weights

    def is_complete(self) -> bool:
        """Whether every cell in the region has been written to by at least one window.

        A cell with a weight of zero was never written to. Dividing by it gives nonsense rather
        than an error, so nothing else catches it.
        """
        return bool(np.all(self.weights > 0))

    def unfilled_count(self) -> int:
        """How many cells no window has reached yet."""
        return int(np.count_nonzero(self.weights == 0))

    def heights(self) -> np.ndarray:
        """The finished height grid, each cell being its sum divided by its weight.

        Raises if any cell is still unfilled, rather than handing back the nan a zero weight
        would produce.
        """
        unfilled = self.unfilled_count()
        if unfilled:
            raise ValueError(
                f"{unfilled} of {self.height * self.width} cells have not been written to yet, "
                "so the region cannot be read"
            )

        return self.sums / self.weights


class TileCache:
    """Finished tiles, kept so the same tile is not generated twice.

    Generating a tile is expensive and the same tile is asked for again as the user moves around.
    An endless world has to stay explorable in a fixed amount of memory though, so the cache holds
    a limited number of tiles and drops the least recently used one to make room.

    A tile is identified by its seed, which already has its coordinate worked into it, so tiles
    from two different worlds cannot collide.
    """

    def __init__(self, capacity: int) -> None:
        """Create an empty cache holding at most `capacity` finished tiles."""
        if capacity < 1:
            raise ValueError(f"cache must hold at least one tile, asked for {capacity}")

        self.capacity = capacity
        # Ordered oldest use first, so the tile to drop is the one at the front.
        self._tiles: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        """How many finished tiles are being held."""
        return len(self._tiles)

    def __contains__(self, seed: int) -> bool:
        """Whether a tile is held, without counting as a use of it."""
        return seed in self._tiles

    def get(self, seed: int) -> np.ndarray | None:
        """The finished tile for a seed, or None if it has not been generated yet.

        Reading counts as using the tile, so a tile that is read constantly is never the one
        dropped. The grid handed back is the cache's own, so treat it as read only.
        """
        tile = self._tiles.get(seed)
        if tile is None:
            return None

        self._tiles.move_to_end(seed)
        return tile

    def put(self, seed: int, tile: np.ndarray) -> None:
        """Store a finished tile, dropping the least recently used one if the cache is full.

        The tile is copied on the way in, so whoever generated it can reuse its buffer without
        changing what was cached.
        """
        self._tiles[seed] = np.array(tile)
        self._tiles.move_to_end(seed)

        if len(self._tiles) > self.capacity:
            self._tiles.popitem(last=False)
