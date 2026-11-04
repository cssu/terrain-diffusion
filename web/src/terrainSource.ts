import { MAX_HEIGHT, MIN_HEIGHT } from './colours'

export type TerrainRequest = {
  seed: number
  row: number
  column: number
  rows: number
  columns: number
}

export type TerrainGrid = {
  request: TerrainRequest
  heights: Uint8Array
}

const OCTAVES = [
  { spacing: 64, weight: 1 },
  { spacing: 16, weight: 0.4 },
  { spacing: 4, weight: 0.15 },
]

function randomAt(seed: number, row: number, column: number) {
  let hash = seed ^ Math.imul(row, 0x27d4eb2d) ^ Math.imul(column, 0x165667b1)
  hash = Math.imul(hash ^ (hash >>> 15), 0x85ebca6b)
  hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35)

  return ((hash ^ (hash >>> 16)) >>> 0) / 0x100000000
}

function ease(along: number) {
  return along * along * (3 - 2 * along)
}

function noiseAt(seed: number, row: number, column: number, spacing: number) {
  const topRow = Math.floor(row / spacing)
  const leftColumn = Math.floor(column / spacing)
  const downwards = ease(row / spacing - topRow)
  const rightwards = ease(column / spacing - leftColumn)

  const topLeft = randomAt(seed, topRow, leftColumn)
  const topRight = randomAt(seed, topRow, leftColumn + 1)
  const bottomLeft = randomAt(seed, topRow + 1, leftColumn)
  const bottomRight = randomAt(seed, topRow + 1, leftColumn + 1)

  const top = topLeft + (topRight - topLeft) * rightwards
  const bottom = bottomLeft + (bottomRight - bottomLeft) * rightwards

  return top + (bottom - top) * downwards
}

export function heightAt(seed: number, row: number, column: number) {
  let total = 0
  let weights = 0

  for (const { spacing, weight } of OCTAVES) {
    total += noiseAt(seed, row, column, spacing) * weight
    weights += weight
  }

  const height = Math.round((total / weights) * MAX_HEIGHT)

  return Math.min(Math.max(height, MIN_HEIGHT), MAX_HEIGHT)
}

// Stands in for the generation service until #75 answers the same request
export function requestTerrain(request: TerrainRequest): Promise<TerrainGrid> {
  const heights = new Uint8Array(request.rows * request.columns)

  for (let row = 0; row < request.rows; row++) {
    for (let column = 0; column < request.columns; column++) {
      heights[row * request.columns + column] = heightAt(
        request.seed,
        request.row + row,
        request.column + column,
      )
    }
  }

  return Promise.resolve({ request, heights })
}
