import { expect, it } from 'vitest'
import { MAX_HEIGHT, MIN_HEIGHT } from '../src/colours'
import { requestTerrain, type TerrainRequest } from '../src/terrainSource'

const REGION: TerrainRequest = { seed: 7, row: 0, column: 0, rows: 8, columns: 8 }

it('fills a grid of the size that was asked for', async () => {
  const { heights } = await requestTerrain({ ...REGION, rows: 5, columns: 3 })

  expect(heights).toHaveLength(15)
})

it('keeps every height within the range the viewer draws', async () => {
  const { heights } = await requestTerrain({ ...REGION, rows: 32, columns: 32 })

  for (const height of heights) {
    expect(height).toBeGreaterThanOrEqual(MIN_HEIGHT)
    expect(height).toBeLessThanOrEqual(MAX_HEIGHT)
  }
})

it('agrees about a cell whichever region asked for it', async () => {
  const whole = await requestTerrain({ ...REGION, rows: 8, columns: 8 })
  const right = await requestTerrain({ ...REGION, column: 4, rows: 8, columns: 4 })

  for (let row = 0; row < 8; row++) {
    const fromWhole = whole.heights.slice(row * 8 + 4, row * 8 + 8)
    const fromRight = right.heights.slice(row * 4, row * 4 + 4)
    expect([...fromRight]).toEqual([...fromWhole])
  }
})

it('reads terrain either side of the world origin', async () => {
  const { heights } = await requestTerrain({ ...REGION, row: -6, column: -3 })

  expect(heights).toHaveLength(64)
})

it('gives a different world to a different seed', async () => {
  const one = await requestTerrain(REGION)
  const another = await requestTerrain({ ...REGION, seed: 8 })

  expect([...another.heights]).not.toEqual([...one.heights])
})

it('varies across the grid rather than returning one flat height', async () => {
  const { heights } = await requestTerrain({ ...REGION, rows: 32, columns: 32 })

  expect(new Set(heights).size).toBeGreaterThan(1)
})
