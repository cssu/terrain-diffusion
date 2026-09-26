import { NearestFilter } from 'three'
import { expect, it } from 'vitest'
import { heightColour } from '../src/colours'
import { heightTexture, sampleHeights } from '../src/terrain'

it('gives the texture one opaque pixel per cell', () => {
  const heights = [0, 90, 160, 255]

  const { image } = heightTexture(heights, 2, 2)

  expect(image.width).toBe(2)
  expect(image.height).toBe(2)
  expect([...image.data!]).toEqual(heights.flatMap((height) => [...heightColour(height), 255]))
})

it('keeps cell edges sharp instead of blurring between them', () => {
  const texture = heightTexture([0, 255], 2, 1)

  expect(texture.magFilter).toBe(NearestFilter)
  expect(texture.minFilter).toBe(NearestFilter)
})

it('samples a grid that covers the range of heights', () => {
  const heights = sampleHeights(32)

  expect(heights).toHaveLength(32 * 32)
  expect(Math.min(...heights)).toBe(0)
  expect(Math.max(...heights)).toBe(255)
})
