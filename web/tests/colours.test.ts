import { describe, expect, it } from 'vitest'
import { gridColours, heightColour, MAX_HEIGHT, MIN_HEIGHT } from '../src/colours'

describe('heightColour', () => {
  it('gives a different colour to low, middle and high ground', () => {
    const low = heightColour(20)
    const middle = heightColour(140)
    const high = heightColour(240)

    expect(low).not.toEqual(middle)
    expect(middle).not.toEqual(high)
  })

  it.each([
    [-40, MIN_HEIGHT],
    [1000, MAX_HEIGHT],
  ])('pulls %i to the colour of height %i', (outside, end) => {
    expect(heightColour(outside)).toEqual(heightColour(end))
  })

  it('stays within a byte per channel', () => {
    for (let height = MIN_HEIGHT; height <= MAX_HEIGHT; height++) {
      for (const channel of heightColour(height)) {
        expect(channel).toBeGreaterThanOrEqual(0)
        expect(channel).toBeLessThanOrEqual(255)
        expect(Number.isInteger(channel)).toBe(true)
      }
    }
  })
})

describe('gridColours', () => {
  it('gives every cell three bytes, in the order the heights came in', () => {
    const heights = [0, 128, 255]

    const colours = gridColours(heights)

    expect(colours).toHaveLength(heights.length * 3)
    expect([...colours]).toEqual(heights.flatMap(heightColour))
  })
})
