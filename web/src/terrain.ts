import { DataTexture, NearestFilter, SRGBColorSpace } from 'three'
import { gridPixels } from './colours'

export function heightTexture(heights: ArrayLike<number>, width: number, height: number) {
  const texture = new DataTexture(gridPixels(heights), width, height)
  texture.magFilter = NearestFilter
  texture.minFilter = NearestFilter
  texture.colorSpace = SRGBColorSpace
  texture.needsUpdate = true

  return texture
}

// Stands in for generated terrain until #85 produces it
export function sampleHeights(size: number) {
  const heights = new Uint8Array(size * size)

  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      heights[row * size + column] = ((row + column) / (2 * (size - 1))) * 255
    }
  }

  return heights
}
