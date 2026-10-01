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
